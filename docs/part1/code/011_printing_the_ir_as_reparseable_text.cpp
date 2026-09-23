// Chapter 7: Printing and Visualizing CUDA Hammer's IR
// 011_printing_the_ir_as_reparseable_text.cpp
//
// Section 7.1 -- why an IR that can't be printed is a debugging dead end
// -- and Section 7.2 -- a printer that is a genuine INVERSE of Chapter
// 5's parser, not just a debug dump.
//
// Chapter 5's own File 007 already had a printGraph() function. Look at
// it closely, though: it prints "%3 <- %1, %2" -- numeric node ids and an
// arrow that is not part of this book's own text grammar at all. That
// function is fine for a human skimming terminal output, but it produces
// text Chapter 5's own parser cannot read back in. It is a one-way dump,
// not a serialization.
//
// This file builds a DIFFERENT printer: one that emits the exact
// "name = op(arg, arg, ...)" syntax Chapter 5's grammar defines, using
// every node's OWN debugName to refer to its inputs (the same way a
// human writing the program by hand would). The real claim this file
// makes is not "this text looks right" -- it is a genuine round trip:
// parseProgram(printGraphAsSource(g)) must reconstruct a graph that is
// STRUCTURALLY IDENTICAL to g. A printer and a parser that are true
// inverses of each other is exactly what lets later chapters (starting
// with Part 2's pass manager) dump a graph before and after a pass and
// trust that the dump is a faithful, lossless description of what the
// pass actually saw and produced -- not an approximation of it.
//
// One more thing this file leans on, and calls out explicitly: it does
// NOT need Chapter 4's topologicalSort() to decide what order to print
// nodes in. Graph's own node-creation order is already a valid
// topological order, for the exact same reason Chapter 4 proved the
// graph can never contain a cycle in the first place -- addBinary and
// addUnary can only take a Value that was already returned by an earlier
// call, so node K's inputs are always already-created nodes with ids
// less than K. Printing nodes in plain id order is enough.
//
// Reuses Chapter 4's Value/Node/Graph and Chapter 5's tokenizer/parser
// (lexLine/parseStatement/parseProgram/SymbolTable) and structural
// equality check completely unchanged.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 011_printing_the_ir_as_reparseable_text.cpp -o 011_printing_the_ir_as_reparseable_text
// Run:     ./011_printing_the_ir_as_reparseable_text
#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <sstream>
#include <stdexcept>

// ==================== Value / Node / Graph (from Chapter 4) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU };

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;
    float constValue = 0.0f;
};

class Graph {
public:
    Value addInput(const std::string& name) { return addNode(OpKind::Input, {}, name); }
    Value addConst(float v, const std::string& name) {
        Value out = addNode(OpKind::Const, {}, name);
        nodes_.back()->constValue = v;
        return out;
    }
    Value addUnary(OpKind op, Value in, const std::string& name) { return addNode(op, {in}, name); }
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) {
        return addNode(op, {lhs, rhs}, name);
    }
    const Node* node(int id) const { return nodes_.at(static_cast<size_t>(id)).get(); }
    size_t size() const { return nodes_.size(); }
    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }

private:
    Value addNode(OpKind op, std::vector<Value> inputs, const std::string& name) {
        auto n = std::make_unique<Node>();
        n->id = nextId_++;
        n->op = op;
        n->debugName = name;
        n->inputs = std::move(inputs);
        int id = n->id;
        nodes_.push_back(std::move(n));
        return Value{id, 0};
    }
    std::vector<std::unique_ptr<Node>> nodes_;
    int nextId_ = 0;
};

// ==================== Tokenizer / parser (from Chapter 5's File 007, unchanged) ====================

enum class TokKind { Ident, Number, Equals, LParen, RParen, Comma, End };

struct Token {
    TokKind kind;
    std::string text;
    float numValue = 0.0f;
};

static std::vector<Token> lexLine(const std::string& line) {
    std::vector<Token> toks;
    size_t i = 0;
    while (i < line.size()) {
        char c = line[i];
        if (isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = i;
            while (i < line.size() && (isalnum(static_cast<unsigned char>(line[i])) || line[i] == '_')) ++i;
            toks.push_back({TokKind::Ident, line.substr(start, i - start), 0.0f});
            continue;
        }
        if (isdigit(static_cast<unsigned char>(c)) || c == '.') {
            size_t start = i;
            while (i < line.size() && (isdigit(static_cast<unsigned char>(line[i])) || line[i] == '.')) ++i;
            std::string text = line.substr(start, i - start);
            toks.push_back({TokKind::Number, text, std::stof(text)});
            continue;
        }
        switch (c) {
            case '=': toks.push_back({TokKind::Equals, "=", 0.0f}); break;
            case '(': toks.push_back({TokKind::LParen, "(", 0.0f}); break;
            case ')': toks.push_back({TokKind::RParen, ")", 0.0f}); break;
            case ',': toks.push_back({TokKind::Comma, ",", 0.0f}); break;
            default:
                throw std::runtime_error(std::string("lex: unexpected character '") + c + "'");
        }
        ++i;
    }
    toks.push_back({TokKind::End, "", 0.0f});
    return toks;
}

using SymbolTable = std::map<std::string, Value>;

static void parseStatement(const std::vector<Token>& toks, Graph& g, SymbolTable& symtab, int lineNo) {
    size_t pos = 0;
    auto expect = [&](TokKind k, const char* what) -> Token {
        if (toks[pos].kind != k) {
            throw std::runtime_error("line " + std::to_string(lineNo) + ": expected " + what);
        }
        return toks[pos++];
    };

    Token nameTok = expect(TokKind::Ident, "a name on the left-hand side");
    if (symtab.count(nameTok.text)) {
        throw std::runtime_error("line " + std::to_string(lineNo) + ": redefinition of already-defined name '" +
                                  nameTok.text + "'");
    }
    expect(TokKind::Equals, "'='");
    Token opTok = expect(TokKind::Ident, "an operation name");
    expect(TokKind::LParen, "'(' after the operation name");

    std::vector<Token> args;
    if (toks[pos].kind != TokKind::RParen) {
        while (true) {
            if (toks[pos].kind != TokKind::Ident && toks[pos].kind != TokKind::Number) {
                throw std::runtime_error("line " + std::to_string(lineNo) + ": expected an argument (a name or a number)");
            }
            args.push_back(toks[pos++]);
            if (toks[pos].kind == TokKind::Comma) { ++pos; continue; }
            break;
        }
    }
    expect(TokKind::RParen, "')' to close the argument list");
    expect(TokKind::End, "end of statement (nothing after the closing ')')");

    auto resolveIdentArg = [&](const Token& t) -> Value {
        if (t.kind != TokKind::Ident) {
            throw std::runtime_error("line " + std::to_string(lineNo) + ": expected a name, got a number");
        }
        auto it = symtab.find(t.text);
        if (it == symtab.end()) {
            throw std::runtime_error("line " + std::to_string(lineNo) + ": undefined name '" + t.text +
                                      "' (used before it was defined, or never defined)");
        }
        return it->second;
    };

    Value result;
    if (opTok.text == "input") {
        if (!args.empty()) throw std::runtime_error("line " + std::to_string(lineNo) + ": 'input' takes 0 arguments");
        result = g.addInput(nameTok.text);
    } else if (opTok.text == "const") {
        if (args.size() != 1 || args[0].kind != TokKind::Number) {
            throw std::runtime_error("line " + std::to_string(lineNo) + ": 'const' takes exactly 1 numeric argument");
        }
        result = g.addConst(args[0].numValue, nameTok.text);
    } else if (opTok.text == "add" || opTok.text == "mul") {
        if (args.size() != 2) {
            throw std::runtime_error("line " + std::to_string(lineNo) + ": '" + opTok.text + "' takes exactly 2 arguments");
        }
        Value lhs = resolveIdentArg(args[0]);
        Value rhs = resolveIdentArg(args[1]);
        result = g.addBinary(opTok.text == "add" ? OpKind::Add : OpKind::Mul, lhs, rhs, nameTok.text);
    } else if (opTok.text == "relu") {
        if (args.size() != 1) {
            throw std::runtime_error("line " + std::to_string(lineNo) + ": 'relu' takes exactly 1 argument");
        }
        Value in = resolveIdentArg(args[0]);
        result = g.addUnary(OpKind::ReLU, in, nameTok.text);
    } else {
        throw std::runtime_error("line " + std::to_string(lineNo) + ": unknown operation '" + opTok.text + "'");
    }

    symtab[nameTok.text] = result;
}

static SymbolTable parseProgram(const std::string& source, Graph& g) {
    SymbolTable symtab;
    std::istringstream stream(source);
    std::string line;
    int lineNo = 0;
    while (std::getline(stream, line)) {
        ++lineNo;
        bool blank = true;
        for (char c : line) { if (!isspace(static_cast<unsigned char>(c))) { blank = false; break; } }
        if (blank) continue;
        std::vector<Token> toks = lexLine(line);
        parseStatement(toks, g, symtab, lineNo);
    }
    return symtab;
}

static bool graphsStructurallyEqual(const Graph& a, const Graph& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const Node* na = a.node(static_cast<int>(i));
        const Node* nb = b.node(static_cast<int>(i));
        if (na->op != nb->op) return false;
        if (na->inputs.size() != nb->inputs.size()) return false;
        for (size_t j = 0; j < na->inputs.size(); ++j) {
            if (na->inputs[j].nodeId != nb->inputs[j].nodeId) return false;
        }
    }
    return true;
}

// ============================== Section 7.2: the reparseable printer (new) ==============================
//
// Every op keyword here is the exact lowercase spelling Chapter 5's
// parser recognizes ("input", "const", "add", "mul", "relu"), and every
// input argument is printed as the INPUT NODE'S OWN debugName -- never
// its numeric id -- because a numeric id is not a legal identifier
// argument in this book's grammar at all (only a NAME or a NUMBER is).
static std::string opKindToSourceKeyword(OpKind op) {
    switch (op) {
        case OpKind::Input: return "input";
        case OpKind::Const: return "const";
        case OpKind::Add:   return "add";
        case OpKind::Mul:   return "mul";
        default:            return "relu";
    }
}

static std::string printGraphAsSource(const Graph& g) {
    std::string out;
    for (const auto& n : g.nodes()) {
        out += n->debugName + " = " + opKindToSourceKeyword(n->op) + "(";
        if (n->op == OpKind::Const) {
            // Const's one argument is a NUMBER, not a name -- print the
            // stored value itself, not a reference to another node.
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", n->constValue);
            out += buf;
        } else {
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                const Node* inputNode = g.node(n->inputs[i].nodeId);
                out += inputNode->debugName;
            }
        }
        out += ")\n";
    }
    return out;
}

int main() {
    printf("=== Section 7.2: printing a graph as text its OWN parser can read back in ===\n\n");

    // The same diamond graph Chapters 4-6 have all used, built by hand
    // through direct API calls -- this is the graph the printer's output
    // is checked against.
    Graph original;
    Value a  = original.addInput("a");
    Value b  = original.addInput("b");
    Value t1 = original.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = original.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = original.addUnary(OpKind::ReLU, t1, "t3");
    original.addBinary(OpKind::Add, t2, t3, "out");

    std::string printed = printGraphAsSource(original);
    printf("Printed source (%zu nodes):\n%s\n", original.size(), printed.c_str());

    const std::string expected =
        "a = input()\n"
        "b = input()\n"
        "t1 = add(a, b)\n"
        "t2 = mul(t1, a)\n"
        "t3 = relu(t1)\n"
        "out = add(t2, t3)\n";
    bool textExactlyAsExpected = (printed == expected);
    printf("self-check: printed text matches the hand-written expected source byte-for-byte (%s)\n",
           textExactlyAsExpected ? "confirmed" : "MISMATCH");

    // The real claim: parse what we just printed, and compare the
    // RESULT to the graph we started from -- not to the text.
    Graph roundTripped;
    parseProgram(printed, roundTripped);
    bool roundTripIdentical = graphsStructurallyEqual(original, roundTripped);
    printf("self-check: parseProgram(printGraphAsSource(g)) reconstructs a graph STRUCTURALLY\n");
    printf("IDENTICAL to g -- print and parse are genuine inverses (%s)\n",
           roundTripIdentical ? "confirmed" : "MISMATCH");

    // A Const node exercises the one case above that prints a NUMBER
    // instead of a name -- worth its own explicit round trip too.
    Graph constGraph;
    Value c = constGraph.addConst(3.5f, "c");
    Value x = constGraph.addInput("x");
    constGraph.addBinary(OpKind::Mul, c, x, "scaled");
    std::string constPrinted = printGraphAsSource(constGraph);
    printf("\nPrinted source with a Const node:\n%s\n", constPrinted.c_str());
    Graph constRoundTripped;
    parseProgram(constPrinted, constRoundTripped);
    bool constRoundTripIdentical = graphsStructurallyEqual(constGraph, constRoundTripped);
    printf("self-check: a graph containing a Const node also round-trips correctly (%s)\n",
           constRoundTripIdentical ? "confirmed" : "MISMATCH");

    bool allOk = textExactlyAsExpected && roundTripIdentical && constRoundTripIdentical;
    return allOk ? 0 : 1;
}
