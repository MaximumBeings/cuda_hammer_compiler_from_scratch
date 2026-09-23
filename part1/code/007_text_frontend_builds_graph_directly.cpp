// Chapter 5: Building the Graph: CUDA Hammer's Frontend
// 007_text_frontend_builds_graph_directly.cpp
//
// Section 5.1 -- a minimal text format for describing a graph -- and
// Section 5.2 -- parsing straight into a Graph, with no AST stage at all.
//
// Chapter 2's frontend went source text -> tokens -> AST -> (a separate
// lowering pass) -> linear IR, because a general arithmetic expression's
// grammar is recursive (an expression contains sub-expressions, which
// can themselves contain sub-expressions) and a recursive-descent parser
// naturally builds a tree while it works. This chapter's text format is
// deliberately NOT recursive: every statement is a flat assignment,
// "name = op(arg, arg, ...)", where every arg is either a NUMBER or the
// NAME of some already-defined value. Because nothing on a statement's
// right-hand side can itself contain a nested statement, there is no
// tree to build in the first place -- the parser can call straight into
// Graph's own addInput/addConst/addUnary/addBinary the moment it finishes
// reading one statement, and move on. This is a real, honest architecture
// choice a real compiler frontend sometimes makes too (LLVM's own IR text
// format works exactly this way: flat, already-named, no re-parsing into
// a tree needed), not a shortcut invented to simplify this book.
//
// The Value/Node/Graph classes below are copied from Chapter 4's File 005
// unchanged. What's new in this file is the tokenizer, the one-pass
// parser/graph-builder, and a real correctness check: the graph this
// frontend builds by PARSING TEXT is compared, node by node, against the
// exact same diamond graph Chapter 4 built BY HAND through direct API
// calls -- if the frontend is correct, the two must be structurally
// identical, since they describe the same computation.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 007_text_frontend_builds_graph_directly.cpp -o 007_text_frontend_builds_graph_directly
// Run:     ./007_text_frontend_builds_graph_directly
#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <sstream>
#include <stdexcept>

// ==================== Value / Node / Graph (from Chapter 4's File 005) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input: return "Input";
        case OpKind::Const: return "Const";
        case OpKind::Add:   return "Add";
        case OpKind::Mul:   return "Mul";
        default:            return "ReLU";
    }
}

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

// ============================== Tokenizer (new) ==============================
//
// One line, one statement, always: NAME '=' OPNAME '(' [ARG (',' ARG)*] ')'.
// Every right-hand side is a call -- even "input()" and "const(3.5)" take
// the same parenthesized-argument-list shape as "add(a, b)" -- one
// grammar rule, no special case for "ops with no arguments."
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

// ==================== Parser / one-pass graph builder (new) ====================
//
// Reads one statement's tokens and calls straight into Graph -- no AST
// node is ever built for a statement. `symtab` maps every name already
// DEFINED (appeared on some earlier statement's left-hand side) to the
// Value it produced; an arg that is an Ident is resolved through
// `symtab`, never re-parsed as an expression, because this grammar does
// not allow nested expressions in the first place.
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

    // Resolves an Ident argument through the symbol table -- the only
    // place a name can come from is an EARLIER statement's left-hand
    // side, the exact same "can only reference something that already
    // exists" structure Chapter 4's addBinary/addUnary API itself has.
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

// Parses an entire program (one statement per non-blank line) straight
// into `g`, returning the final symbol table so the caller can look any
// defined name back up as a Value.
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

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (!n->inputs.empty()) {
            printf(" <- ");
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) printf(", ");
                printf("%%%d", n->inputs[i].nodeId);
            }
        }
        printf("\n");
    }
}

// Structural equality: same number of nodes, and for every node the same
// op kind and the same input node ids in the same order. This is the
// real correctness claim this file makes -- not "the frontend runs
// without crashing," but "the frontend produces the exact graph a
// programmer building it by hand through the API would have produced."
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

int main() {
    printf("=== Section 5.1 / 5.2: parsing text straight into a Graph, no AST stage ===\n\n");

    const std::string program =
        "a = input()\n"
        "b = input()\n"
        "t1 = add(a, b)\n"
        "t2 = mul(t1, a)\n"
        "t3 = relu(t1)\n"
        "out = add(t2, t3)\n";

    printf("Source program:\n%s\n", program.c_str());

    Graph parsedGraph;
    SymbolTable symtab = parseProgram(program, parsedGraph);

    printf("Graph built by PARSING the text above (%zu nodes):\n", parsedGraph.size());
    printGraph(parsedGraph);

    // Chapter 4's own hand-built diamond, via direct API calls -- the
    // reference this file checks the frontend's own output against.
    Graph handBuiltGraph;
    Value a  = handBuiltGraph.addInput("a");
    Value b  = handBuiltGraph.addInput("b");
    Value t1 = handBuiltGraph.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = handBuiltGraph.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = handBuiltGraph.addUnary(OpKind::ReLU, t1, "t3");
    handBuiltGraph.addBinary(OpKind::Add, t2, t3, "out");

    printf("\nGraph built BY HAND through direct API calls (%zu nodes, from Chapter 4):\n", handBuiltGraph.size());
    printGraph(handBuiltGraph);

    bool structurallyEqual = graphsStructurallyEqual(parsedGraph, handBuiltGraph);
    printf("\nself-check: the parsed graph is structurally IDENTICAL to the hand-built one (%s)\n",
           structurallyEqual ? "confirmed" : "MISMATCH");

    bool symtabResolvesOut = symtab.count("out") == 1 && symtab.at("out").nodeId == 5;
    printf("self-check: the symbol table resolves 'out' to node %%5 (%s)\n",
           symtabResolvesOut ? "confirmed" : "MISMATCH");

    return (structurallyEqual && symtabResolvesOut) ? 0 : 1;
}
