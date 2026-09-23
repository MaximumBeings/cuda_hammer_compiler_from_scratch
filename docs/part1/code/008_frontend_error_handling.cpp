// Chapter 5: Building the Graph: CUDA Hammer's Frontend
// 008_frontend_error_handling.cpp
//
// Section 5.3 -- catching an undefined name, a redefinition, an arity
// mismatch, and an unknown operation, instead of crashing or silently
// building a wrong graph.
//
// Reuses File 007's own Value/Node/Graph classes and its tokenizer and
// parser unchanged. The only new code here is the test harness: four
// deliberately invalid programs, one per failure mode, each fed to the
// SAME parseProgram() File 007 already uses on valid input, confirming
// every one is rejected with a real thrown exception naming the actual
// problem -- and one more control case confirming a VALID program still
// parses cleanly, so the error paths above are not accidentally
// triggering on correct input too.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 008_frontend_error_handling.cpp -o 008_frontend_error_handling
// Run:     ./008_frontend_error_handling
#include <cstdio>
#include <cctype>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <sstream>
#include <stdexcept>

// ==================== Value / Node / Graph (from File 005 / 007) ====================

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
    size_t size() const { return nodes_.size(); }

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

// ==================== Tokenizer / parser (from File 007, unchanged) ====================

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

// ============================== Test harness (new) ==============================

struct TestCase {
    std::string name;
    std::string program;
    bool expectError;
};

int main() {
    printf("=== Section 5.3: rejecting invalid programs instead of crashing or misbuilding ===\n\n");

    std::vector<TestCase> tests = {
        {
            "valid program (control case)",
            "a = input()\nb = input()\nc = add(a, b)\n",
            false
        },
        {
            "undefined name",
            "a = input()\nc = add(a, z)\n",  // z was never defined
            true
        },
        {
            "redefinition of an existing name",
            "a = input()\na = input()\n",    // a defined twice
            true
        },
        {
            "arity mismatch (add takes 2 args, given 1)",
            "a = input()\nc = add(a)\n",
            true
        },
        {
            "unknown operation",
            "a = input()\nb = input()\nc = subtract(a, b)\n",  // subtract is not a real op
            true
        },
    };

    bool allBehavedAsExpected = true;

    for (const auto& tc : tests) {
        printf("--- %s ---\n", tc.name.c_str());
        Graph g;
        bool threw = false;
        std::string errorMessage;
        try {
            parseProgram(tc.program, g);
        } catch (const std::exception& e) {
            threw = true;
            errorMessage = e.what();
        }

        bool behavedAsExpected = (threw == tc.expectError);
        allBehavedAsExpected = allBehavedAsExpected && behavedAsExpected;

        if (threw) {
            printf("  threw: \"%s\"\n", errorMessage.c_str());
        } else {
            printf("  parsed successfully (%zu nodes), no exception thrown\n", g.size());
        }
        printf("  expected %s, got %s -- %s\n\n",
               tc.expectError ? "an exception" : "success",
               threw ? "an exception" : "success",
               behavedAsExpected ? "confirmed" : "MISMATCH");
    }

    printf("self-check: every test case behaved exactly as expected (%s)\n",
           allBehavedAsExpected ? "confirmed" : "MISMATCH");

    return allBehavedAsExpected ? 0 : 1;
}
