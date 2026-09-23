// Chapter 2: What a Compiler Pipeline Looks Like
// 003_parsing_lowering_and_ir.cpp
//
// Section 2.1 -- source text to an abstract syntax tree -- and
// Section 2.2 -- lowering that tree to a linear intermediate
// representation (IR).
//
// This file implements a real, complete front end for a tiny arithmetic
// expression language (integer literals, single-letter variables, the
// operators + - * /, and parentheses): a lexer, a recursive-descent
// parser that builds a real AST, an AST-to-linear-IR lowering pass (the
// same three-address-code shape every real compiler uses internally),
// and a real IR interpreter. A second, independent evaluator that walks
// the AST directly (never touching the IR at all) is built purely as a
// correctness reference -- every expression's IR-interpreted result is
// checked against this independent AST evaluation, for several
// expressions and several variable bindings each.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 003_parsing_lowering_and_ir.cpp -o 003_parsing_lowering_and_ir
// Run:     ./003_parsing_lowering_and_ir
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>

// ============================== Lexer ==============================

enum class TokKind { Num, Var, Plus, Minus, Star, Slash, LParen, RParen, End };

struct Token {
    TokKind kind;
    int numValue = 0;
    char varName = 0;
};

static std::vector<Token> lex(const std::string& src) {
    std::vector<Token> toks;
    size_t i = 0;
    while (i < src.size()) {
        char c = src[i];
        if (c == ' ') { ++i; continue; }
        if (isdigit(static_cast<unsigned char>(c))) {
            int value = 0;
            while (i < src.size() && isdigit(static_cast<unsigned char>(src[i]))) {
                value = value * 10 + (src[i] - '0');
                ++i;
            }
            toks.push_back({TokKind::Num, value, 0});
            continue;
        }
        if (isalpha(static_cast<unsigned char>(c))) {
            toks.push_back({TokKind::Var, 0, c});
            ++i;
            continue;
        }
        switch (c) {
            case '+': toks.push_back({TokKind::Plus, 0, 0}); break;
            case '-': toks.push_back({TokKind::Minus, 0, 0}); break;
            case '*': toks.push_back({TokKind::Star, 0, 0}); break;
            case '/': toks.push_back({TokKind::Slash, 0, 0}); break;
            case '(': toks.push_back({TokKind::LParen, 0, 0}); break;
            case ')': toks.push_back({TokKind::RParen, 0, 0}); break;
            default: throw std::runtime_error(std::string("lex: unexpected character '") + c + "'");
        }
        ++i;
    }
    toks.push_back({TokKind::End, 0, 0});
    return toks;
}

// ================================ AST ================================

enum class ExprKind { Literal, Variable, BinaryOp };

struct Expr {
    ExprKind kind;
    int literalValue = 0;
    char varName = 0;
    char op = 0;
    std::unique_ptr<Expr> left, right;
};

static std::unique_ptr<Expr> makeLiteral(int v) {
    auto e = std::make_unique<Expr>();
    e->kind = ExprKind::Literal;
    e->literalValue = v;
    return e;
}
static std::unique_ptr<Expr> makeVariable(char name) {
    auto e = std::make_unique<Expr>();
    e->kind = ExprKind::Variable;
    e->varName = name;
    return e;
}
static std::unique_ptr<Expr> makeBinaryOp(char op, std::unique_ptr<Expr> l, std::unique_ptr<Expr> r) {
    auto e = std::make_unique<Expr>();
    e->kind = ExprKind::BinaryOp;
    e->op = op;
    e->left = std::move(l);
    e->right = std::move(r);
    return e;
}

// ============================== Parser ==============================
// Grammar (standard precedence, left-associative):
//   expr   := term (('+' | '-') term)*
//   term   := factor (('*' | '/') factor)*
//   factor := NUM | VAR | '(' expr ')'

struct Parser {
    const std::vector<Token>& toks;
    size_t pos = 0;
    explicit Parser(const std::vector<Token>& t) : toks(t) {}

    const Token& peek() const { return toks[pos]; }
    Token advance() { return toks[pos++]; }

    std::unique_ptr<Expr> parseExpr() {
        auto left = parseTerm();
        while (peek().kind == TokKind::Plus || peek().kind == TokKind::Minus) {
            char op = (advance().kind == TokKind::Plus) ? '+' : '-';
            auto right = parseTerm();
            left = makeBinaryOp(op, std::move(left), std::move(right));
        }
        return left;
    }
    std::unique_ptr<Expr> parseTerm() {
        auto left = parseFactor();
        while (peek().kind == TokKind::Star || peek().kind == TokKind::Slash) {
            char op = (advance().kind == TokKind::Star) ? '*' : '/';
            auto right = parseFactor();
            left = makeBinaryOp(op, std::move(left), std::move(right));
        }
        return left;
    }
    std::unique_ptr<Expr> parseFactor() {
        Token t = peek();
        if (t.kind == TokKind::Num) { advance(); return makeLiteral(t.numValue); }
        if (t.kind == TokKind::Var) { advance(); return makeVariable(t.varName); }
        if (t.kind == TokKind::LParen) {
            advance();
            auto e = parseExpr();
            if (peek().kind != TokKind::RParen) throw std::runtime_error("parse: expected ')'");
            advance();
            return e;
        }
        throw std::runtime_error("parse: unexpected token in factor position");
    }
};

static void printAst(const Expr* e, int indent, std::string& out) {
    std::string pad(static_cast<size_t>(indent) * 2, ' ');
    if (e->kind == ExprKind::Literal) {
        out += pad + std::to_string(e->literalValue) + "\n";
    } else if (e->kind == ExprKind::Variable) {
        out += pad + std::string(1, e->varName) + "\n";
    } else {
        out += pad + std::string(1, e->op) + "\n";
        printAst(e->left.get(), indent + 1, out);
        printAst(e->right.get(), indent + 1, out);
    }
}

// ========================= Linear IR (2.2) =========================

enum class OperandKind { Temp, Literal, Variable };

struct Operand {
    OperandKind kind;
    int tempIndex = -1;
    int literalValue = 0;
    char varName = 0;
};

struct Instruction {
    int destTemp;
    char op;
    Operand lhs, rhs;
};

// Lowers an AST node into the instruction list, returning the Operand
// that holds its value: a leaf (Literal/Variable) needs no instruction
// at all and is returned directly; a BinaryOp emits exactly one
// instruction and returns a reference to its fresh destination temp.
static Operand lower(const Expr* e, std::vector<Instruction>& out, int& nextTemp) {
    if (e->kind == ExprKind::Literal) {
        return Operand{OperandKind::Literal, -1, e->literalValue, 0};
    }
    if (e->kind == ExprKind::Variable) {
        return Operand{OperandKind::Variable, -1, 0, e->varName};
    }
    Operand lhs = lower(e->left.get(), out, nextTemp);
    Operand rhs = lower(e->right.get(), out, nextTemp);
    int dest = nextTemp++;
    out.push_back({dest, e->op, lhs, rhs});
    return Operand{OperandKind::Temp, dest, 0, 0};
}

static std::string operandStr(const Operand& o) {
    if (o.kind == OperandKind::Literal) return std::to_string(o.literalValue);
    if (o.kind == OperandKind::Variable) return std::string(1, o.varName);
    return "t" + std::to_string(o.tempIndex);
}

static void printIr(const std::vector<Instruction>& ir, std::string& out) {
    for (const auto& ins : ir) {
        out += "  t" + std::to_string(ins.destTemp) + " = " + operandStr(ins.lhs) +
               " " + std::string(1, ins.op) + " " + operandStr(ins.rhs) + "\n";
    }
}

// ============================ Interpreter ============================

static int resolveOperand(const Operand& o, const std::vector<int>& temps,
                           const std::map<char, int>& env) {
    if (o.kind == OperandKind::Literal) return o.literalValue;
    if (o.kind == OperandKind::Variable) return env.at(o.varName);
    return temps[o.tempIndex];
}

static int applyOp(char op, int a, int b) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        default:  return a / b;
    }
}

// Interprets the IR in order, returning the final expression's value.
// resultOperand is what lower() returned for the whole expression --
// for a bare literal or variable (zero instructions), it IS the answer.
static int interpretIr(const std::vector<Instruction>& ir, const Operand& resultOperand,
                        const std::map<char, int>& env) {
    std::vector<int> temps(ir.size(), 0);
    for (const auto& ins : ir) {
        int a = resolveOperand(ins.lhs, temps, env);
        int b = resolveOperand(ins.rhs, temps, env);
        temps[ins.destTemp] = applyOp(ins.op, a, b);
    }
    return resolveOperand(resultOperand, temps, env);
}

// Independent reference: evaluates the AST directly, never touching the
// IR at all. Used purely to cross-check the IR interpreter's result.
static int evalAstDirect(const Expr* e, const std::map<char, int>& env) {
    if (e->kind == ExprKind::Literal) return e->literalValue;
    if (e->kind == ExprKind::Variable) return env.at(e->varName);
    int l = evalAstDirect(e->left.get(), env);
    int r = evalAstDirect(e->right.get(), env);
    return applyOp(e->op, l, r);
}

int main() {
    printf("=== Section 2.1 / 2.2: source -> tokens -> AST -> linear IR -> result ===\n\n");

    struct TestCase { std::string source; std::map<char, int> env; };
    std::vector<TestCase> tests = {
        {"a + b * c",        {{'a', 2}, {'b', 3}, {'c', 4}}},
        {"(a + b) * c",      {{'a', 2}, {'b', 3}, {'c', 4}}},
        {"a - b - c",        {{'a', 10}, {'b', 3}, {'c', 2}}},
        {"(a + 2) * (b - 1)", {{'a', 5}, {'b', 4}}},
        {"a / b + c * d",    {{'a', 20}, {'b', 4}, {'c', 3}, {'d', 2}}},
    };

    bool allMatch = true;
    for (const auto& tc : tests) {
        printf("--- source: \"%s\" ---\n", tc.source.c_str());

        std::vector<Token> toks = lex(tc.source);
        Parser parser(toks);
        std::unique_ptr<Expr> ast = parser.parseExpr();

        std::string astText;
        printAst(ast.get(), 1, astText);
        printf("AST:\n%s", astText.c_str());

        std::vector<Instruction> ir;
        int nextTemp = 0;
        Operand result = lower(ast.get(), ir, nextTemp);

        std::string irText;
        printIr(ir, irText);
        if (ir.empty()) irText = "  (no instructions -- expression is a bare literal or variable)\n";
        printf("IR (%zu instruction%s):\n%s", ir.size(), ir.size() == 1 ? "" : "s", irText.c_str());

        int viaIr = interpretIr(ir, result, tc.env);
        int viaAst = evalAstDirect(ast.get(), tc.env);
        bool match = (viaIr == viaAst);
        allMatch = allMatch && match;

        printf("IR-interpreted result: %d | direct-AST-evaluated result: %d | match: %s\n\n",
               viaIr, viaAst, match ? "yes" : "NO");
    }

    printf("self-check: every expression's IR-interpreted result matches its independent\n");
    printf("direct-AST-evaluated result (%s)\n", allMatch ? "confirmed" : "MISMATCH");

    return allMatch ? 0 : 1;
}
