// Chapter 2: What a Compiler Pipeline Looks Like
// 004_constant_folding_pass.cpp
//
// Section 2.3 -- passes: transforming the IR without ever touching
// source text again -- and code generation as "something that actually
// produces a value."
//
// Reuses File 003's own lexer, parser, AST, and lowering pass unchanged
// (a real compiler's later stages never re-derive the earlier ones; this
// file's only new code is the pass itself and its own driver). The pass
// added here is real constant folding with constant propagation: it
// walks the linear IR exactly once, in instruction order, and whenever
// an instruction's operands are BOTH statically known -- either a
// literal directly, or a temp whose own defining instruction was already
// folded earlier in this same pass -- it computes the result at compile
// time and drops the instruction entirely, rewriting every later
// reference to that temp into a literal operand. An instruction with at
// least one genuinely unknown operand (a variable, or a temp that could
// not be folded) is kept, with its own operands rewritten wherever a
// folded constant applies. The pass never looks at source text, the
// token stream, or the AST -- only at the IR data structure itself.
//
// Every test expression is interpreted BOTH before and after folding,
// on the same variable bindings, and the two results are compared -- a
// pass that changes what a program computes is a broken pass, so this
// is the correctness bar folding has to clear, on top of the genuinely
// counted reduction in instruction count folding achieves.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 004_constant_folding_pass.cpp -o 004_constant_folding_pass
// Run:     ./004_constant_folding_pass
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <optional>
#include <stdexcept>

// ==================== Lexer / AST / Parser (from File 003) ====================

enum class TokKind { Num, Var, Plus, Minus, Star, Slash, LParen, RParen, End };
struct Token { TokKind kind; int numValue = 0; char varName = 0; };

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
        if (isalpha(static_cast<unsigned char>(c))) { toks.push_back({TokKind::Var, 0, c}); ++i; continue; }
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

enum class ExprKind { Literal, Variable, BinaryOp };
struct Expr {
    ExprKind kind;
    int literalValue = 0;
    char varName = 0;
    char op = 0;
    std::unique_ptr<Expr> left, right;
};
static std::unique_ptr<Expr> makeLiteral(int v) { auto e = std::make_unique<Expr>(); e->kind = ExprKind::Literal; e->literalValue = v; return e; }
static std::unique_ptr<Expr> makeVariable(char n) { auto e = std::make_unique<Expr>(); e->kind = ExprKind::Variable; e->varName = n; return e; }
static std::unique_ptr<Expr> makeBinaryOp(char op, std::unique_ptr<Expr> l, std::unique_ptr<Expr> r) {
    auto e = std::make_unique<Expr>(); e->kind = ExprKind::BinaryOp; e->op = op; e->left = std::move(l); e->right = std::move(r); return e;
}

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
            left = makeBinaryOp(op, std::move(left), parseTerm());
        }
        return left;
    }
    std::unique_ptr<Expr> parseTerm() {
        auto left = parseFactor();
        while (peek().kind == TokKind::Star || peek().kind == TokKind::Slash) {
            char op = (advance().kind == TokKind::Star) ? '*' : '/';
            left = makeBinaryOp(op, std::move(left), parseFactor());
        }
        return left;
    }
    std::unique_ptr<Expr> parseFactor() {
        Token t = peek();
        if (t.kind == TokKind::Num) { advance(); return makeLiteral(t.numValue); }
        if (t.kind == TokKind::Var) { advance(); return makeVariable(t.varName); }
        if (t.kind == TokKind::LParen) { advance(); auto e = parseExpr(); if (peek().kind != TokKind::RParen) throw std::runtime_error("parse: expected ')'"); advance(); return e; }
        throw std::runtime_error("parse: unexpected token in factor position");
    }
};

enum class OperandKind { Temp, Literal, Variable };
struct Operand { OperandKind kind; int tempIndex = -1; int literalValue = 0; char varName = 0; };
struct Instruction { int destTemp; char op; Operand lhs, rhs; };

static Operand lower(const Expr* e, std::vector<Instruction>& out, int& nextTemp) {
    if (e->kind == ExprKind::Literal) return Operand{OperandKind::Literal, -1, e->literalValue, 0};
    if (e->kind == ExprKind::Variable) return Operand{OperandKind::Variable, -1, 0, e->varName};
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
    for (const auto& ins : ir)
        out += "  t" + std::to_string(ins.destTemp) + " = " + operandStr(ins.lhs) + " " + std::string(1, ins.op) + " " + operandStr(ins.rhs) + "\n";
}
static int resolveOperand(const Operand& o, const std::vector<int>& temps, const std::map<char, int>& env) {
    if (o.kind == OperandKind::Literal) return o.literalValue;
    if (o.kind == OperandKind::Variable) return env.at(o.varName);
    return temps[o.tempIndex];
}
static int applyOp(char op, int a, int b) {
    switch (op) { case '+': return a + b; case '-': return a - b; case '*': return a * b; default: return a / b; }
}
static int interpretIr(const std::vector<Instruction>& ir, const Operand& resultOperand, const std::map<char, int>& env) {
    std::vector<int> temps(ir.size(), 0);
    for (const auto& ins : ir) {
        int a = resolveOperand(ins.lhs, temps, env);
        int b = resolveOperand(ins.rhs, temps, env);
        temps[ins.destTemp] = applyOp(ins.op, a, b);
    }
    return resolveOperand(resultOperand, temps, env);
}

// ========================== Section 2.3: the pass ==========================
//
// Real constant folding with constant propagation, operating ONLY on the
// IR (never the AST, never source text). knownValue[tempIndex] holds the
// compile-time-known value of a temp that has already been folded away;
// a temp not present in that map was kept as a real instruction.
static std::vector<Instruction> foldConstants(const std::vector<Instruction>& ir,
                                               Operand& resultOperand) {
    std::map<int, int> knownValue;
    std::vector<Instruction> folded;

    auto tryResolveStatic = [&](const Operand& o, int& outValue) -> bool {
        if (o.kind == OperandKind::Literal) { outValue = o.literalValue; return true; }
        if (o.kind == OperandKind::Temp) {
            auto it = knownValue.find(o.tempIndex);
            if (it != knownValue.end()) { outValue = it->second; return true; }
        }
        return false; // Variable operands are never statically known.
    };
    auto rewriteOperand = [&](const Operand& o) -> Operand {
        int v;
        if (o.kind == OperandKind::Temp && tryResolveStatic(o, v)) {
            return Operand{OperandKind::Literal, -1, v, 0};
        }
        return o;
    };

    for (const auto& ins : ir) {
        Operand lhs = rewriteOperand(ins.lhs);
        Operand rhs = rewriteOperand(ins.rhs);
        int lv, rv;
        if (tryResolveStatic(lhs, lv) && tryResolveStatic(rhs, rv)) {
            // Both operands statically known -- fold, do not emit an instruction.
            knownValue[ins.destTemp] = applyOp(ins.op, lv, rv);
        } else {
            folded.push_back({ins.destTemp, ins.op, lhs, rhs});
        }
    }

    resultOperand = rewriteOperand(resultOperand);
    return folded;
}

int main() {
    printf("=== Section 2.3: a real constant-folding pass over the IR ===\n\n");

    struct TestCase { std::string source; std::map<char, int> env; };
    std::vector<TestCase> tests = {
        {"a + (2 * 3)",         {{'a', 10}}},
        {"(4 - 1) * a",          {{'a', 5}}},
        {"(2 + 3) * (4 - 1)",    {}},
        {"a + b",                {{'a', 7}, {'b', 8}}},           // nothing foldable
        {"(1 + 1) + (1 + 1) + a", {{'a', 100}}},                   // chained folding
    };

    bool allMatch = true;
    int totalBefore = 0, totalAfter = 0;

    for (const auto& tc : tests) {
        printf("--- source: \"%s\" ---\n", tc.source.c_str());

        std::vector<Token> toks = lex(tc.source);
        Parser parser(toks);
        std::unique_ptr<Expr> ast = parser.parseExpr();

        std::vector<Instruction> ir;
        int nextTemp = 0;
        Operand result = lower(ast.get(), ir, nextTemp);

        std::string beforeText;
        printIr(ir, beforeText);
        if (ir.empty()) beforeText = "  (no instructions)\n";
        printf("IR before folding (%zu instruction%s):\n%s", ir.size(), ir.size() == 1 ? "" : "s", beforeText.c_str());

        Operand foldedResult = result;
        std::vector<Instruction> folded = foldConstants(ir, foldedResult);

        std::string afterText;
        printIr(folded, afterText);
        if (folded.empty()) afterText = "  (no instructions -- fully folded to a constant)\n";
        printf("IR after folding (%zu instruction%s):\n%s", folded.size(), folded.size() == 1 ? "" : "s", afterText.c_str());

        int before = interpretIr(ir, result, tc.env);
        int after = interpretIr(folded, foldedResult, tc.env);
        bool match = (before == after);
        allMatch = allMatch && match;
        totalBefore += static_cast<int>(ir.size());
        totalAfter += static_cast<int>(folded.size());

        printf("result before: %d | result after: %d | match: %s | instructions eliminated: %zu\n\n",
               before, after, match ? "yes" : "NO", ir.size() - folded.size());
    }

    printf("=== totals across all %zu test expressions ===\n", tests.size());
    printf("instructions before folding: %d\n", totalBefore);
    printf("instructions after folding:  %d\n", totalAfter);
    printf("total eliminated: %d (%.1f%% reduction)\n\n", totalBefore - totalAfter,
           100.0 * (totalBefore - totalAfter) / totalBefore);

    printf("self-check: every expression's pre-folding and post-folding result matches\n");
    printf("(%s) -- the pass changed the IR's shape but never its meaning\n",
           allMatch ? "confirmed" : "MISMATCH");

    return allMatch ? 0 : 1;
}
