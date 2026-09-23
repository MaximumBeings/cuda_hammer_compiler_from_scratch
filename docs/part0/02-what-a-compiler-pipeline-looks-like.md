# 2. What a Compiler Pipeline Looks Like

**What you will understand:** the general vocabulary every real compiler is built from -- lexing, parsing, an abstract syntax tree (AST), lowering, a linear intermediate representation (IR), a pass, and code generation -- built here on a tiny arithmetic expression language, genuinely implemented and run, before Part 1 ever starts building CUDA Hammer's own tensor-graph version of the same pipeline.

**What you need to know first:** Chapter 1's argument for why fusing operations matters. Basic C++ (classes, recursion, `std::unique_ptr`). No prior compiler-construction background is assumed -- that is exactly what this chapter builds.

---

Chapter 1 argued that eager execution pays real, countable costs, and that a compiler which sees a whole chain of operations at once can remove them. It never said what a compiler actually *is*, mechanically. This chapter answers that with the smallest complete example that still has every real stage: a compiler for ordinary arithmetic expressions like `(a + b) * c - d`, with integer literals, single-letter variables, and the four usual operators. The language is deliberately not tensors -- that starts in Part 1 -- because the vocabulary this chapter builds (parse, lower, pass, codegen) is exactly the same vocabulary CUDA Hammer uses later, and it's easier to see clearly on expressions any reader can evaluate in their head.

```text
SOURCE TEXT        front end                     middle end                back end
"a + b * c"  -->  LEX  -->  tokens  -->  PARSE  -->  AST  -->  LOWER  -->  linear IR  -->  PASSES  -->  IR  -->  CODEGEN  -->  result

  front end:  turns text into a tree that reflects the language's real
              grammar and precedence (Section 2.1)
  middle end: turns that tree into a linear IR passes can transform
              (Section 2.2), then runs passes purely on the IR itself,
              never touching source text again (Section 2.3)
  back end:   turns the (possibly transformed) IR into something that
              actually produces a value -- an interpreter here, real
              machine code from Chapter 18 onward
```

## 2.1 Source Code to an Abstract Syntax Tree

### Intuition

A sentence like "the dog that chased the cat barked" is ambiguous if you only read it left to right one word at a time -- you need to know which clause modifies which noun to understand who actually barked. A parse tree resolves exactly this kind of ambiguity for a sentence's grammar; an AST does the identical job for a programming language's grammar. `a + b * c` is ambiguous read strictly left to right (compute `a + b` first? or `b * c` first?) until the language's precedence rules are applied, and an AST is precedence made concrete: the multiplication ends up *underneath* the addition in the tree specifically because it has to be computed first.

```text
source: a + b * c   (parsed so that * binds tighter than +)

  +
    a
    *
      b
      c

this indented shape is exactly what File 003's own printAst() prints:
the top-level operator is "+", and its right child is a whole "*"
subtree that must be computed before the "+" can use its result --
precedence made concrete as tree shape, not as a rule applied at
evaluation time.
```

### Background

The code below is a complete front end for the expression language: a lexer that turns source text into a token stream, and a recursive-descent parser that consumes that stream and builds a real AST, respecting the standard precedence (`*` and `/` bind tighter than `+` and `-`) and parentheses. This same file also lowers the AST to a linear IR and interprets it -- that is Section 2.2's own subject, covered together with this section's parsing code because they are, in a real compiler, genuinely two stages of one pipeline running back to back on the same data.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 003_parsing_lowering_and_ir.cpp -o 003_parsing_lowering_and_ir
./003_parsing_lowering_and_ir
```

**Output:**

```text
=== Section 2.1 / 2.2: source -> tokens -> AST -> linear IR -> result ===

--- source: "a + b * c" ---
AST:
  +
    a
    *
      b
      c
IR (2 instructions):
  t0 = b * c
  t1 = a + t0
IR-interpreted result: 14 | direct-AST-evaluated result: 14 | match: yes

--- source: "(a + b) * c" ---
AST:
  *
    +
      a
      b
    c
IR (2 instructions):
  t0 = a + b
  t1 = t0 * c
IR-interpreted result: 20 | direct-AST-evaluated result: 20 | match: yes

--- source: "a - b - c" ---
AST:
  -
    -
      a
      b
    c
IR (2 instructions):
  t0 = a - b
  t1 = t0 - c
IR-interpreted result: 5 | direct-AST-evaluated result: 5 | match: yes

--- source: "(a + 2) * (b - 1)" ---
AST:
  *
    +
      a
      2
    -
      b
      1
IR (3 instructions):
  t0 = a + 2
  t1 = b - 1
  t2 = t0 * t1
IR-interpreted result: 21 | direct-AST-evaluated result: 21 | match: yes

--- source: "a / b + c * d" ---
AST:
  +
    /
      a
      b
    *
      c
      d
IR (3 instructions):
  t0 = a / b
  t1 = c * d
  t2 = t0 + t1
IR-interpreted result: 11 | direct-AST-evaluated result: 11 | match: yes

self-check: every expression's IR-interpreted result matches its independent
direct-AST-evaluated result (confirmed)
```

!!! warning "[COMMON TRAP] Assuming the AST's shape mirrors source text left to right"
    `a - b - c`'s AST has `a - b` as the *left* child of the outer `-`, not `b - c`, because subtraction is left-associative: it must mean `(a - b) - c`, not `a - (b - c)`, and those give different answers whenever the operands differ. The parser's `while` loop in `parseExpr`/`parseTerm` builds exactly this left-leaning shape on purpose -- a parser that instead recursed on the left side would build a right-leaning tree and silently compute the wrong association for any non-commutative operator.

## 2.2 From an AST to a Linear IR

### Intuition

A tree is a natural shape for representing *how* an expression is structured, but it is an awkward shape for a pass to transform mechanically -- there's no single, ordered sequence of "steps" to walk through and rewrite. A linear IR fixes that: it's the same computation, flattened into an ordered list of simple instructions, each one computing exactly one operation and storing its result in a numbered temporary. This is the same flattening a recipe writer does converting "a dish that needs the sauce, which needs the reduction, which needs the stock" into a numbered list of steps a line cook can execute one at a time without needing to hold the whole recipe's structure in their head at once.

```text
AST (indented)          linear IR (three-address code)

  +
    a                -->    t0 = b * c
    *                       t1 = a + t0
      b
      c

each BinaryOp node becomes exactly one instruction with a fresh
destination temp; each Literal or Variable leaf becomes an operand
directly, wherever it's used, with no instruction of its own.
```

### Background

File 003's own `lower()` function (shown in full above, in Section 2.1's background) is this section's real code -- the AST and the linear IR are two views the *same run of the program* produces, not two separate examples. `lower()` recurses over the AST exactly once: a leaf returns an `Operand` directly (no instruction needed, since a literal or a variable is already "computed"), and a `BinaryOp` node first lowers both its children, then emits one `Instruction` naming a fresh destination temp, and returns a reference to that temp as its own operand. This is a real, general lowering rule -- it works unchanged no matter how deeply the AST nests, which is exactly what the `(a + 2) * (b - 1)` and `a / b + c * d` cases in the locked output above demonstrate: three real instructions each, with the correct operand references threading temps from inner subexpressions into the outer ones.

!!! warning "[COMMON TRAP] Assuming every AST node needs its own IR instruction"
    Literal and `Variable` leaves never get an instruction in this lowering -- look at any locked IR block above and count: a 3-node subtree like `a + 2` produces exactly one instruction, `t0 = a + 2`, not three. Giving every AST node its own instruction is a real, common mistake in a first attempt at a lowering pass, and it isn't just wasteful: it also means a later pass has to work harder to see that `a` and `2` were ever related, since they'd be sitting in separate, disconnected instructions instead of appearing directly as one instruction's own operands.

## 2.3 Passes and Codegen: Transforming and Executing the IR

### Intuition

A copy editor revising a manuscript works entirely from the manuscript itself: they don't re-interview the author to check whether a sentence could be tightened, they read the actual words on the page and rewrite them. A compiler pass has exactly this discipline, and it is what makes passes composable: a pass reads the IR, decides what to change, and writes a new (or modified) IR, never reaching back to the source text or the AST that produced it. This chapter's one real pass, constant folding, does the smallest useful version of this: any instruction whose operands are already known at compile time gets computed right then, and the instruction disappears from the program entirely.

```text
before folding (3 instructions):        after folding (0 instructions):
  t0 = 2 + 3                              (fully resolved to the
  t1 = 4 - 1                               literal 15 at compile time --
  t2 = t0 * t1                             nothing left to execute)

the pass never re-parsed "(2 + 3) * (4 - 1)" -- it only ever read and
rewrote the instruction list above, exactly the way an editor revises a
manuscript without re-interviewing the author.
```

### Background

The pass below, `foldConstants`, walks File 003's own IR exactly once, in instruction order. For each instruction it first rewrites both operands (substituting in any already-folded constant from an earlier instruction in the same walk -- this is constant *propagation*, riding along with the folding), then checks whether both are now statically known. If they are, it computes the result immediately and records it in `knownValue`, without emitting an instruction; otherwise it keeps the instruction, with its rewritten operands. This single-pass design is exactly why `(1 + 1) + (1 + 1) + a` folds all the way down to `t3 = 4 + a` in the locked output below, even though the outer addition's own operands (`t0` and `t1`) are themselves temps, not literals, at the moment they're first read: by the time the pass reaches the instruction that uses them, both have already been resolved to known values earlier in the same walk. Every test expression is interpreted both before and after folding on the same variable bindings specifically to check the one property a pass must never violate: it can change how many instructions exist, but never what the program computes.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 004_constant_folding_pass.cpp -o 004_constant_folding_pass
./004_constant_folding_pass
```

**Output:**

```text
=== Section 2.3: a real constant-folding pass over the IR ===

--- source: "a + (2 * 3)" ---
IR before folding (2 instructions):
  t0 = 2 * 3
  t1 = a + t0
IR after folding (1 instruction):
  t1 = a + 6
result before: 16 | result after: 16 | match: yes | instructions eliminated: 1

--- source: "(4 - 1) * a" ---
IR before folding (2 instructions):
  t0 = 4 - 1
  t1 = t0 * a
IR after folding (1 instruction):
  t1 = 3 * a
result before: 15 | result after: 15 | match: yes | instructions eliminated: 1

--- source: "(2 + 3) * (4 - 1)" ---
IR before folding (3 instructions):
  t0 = 2 + 3
  t1 = 4 - 1
  t2 = t0 * t1
IR after folding (0 instructions):
  (no instructions -- fully folded to a constant)
result before: 15 | result after: 15 | match: yes | instructions eliminated: 3

--- source: "a + b" ---
IR before folding (1 instruction):
  t0 = a + b
IR after folding (1 instruction):
  t0 = a + b
result before: 15 | result after: 15 | match: yes | instructions eliminated: 0

--- source: "(1 + 1) + (1 + 1) + a" ---
IR before folding (4 instructions):
  t0 = 1 + 1
  t1 = 1 + 1
  t2 = t0 + t1
  t3 = t2 + a
IR after folding (1 instruction):
  t3 = 4 + a
result before: 104 | result after: 104 | match: yes | instructions eliminated: 3

=== totals across all 5 test expressions ===
instructions before folding: 12
instructions after folding:  4
total eliminated: 8 (66.7% reduction)

self-check: every expression's pre-folding and post-folding result matches
(confirmed) -- the pass changed the IR's shape but never its meaning
```

!!! warning "[COMMON TRAP] Forgetting to rewrite an operand that referenced an already-folded temp"
    `(1 + 1) + (1 + 1) + a`'s third instruction, `t2 = t0 + t1`, reads `t0` and `t1` -- both of which were *already folded away* and no longer exist as real instructions by the time this one is processed. A pass that forgets to rewrite operands referencing folded temps (only removing the folded instructions themselves, without redirecting their readers to the literal value) produces IR that references a temp nothing ever defines -- a real, silent bug that would only surface later, as a crash or a garbage value, when something finally tries to interpret or codegen the broken IR. `rewriteOperand`'s job above exists specifically to prevent this: it runs on every operand, every instruction, before folding is even attempted on that instruction.

## Chapter Summary

A compiler pipeline has three genuine stages, each with its own job and its own data structure. The front end (lexer plus parser) turns source text into an AST whose shape reflects the language's real grammar and precedence -- Section 2.1 built this for arithmetic expressions and showed a locked example where left-associativity, not left-to-right reading order, decides the tree's shape. The middle end lowers that tree into a linear IR (Section 2.2's `lower()`), a flat, ordered instruction list that a pass can walk and rewrite without needing to hold a whole tree's structure in mind, and then runs passes purely on that IR -- Section 2.3's constant-folding pass genuinely eliminated up to 100% of an expression's instructions when every operand was statically known, while a real correctness check (interpreting before and after) confirmed folding never changed what any expression computed. The back end turns the (possibly transformed) IR into something that actually produces a value; here that was a plain interpreter, and from Chapter 18 onward in this book it will be real generated machine code. Every one of these stages, and the vocabulary that names them, carries forward unchanged into Part 1, where CUDA Hammer's own front end builds a tensor graph instead of an arithmetic AST, and its own passes -- starting in Part 2 -- run on that tensor graph's own linear IR exactly the way `foldConstants` ran here.

## Self-Check Questions

1. Why does `a - b - c` need to parse as `(a - b) - c` rather than `a - (b - c)`, and which part of `Parser::parseExpr` is responsible for producing that specific shape?
2. In `lower()`, why does a `Literal` or `Variable` AST node never produce an `Instruction`, while a `BinaryOp` node always produces exactly one?
3. File 003 checks the IR interpreter's result against `evalAstDirect`'s result. What real bug class would this cross-check catch that comparing the IR interpreter's result only against itself (rerun twice) never could?
4. `foldConstants` rewrites an instruction's operands *before* checking whether both are statically known. What would go wrong for `(1 + 1) + (1 + 1) + a`'s third instruction if that order were reversed?
5. `(2 + 3) * (4 - 1)` folds down to zero instructions. What does File 004's own interpreter actually return for an expression whose IR is empty, and why is that the right answer without needing any special-case code?
6. Why is the pass in Section 2.3 described as operating "purely on the IR," and what concrete thing would break this book's own stated pass discipline if `foldConstants` took the original `Expr*` AST as one of its parameters?
7. The chapter-opening diagram shows PASSES sitting between the linear IR and CODEGEN. Could a real compiler run passes before lowering, directly on the AST, instead? What would it lose by doing that, given what Section 2.2 said about why a linear IR exists?
8. Part 1 replaces this chapter's arithmetic AST with a tensor computation graph. Name the three pipeline stages this chapter built (front end, middle end, back end) and, in one sentence each, what CUDA Hammer's own version of each stage will need to produce instead.

## Where We Go Next

Chapter 2 built the general vocabulary -- AST, linear IR, pass, codegen -- on the smallest language that still needed all of it. Chapter 3 closes out Part 0 with a tour of three real, production ML compilers (XLA, TVM, and Triton), reading their own real architecture through exactly this chapter's vocabulary before Part 1 starts building CUDA Hammer's own version of it.

## Worked Solutions

1. Subtraction is not associative: `(a - b) - c` and `a - (b - c)` give different results whenever `b` and `c` differ (e.g. `10 - 3 - 2`: `(10-3)-2 = 5`, but `10-(3-2) = 9`). The real arithmetic convention is left-associativity, and `parseExpr`'s `while` loop produces it directly: each time around the loop, the previously-built `left` subtree (not a freshly parsed term) becomes the left child of the new `BinaryOp`, so the tree grows leftward and the leftmost operation ends up nested deepest -- computed first.
2. A `Literal` or `Variable` already IS a value (or a reference to one) the moment it's encountered -- there is nothing to *compute*, only something to *name*, so `lower()` returns an `Operand` referring to it directly. A `BinaryOp`, by contrast, genuinely computes a new value from its two children's values, and a linear IR represents "compute a new value" as an instruction with a destination -- that's what an `Instruction` *is* in this IR.
3. It would catch a bug that exists identically in both runs of the same code -- for instance, if `lower()` swapped an operator's left and right operands (e.g. building `t0 = c - b` when the source said `b - c`), both the IR interpreter and a second identical run of the IR interpreter would reproduce that same swapped-operand bug and agree with each other while both being wrong. `evalAstDirect` is a genuinely independent code path (different function, different traversal, reads the AST rather than the IR) that would compute `b - c` correctly and catch the mismatch.
4. `t2`'s own operands are `t0` and `t1` -- both already folded away by the time the pass reaches this instruction, so `knownValue` has entries for temp indices 0 and 1 but the `folded` output list has no actual instruction defining them. If the pass checked "are both operands statically known" using the ORIGINAL, un-rewritten operands (which still say `Temp 0` and `Temp 1`) instead of rewriting them to literals first, `tryResolveStatic` would still correctly find them in `knownValue` and fold correctly in this specific case -- but if `t2` had NOT been fully foldable (say, if it were `t2 = t0 + a`), the kept instruction would end up referencing `t0`, a temp that no longer exists anywhere in the folded IR, producing exactly the silent dangling-reference bug the `[COMMON TRAP]` above describes.
5. `interpretIr` builds `temps` sized to the (now empty) instruction list, runs its `for` loop zero times, and immediately calls `resolveOperand(resultOperand, temps, env)` -- and `foldConstants` already rewrote `resultOperand` itself into a `Literal` operand (value 15) before returning, in its own final line (`resultOperand = rewriteOperand(resultOperand);`). `resolveOperand` on a `Literal` operand just returns `literalValue` directly, with no dependency on `temps` or `ir` at all -- so the empty-instruction-list case falls out of the exact same code path as every other case, with no special-casing needed, because the *result operand itself*, not just the instruction list, was a first-class thing the pass tracked and rewrote.
6. "Purely on the IR" means the pass's entire decision process -- what to fold, what to keep, how to rewrite an operand -- reads and writes nothing but `Instruction`/`Operand` values; it has no access to variable names' original source positions, no access to the AST's tree shape, nothing but the flat instruction list. If `foldConstants` took the `Expr*` AST as a parameter, nothing forces it to actually ignore that parameter -- the moment a real implementation used it (even just to double check an operator), the pass would no longer be reusable on IR that was never produced by lowering an AST (hand-written IR, or IR another pass already transformed), which is exactly the composability a real pass pipeline (Chapter 8 onward) depends on.
7. A pass could run on the AST directly, but it would lose exactly what Section 2.2's intuition named: an ordered, flat sequence to walk mechanically. Constant folding on a tree means writing a recursive tree-rewriting function that has to reconstruct new subtrees and handle each node kind's own recursion by hand; on a linear IR it's a single `for` loop reading previous results left to right, which is why every real compiler lowers before it starts running most of its optimization passes.
8. Front end: instead of parsing arithmetic text into an AST, CUDA Hammer's frontend (Chapter 5) will need to produce a tensor computation GRAPH from a small builder API -- nodes are tensor operations, not arithmetic operators, and there's no source text to lex at all. Middle end: instead of `lower()`'s three-address arithmetic IR, CUDA Hammer's own IR (Chapters 4-7) represents each node's tensor shape and dtype alongside the operation, and its own passes (Part 2 onward) will include fusion, not just constant folding. Back end: instead of a simple interpreter, CUDA Hammer's codegen (Part 4) will need to emit real CUDA C++ or real vectorized CPU code that a separate compiler (`nvcc` or `g++`) then compiles -- "codegen" here means emitting compilable source, not directly producing a numeric answer.

---

**Sources cited in this chapter:** none -- this chapter builds general compiler-construction vocabulary from first principles, using only this book's own genuinely compiled and run code as evidence.
