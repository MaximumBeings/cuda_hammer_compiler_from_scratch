# 5. Building the Graph: CUDA Hammer's Frontend

**What you will understand:** a minimal text format for describing a tensor computation, and how CUDA Hammer's own frontend parses it straight into the `Value`/`Node`/`Graph` structures Chapter 4 built -- with no AST stage in between at all. This is a real, defensible architecture choice, not a shortcut: whether a frontend needs a tree stage depends on whether its *source grammar* is recursive, and this chapter's grammar deliberately is not.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph` classes (reused here unchanged) and Chapter 2's vocabulary (lexer, parser, symbol resolution).

---

```text
CHAPTER 2's FRONTEND (a recursive grammar needs a tree in between):
  text --> tokens --> AST --> lower() --> linear IR

CHAPTER 5's FRONTEND (a flat, already-named grammar needs no tree at all):
  text --> tokens --> Graph              (built directly, one statement at a time)
```

## 5.1 A Minimal Text Format for Describing a Graph

### Intuition

A shipping manifest lists items one per line, and every line that refers to an earlier container gives its ID directly: "pack container C7 into container C12" never says "pack whatever was packed three lines ago into whatever gets packed two lines from now." A warehouse worker reads the manifest top to bottom, processes one line at a time, and never needs to hold the whole shipment's structure in their head before starting -- every reference the worker needs is already a concrete, already-assigned ID by the time they reach it. Chapter 2's arithmetic expressions were not like this: `(a + b) * c` names no intermediate result at all, so a parser has to discover the whole nested structure (build a tree) before it can decide what to compute first. CUDA Hammer's own text format is written like the manifest instead.

```text
CHAPTER 2's GRAMMAR (recursive -- an expression contains expressions):
  (a + b) * c
  parsing this REQUIRES discovering nesting depth before anything can be
  evaluated -- there is no way to know "compute a+b first" without first
  building (or at least walking) the whole parenthesized structure.

CHAPTER 5's GRAMMAR (flat -- one named result per line, always):
  t1 = add(a, b)
  out = mul(t1, c)
  every right-hand side names its OWN inputs directly by an already-
  assigned name -- nothing here is nested inside anything else.
```

### Background

Every statement in CUDA Hammer's own text format has the exact same shape: `name = opname(arg, arg, ...)`, with no exceptions -- even `input()` and `const(3.5)` take the same parenthesized argument list `add(a, b)` does, just with zero or one argument instead of two. Reading the grammar out loud makes the whole format explicit: a statement is a *name*, then `=`, then an *operation name*, then an *argument list* wrapped in parentheses, where an argument is either another already-defined name or a plain number, and arguments are separated by commas. That is the entire grammar -- five kinds of thing to recognize (a name, `=`, an operation name, parentheses, an argument list), combined exactly one way, every single time. This uniformity is a deliberate grammar choice, not an accident: a grammar with one rule and no special cases ("ops with zero arguments work differently from ops with two") is genuinely easier to parse correctly and easier for a reader to learn, and CUDA Hammer's own text format has no real cost to paying for that uniformity, since `input()` reads barely differently from a bare `input`.

Before any of that structure can be recognized, the raw characters of one line have to be split into meaningful chunks -- the same job Chapter 2's `lex()` did for arithmetic expressions, just simplified here, since this format never needs to recognize `+`, `-`, `*`, `/`, or parentheses-as-grouping the way arithmetic expression text does. `lexLine()` walks one line's characters left to right, classifying each run of characters into exactly one of six token kinds:

```text
TOKENIZING ONE LINE: "t2 = mul(t1, a)"

  input line, left to right:
    t2 = mul(t1, a)

  read as a flat sequence of tokens, one per meaningful chunk (every
  space between chunks is skipped, never turned into a token of its own):

    1. Ident  "t2"     -- the name being defined
    2. Equals "="
    3. Ident  "mul"    -- the operation name
    4. LParen "("
    5. Ident  "t1"     -- first argument
    6. Comma  ","
    7. Ident  "a"      -- second argument
    8. RParen ")"
    9. End             -- a marker confirming nothing trails the ')'

  the parser below reads this same flat sequence left to right, exactly
  once, never needing to look more than one token ahead.
```

Six kinds of token is all this format ever needs: `Ident` (a name, whether it's the thing being defined, an operation name, or an argument), `Number`, `=`, `(`, `)`, and `,`, plus an `End` marker the parser uses to confirm nothing trails the closing `)`. `lexLine()` reuses the exact same character-classification approach Chapter 2's own `lex()` used -- `isalpha`/`isalnum` for the start and continuation of a name, `isdigit`/`.` for a number -- reading one character at a time and deciding, from that single character, which kind of token it's the start of.

!!! warning "[COMMON TRAP] Assuming a frontend must always build an AST"
    It's tempting to think every real compiler frontend needs a tree stage, since Chapter 2's did. Whether a tree is needed depends entirely on whether the *source grammar itself* is recursive -- Chapter 2's arithmetic expressions are (an expression contains sub-expressions), so a recursive-descent parser naturally builds a tree while discovering that nesting. This chapter's own grammar is flat by design: every statement names its own inputs directly, by an already-assigned name, so there is no nesting to discover in the first place. This is not unique to a toy book format either -- LLVM's own real IR text format works exactly this way, one flat, already-named instruction per line, parsed directly into LLVM's real IR with no separate tree stage. A flat grammar is not a "simpler, lesser" kind of frontend; it is a real, legitimate design a real production compiler's own text format actually uses.

## 5.2 Parsing Directly Into a Graph, No AST Required

### Intuition

Assembling furniture from instructions that say "attach part C to the frame you built in step 2" works because step 2 already happened -- the instructions never ask you to attach something you haven't built yet. A symbol table plays exactly this role for CUDA Hammer's frontend: it is the running record of "which names have already been built," and every statement's own arguments are looked up in that record the instant they're needed, never guessed at or resolved later. Because the grammar only allows referencing a name from some *earlier* statement, the parser can call straight into `Graph::addBinary`/`addUnary`/`addInput`/`addConst` the moment it finishes reading one line, with nothing left pending and no tree to assemble first.

```text
PARSING "t2 = mul(t1, a)", one statement, straight into the graph:

  1. read "t2"                     -- the name this statement defines
  2. read "mul"                    -- the operation
  3. read "(t1, a)"                -- the arguments, as names
  4. look up "t1" in the symbol table --> Value{nodeId: 2}
  5. look up "a"  in the symbol table --> Value{nodeId: 0}
  6. call g.addBinary(Mul, t1_value, a_value, "t2")  -- DONE, right here
     no AST node was ever built for this statement -- there was nothing
     left to discover once steps 1 through 5 finished.
```

### Background

File 007 below reuses Chapter 4's own `Value`/`Node`/`Graph` classes without any change, and adds exactly two new things: the tokenizer from Section 5.1, and `parseStatement()`/`parseProgram()`, which read one statement's tokens and call directly into `Graph`'s own `addInput`/`addConst`/`addUnary`/`addBinary`. `parseProgram()` itself does very little: it splits the source text into lines, skips any blank line, and hands each remaining line's tokens to `parseStatement()` in order, top to bottom -- there is no backtracking, no lookahead past the current statement, and nothing saved between one statement and the next except the running `SymbolTable`. That table (`std::map<std::string, Value>`) is the whole mechanism that makes one-pass parsing possible here: every name the parser has already defined is in it, and every name it has not yet seen is absent from it, which is exactly the information `parseStatement()` needs to resolve an argument or reject one.

Walking the diamond program statement by statement makes this concrete -- the symbol table only ever grows, one entry at a time, and every argument any statement reads is already sitting in it by the time that statement is parsed:

```text
SYMBOL TABLE, GROWING ONE STATEMENT AT A TIME:

  before parsing anything:              symtab = {}

  "a = input()"        -->  a resolves to no args (Input takes none)
                             symtab = {a}

  "b = input()"        -->  symtab = {a, b}

  "t1 = add(a, b)"      -->  a, b looked up in symtab -- BOTH already there
                             symtab = {a, b, t1}

  "t2 = mul(t1, a)"     -->  t1, a looked up -- BOTH already there
                             symtab = {a, b, t1, t2}

  "t3 = relu(t1)"       -->  t1 looked up -- already there
                             symtab = {a, b, t1, t2, t3}

  "out = add(t2, t3)"   -->  t2, t3 looked up -- BOTH already there
                             symtab = {a, b, t1, t2, t3, out}

  every single lookup above succeeds on the FIRST attempt, because the
  grammar guarantees an argument's name was already added to symtab by
  some earlier statement -- there is no statement in this program, or
  any program this grammar can express, where a lookup could need to
  "wait" for a name that hasn't been defined yet.
```

Because every lookup this way is a single, immediate map access -- never a search, never a retry, never something deferred until later -- `parseStatement()` can call straight into `Graph::addBinary`/`addUnary`/`addInput`/`addConst` the moment it finishes reading one statement's tokens, exactly the same "can only reference something that already exists" structure Chapter 4's own `addBinary`/`addUnary` API already has at the C++ level, just enforced one level up, at the text-parsing level instead. The real correctness claim File 007 makes is not merely "the parser runs without crashing" -- it is that parsing the diamond program above produces a graph that is *structurally identical*, node for node, edge for edge, to the exact graph Chapter 4 built by hand through direct API calls, checked by `graphsStructurallyEqual()` comparing every node's op kind and every input's node id in order.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 007_text_frontend_builds_graph_directly.cpp -o 007_text_frontend_builds_graph_directly
./007_text_frontend_builds_graph_directly
```

**Output:**

```text
=== Section 5.1 / 5.2: parsing text straight into a Graph, no AST stage ===

Source program:
a = input()
b = input()
t1 = add(a, b)
t2 = mul(t1, a)
t3 = relu(t1)
out = add(t2, t3)

Graph built by PARSING the text above (6 nodes):
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1) <- %0, %1
  %3 = Mul(t2) <- %2, %0
  %4 = ReLU(t3) <- %2
  %5 = Add(out) <- %3, %4

Graph built BY HAND through direct API calls (6 nodes, from Chapter 4):
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1) <- %0, %1
  %3 = Mul(t2) <- %2, %0
  %4 = ReLU(t3) <- %2
  %5 = Add(out) <- %3, %4

self-check: the parsed graph is structurally IDENTICAL to the hand-built one (confirmed)
self-check: the symbol table resolves 'out' to node %5 (confirmed)
```

!!! warning "[COMMON TRAP] Assuming this one-pass approach would work for ANY grammar"
    Resolving an identifier through a single symbol-table lookup, the instant it's read, only works because this grammar guarantees a name can never be referenced before it's defined -- there is no forward reference, and certainly no mutual recursion between two statements, either of which would need a name to resolve to something that does not exist yet at parse time. A real language that allows forward references (two functions calling each other, defined in either order) needs a different strategy entirely -- typically two passes, one to collect every name that will eventually exist, a second to actually resolve arguments against that complete set. This chapter's one-pass approach is correct specifically because CUDA Hammer's own text format was designed to rule forward references out, not because one-pass parsing is always sufficient.

## 5.3 Symbol Table Errors: Catching a Broken Program Before It Runs

### Intuition

A form with a required field left blank, or the same reference number filled in twice by mistake, is caught by whoever processes it *before* it gets filed -- not after, when tracing the mistake back would mean unwinding everything that was built on top of it. CUDA Hammer's own frontend applies the same discipline: an undefined name, a redefined name, or the wrong number of arguments for an operation is caught the moment the offending statement is parsed, before a single wrong node is ever added to the graph -- never discovered later by some downstream pass tripping over a graph that was already built wrong.

```text
A FORM PROCESSED CAREFULLY (checked before filing):
  read field --> valid? --> yes --> file it
                         --> no  --> REJECT, explain why, stop here

CUDA HAMMER'S FRONTEND (checked before adding to the graph):
  read statement --> name already used? --> reject: "redefinition of ..."
                  --> arg name undefined? --> reject: "undefined name ..."
                  --> wrong arg count?    --> reject: "takes exactly N ..."
                  --> otherwise: add the node, remember the name, continue
```

### Background

Look back at `parseStatement()` in File 007: every one of its checks -- has this name already been defined, does this name resolve to something in `symtab`, does this operation's argument count match what it expects -- was already there, sitting right next to the code path that succeeds. Section 5.3 does not add new parsing logic; it proves those checks actually fire, and fire *only* when they should, by deliberately feeding the parser input designed to trip each one. Four different ways a program can be wrong are tested, each isolated so it is clear which specific check catches it:

```text
FOUR WAYS A PROGRAM CAN BE WRONG, AND WHERE EACH IS CAUGHT:

  "c = add(a, z)"        z was never defined anywhere earlier
                         --> caught by the symbol-table LOOKUP in
                             resolveIdentArg(): z is absent from symtab

  "a = input()"           the name "a" is being defined a SECOND time
  "a = input()"          --> caught by the symbol-table MEMBERSHIP check
                             at the top of parseStatement(): a is
                             already present in symtab

  "c = add(a)"            add expects exactly 2 arguments, got 1
                         --> caught by the ARITY check inside the
                             "add"/"mul" branch: args.size() != 2

  "c = subtract(a, b)"    "subtract" is not one of this format's five
                          real operations (input, const, add, mul, relu)
                         --> caught by the final "else" branch: no
                             opTok.text matched any known operation name
```

Each of these is a different KIND of mistake -- a bad reference, a bad name, a bad argument count, a bad operation -- and each is caught by a different, specific piece of `parseStatement()`, which is exactly why File 008 tests them as four separate cases rather than one: a single combined test could pass for the wrong reason (any one of the four checks firing would make it pass), where four separate tests confirm each check individually does its own job. File 008 below reuses File 007's own tokenizer and parser completely unchanged, feeding each of these four deliberately invalid programs to the same `parseProgram()` File 007 already uses on valid input. Each one is expected to throw a `std::runtime_error` naming the actual problem, and a fifth, valid control program is included specifically to confirm none of these checks accidentally reject correct input too -- the same "prove the negative case AND the positive case" discipline Chapter 4's own cycle-detection test used.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 008_frontend_error_handling.cpp -o 008_frontend_error_handling
./008_frontend_error_handling
```

**Output:**

```text
=== Section 5.3: rejecting invalid programs instead of crashing or misbuilding ===

--- valid program (control case) ---
  parsed successfully (3 nodes), no exception thrown
  expected success, got success -- confirmed

--- undefined name ---
  threw: "line 2: undefined name 'z' (used before it was defined, or never defined)"
  expected an exception, got an exception -- confirmed

--- redefinition of an existing name ---
  threw: "line 2: redefinition of already-defined name 'a'"
  expected an exception, got an exception -- confirmed

--- arity mismatch (add takes 2 args, given 1) ---
  threw: "line 2: 'add' takes exactly 2 arguments"
  expected an exception, got an exception -- confirmed

--- unknown operation ---
  threw: "line 3: unknown operation 'subtract'"
  expected an exception, got an exception -- confirmed

self-check: every test case behaved exactly as expected (confirmed)
```

!!! warning "[COMMON TRAP] Assuming redefinition should be allowed for convenience"
    It's tempting to think rejecting `a = input()` followed by a second `a = input()` is needlessly strict -- plenty of real languages let a variable be reassigned. CUDA Hammer's frontend deliberately does not, and the reason is not pickiness: once a name is only ever allowed to mean one specific `Value` for its entire lifetime, every later piece of code that looks a name up (a pass, a printer, a future frontend feature) can trust that lookup never changes meaning out from under it. This is exactly the same single-assignment discipline real compilers rely on internally (it is, not coincidentally, the same idea behind SSA form, which Chapter 9 touches again when it studies real passes) -- CUDA Hammer's frontend just enforces it one level up, at parse time, instead of needing a later pass to establish it.

## Chapter Summary

CUDA Hammer's own frontend parses a minimal, flat text format straight into a `Graph`, with no AST stage in between -- a real, deliberate architecture choice that follows directly from the source grammar itself being flat (every statement names its own inputs by an already-assigned name) rather than recursive (the way Chapter 2's arithmetic expressions were), the same shape LLVM's own real IR text format uses for exactly the same reason. File 007 built a tokenizer and a one-pass `parseStatement()`/`parseProgram()` that call straight into Chapter 4's own `addInput`/`addConst`/`addUnary`/`addBinary`, resolving every identifier argument through a symbol table the instant it's read -- and proved correctness the strongest way available here: parsing Chapter 4's own diamond program produces a graph that is structurally identical, node for node and edge for edge, to the graph Chapter 4 built by hand. File 008 proved the frontend's error handling works both ways: four deliberately invalid programs (an undefined name, a redefinition, an arity mismatch, an unknown operation) are each correctly rejected with a specific, named error, while a fifth, valid control program confirms those same checks don't misfire on correct input. Chapter 6 builds directly on this frontend: once a graph can be built from text, the next real question is whether that graph is actually a *valid tensor computation* -- do the shapes involved actually work out for `add` and `mul` -- which is what shape inference and graph validation answer.

## Self-Check Questions

1. Why does Chapter 2's arithmetic-expression grammar require a tree stage, while this chapter's own grammar does not?
2. Every operation in this chapter's text format takes a parenthesized argument list, including `input()`, which takes none. What is the actual argument for choosing this uniform grammar over a special-cased one where `input` (no parentheses) parses differently from `add(a, b)`?
3. What role does `SymbolTable` play in making one-pass parsing possible here, and what specific guarantee about the grammar does that role depend on?
4. Per Section 5.2's [COMMON TRAP], what would break if this chapter's text format allowed one function to call another function defined later in the same file?
5. File 007's own correctness check is not "the parser runs without crashing." What is the actual claim `graphsStructurallyEqual()` checks, and why is that a stronger claim?
6. File 008 includes one valid program among its four invalid ones. What would be lost if that valid control case were removed and only the four invalid programs were tested?
7. Per Section 5.3's [COMMON TRAP], what concrete benefit does forbidding redefinition (`a = input()` twice) actually buy, beyond just being stricter?
8. If `parseStatement()` resolved an identifier argument by searching some other structure -- say, scanning back through the whole token stream for the most recent statement that defined that name -- instead of a `SymbolTable` lookup, would File 007's own structural-equality self-check still be capable of catching a bug in that alternate implementation? Why or why not?

## Where We Go Next

This chapter's frontend can turn a small, flat program into a real `Graph` -- but nothing about parsing checks whether that graph is actually a *valid tensor computation*. Chapter 6 builds graph validation and shape inference: given a graph of `Input`/`Const`/`Add`/`Mul`/`ReLU` nodes, does every operation's actual inputs make sense together, and what shape does every node's own output have. That is a different kind of correctness question than anything this chapter asked, and it is the last piece Part 1 needs before Part 2 starts writing real optimization passes over these graphs.

## Worked Solutions

1. Chapter 2's grammar is recursive: an expression like `(a + b) * c` contains sub-expressions as its own operands, so a parser has to discover that nesting structure before it can decide what to compute in what order, and a tree is the natural structure that nesting discovery produces. This chapter's grammar is flat: every statement's right-hand side names its own inputs directly, by an already-assigned name (`mul(t1, a)`, never `mul(add(a, b), a)`), so there is no nesting to discover -- each statement is already exactly as "unpacked" as Chapter 2's own linear IR was, just written that way in the source text itself instead of produced by a lowering pass.
2. A grammar with one rule and no exceptions is easier to implement correctly (the parser has exactly one code path for "read an operation's argument list," not two) and easier for a person reading the format to learn (every statement has the same shape, full stop). The cost of that uniformity is close to zero here -- `input()` reads barely differently from a bare `input` -- so there is no real trade-off being made, just an avoided special case.
3. `SymbolTable` maps every name that has already been defined (appeared on some earlier statement's left-hand side) to the `Value` it produced, letting an identifier argument be resolved with a single map lookup the instant it's read. That works only because the grammar guarantees a name can never be referenced before some earlier statement defines it -- if the grammar allowed forward references, a lookup performed "the instant it's read" could fail even for a name that will eventually be valid, simply because that name's own defining statement has not been parsed yet.
4. A one-pass parser that resolves each name via a single symbol-table lookup, in the exact order statements are read, would fail to resolve a call to a function defined later in the same file, because that name would not exist in the symbol table yet at the point it's referenced. Supporting that would require a different strategy -- typically two passes, one to collect every name that will eventually exist before resolving anything, a second to actually resolve arguments against that now-complete set.
5. The claim is not "no exception was thrown" -- it is that the graph produced by parsing text is *structurally identical* to a graph built by an entirely different method (direct API calls), node for node (same op kind at each position) and edge for edge (same input node ids, in the same order). That is a much stronger claim than "didn't crash," because a frontend could easily run to completion without throwing and still silently build the wrong graph (wrong op, swapped arguments, wrong node count) -- `graphsStructurallyEqual()` would catch exactly that kind of bug, where merely not crashing would not.
6. Without the valid control case, a bug that made every one of `parseStatement()`'s checks fire unconditionally -- rejecting every program, valid or not -- would make all four "invalid program" tests pass for entirely the wrong reason (they'd all throw, as expected, but so would a perfectly correct program). The control case is what rules that specific failure mode out: it proves the error paths are actually discriminating between valid and invalid input, not just always throwing.
7. Once a name is guaranteed to mean exactly one `Value` for its entire lifetime, any later code that looks that name up -- a pass, a printer, a future frontend feature -- never has to ask "which definition of this name is currently active" the way code working with a reassignable variable would. That guarantee is established once, at parse time, by forbidding redefinition outright, rather than needing every later consumer of the symbol table to independently track which assignment is "current."
8. Yes -- `graphsStructurallyEqual()` only inspects the two finished `Graph` objects' own nodes and edges; it has no knowledge of, and does not care, how either graph was built. A bug in an alternate identifier-resolution strategy (scanning the token stream instead of using a map) that resolved an argument to the *wrong* node -- pointing `t2`'s second input at the wrong earlier value, for instance -- would produce a graph whose edges genuinely differ from the hand-built reference graph's edges, which is exactly the mismatch `graphsStructurallyEqual()` is built to detect, regardless of what internal mechanism produced that wrong graph.

---

**Sources cited in this chapter:** none -- CUDA Hammer's own text format and frontend are original to this book. The observation that LLVM's real IR text format is similarly flat and parsed with no separate AST stage is a general, well-known architectural fact about LLVM, not a specific claim requiring its own citation the way Chapter 3's claims about XLA, TVM, and Triton did.
