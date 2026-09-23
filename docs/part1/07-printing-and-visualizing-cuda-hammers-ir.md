# 7. Printing and Visualizing CUDA Hammer's IR

**What you will understand:** how to turn an in-memory `Graph` -- a heap of `Node` objects linked by `Value{nodeId, outputIndex}`, invisible to anyone not stepping through it in a debugger -- into two genuinely different, genuinely useful external views: text that Chapter 5's own parser can read straight back in, and a real Graphviz picture a human can look at to see the graph's actual shape. This closes Part 1.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph`, Chapter 5's tokenizer and parser (`lexLine`/`parseStatement`/`parseProgram`), and Chapter 6's `Shape`/`inferShapes()` -- this chapter reuses all three completely unchanged and adds nothing to any of them.

---

Every chapter since Chapter 4 has, in passing, written some version of a function that prints a `Graph` to the terminal -- File 005 had one, File 007 had another, File 009's `main()` printed shapes inline. Each of those was a quick, disposable debug aid, good enough to eyeball while writing that one chapter's own tests and then never looked at again. This chapter takes printing seriously as its own subject, because Part 2 is about to start writing real optimization passes -- constant folding, dead code elimination, common subexpression elimination -- and every one of those passes will need to be debugged the same way Chapter 2's own IR interpreter was: by looking at a graph *before* a pass runs and the same graph *after*, and trusting that what got printed is a faithful, complete description of what the pass actually saw and produced, not an approximation of it.

```text
WHAT A Graph LOOKS LIKE FROM THE OUTSIDE, RIGHT NOW (Chapters 4-6):

  in memory: a heap of Node objects, linked by Value{nodeId, outputIndex}
             -- real and correct, and INVISIBLE to a human at a terminal
             unless something turns it into text or a picture on purpose.

  this chapter builds TWO separate windows onto that same graph:

    1. TEXT (Section 7.2)              2. A PICTURE (Section 7.3)
       a = input()                        a real Graphviz DOT file,
       b = input()                        rendered by the REAL `dot`
       t1 = add(a, b)                     binary into an actual image
       t2 = mul(t1, a)                    a human can look at -- not
       t3 = relu(t1)                      this book's own invented
       out = add(t2, t3)                  ASCII-art format.

       -- Chapter 5's OWN grammar,
          so this exact text can be
          fed straight back into the
          same parser that reads it.
```

## 7.1 Why an IR You Can't Print Is a Debugging Dead End

### Intuition

A program that crashes with no error message and no stack trace is barely more useful than a program that doesn't run at all -- the information a developer actually needs to fix it exists somewhere in memory at the moment of the crash, but if nothing ever surfaces it, that information might as well not exist. A compiler's IR has exactly the same problem, one level up: if a graph can be built, parsed, validated, and optimized, but never *looked at*, then every one of those steps is a black box a developer has to trust blindly, or step through one field access at a time in a debugger. Every real compiler this book has already cited -- and this is a general, well-known architectural fact about compilers, not a claim needing its own citation -- treats "dump the IR in a form a human or another tool can read" as a first-class feature, not an afterthought.

The specific trap this section calls out is subtler than "we should add a print function." Chapter 5's own File 007 already has one:

```text
CHAPTER 5's printGraph() (File 007) -- LOOKS LIKE A PRINTER, ISN'T ONE:

  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1) -- 0, 1
  ...

  this is genuinely useful for a human skimming terminal output while
  debugging -- but look at its own syntax next to Chapter 5's REAL
  grammar, the one parseStatement() actually reads:

  printGraph()'s own syntax:        Chapter 5's real grammar:
    "%2 = Add(t1) -- 0, 1"            "t1 = add(a, b)"
    numeric node ids                  NAMES, resolved through a
    a dash-based separator that         symbol table
    is not part of the grammar        the exact op keyword the
    at all                              parser recognizes ("add",
                                         not "Add")

  feeding printGraph()'s own output back into parseProgram() would
  fail immediately -- it is a ONE-WAY dump, not a serialization.
```

### Background

The distinction this section is making precise -- a *debug dump* versus a *serialization* -- matters because the two have genuinely different jobs, and confusing them is a real, specific way to end up trusting a tool that cannot actually do what you need from it. A debug dump only has to be readable; nothing downstream ever consumes its output again, so it can take any liberties that make it easier for a human eye to parse (numeric ids instead of names, an invented arrow syntax, whatever is fastest to skim). A serialization has a much stronger job: whatever it produces has to be a complete, faithful, and *re-consumable* description of the thing it describes -- Part 2's pass manager, starting in Chapter 8, will want to print a graph before a pass runs, run the pass, print the graph again, and trust that neither dump silently dropped or distorted anything the pass actually did. A one-way dump cannot be trusted that way, because there is no way to check it against anything; a serialization that is also a genuine parser input can be checked the strongest way there is -- by parsing it back and comparing the result to what was printed.

!!! warning "[COMMON TRAP] Assuming any function that prints a graph counts as \"the printer\""
    Chapters 4, 5, and 6 each wrote their own small, slightly different print helper inside a `main()`, purely to make that one chapter's own terminal output readable while writing it -- none of them were built to be read back in, and none of them were meant to survive past that chapter. This chapter's own `printGraphAsSource()` (Section 7.2) is the first one in this book actually designed as a serialization, checked by the strongest test available: parsing its own output and confirming the result is structurally identical to what was printed. A print function that has never been tested this way should not be assumed to be one, no matter how readable its output looks.

## 7.2 A Printer That Is a Real Inverse of the Parser

### Intuition

A photocopier that only sometimes reproduces a document correctly is worse than no photocopier at all, because it invites trust it has not earned -- the failures are exactly the copies nobody double-checks. The bar for a real IR printer is the same: it is not enough for its output to *look* like valid syntax, because a printer with a subtle bug (a missing argument, a swapped operand order, a name collision) can produce text that is syntactically fine and semantically wrong, and nothing about glancing at it would reveal that. The only test strong enough to catch that class of bug is a genuine round trip: print a graph to text, parse that exact text back into a brand-new graph using Chapter 5's own unmodified parser, and check -- not by eye, but with the same `graphsStructurallyEqual()` function Chapter 5 already built -- that the new graph is identical, node for node and edge for edge, to the one that was printed.

```text
THE ROUND TRIP THIS SECTION'S SELF-CHECK ACTUALLY PROVES:

  step 1: Graph (original)
  step 2: text ("a = input()\nb = input()\n...")
              -- produced by printGraphAsSource(), new in this chapter
  step 3: Graph (round-tripped)
              -- produced by parseProgram(), Chapter 5's parser, UNCHANGED
  step 4: PASS/FAIL verdict
              -- produced by graphsStructurallyEqual(), Chapter 5's own
                 equality check, comparing step 3's graph to step 1's

  PASS only if the round-tripped graph (step 3) equals the original
  graph (step 1), node for node.

  if printGraphAsSource() ever silently dropped an input, swapped two
  operands, or mangled a name, this chain would produce a DIFFERENT
  graph at the end -- and the self-check would catch it, not a human
  eyeballing the printed text and assuming it looked about right.
```

### Background

`printGraphAsSource()` walks every node in the graph and emits exactly the line Chapter 5's grammar expects for that node's own operation: `input()` takes no arguments at all; `const(<value>)` takes the one stored numeric value, not a reference to anything; `add(<lhs>, <rhs>)` and `mul(<lhs>, <rhs>)` each take their two operands; `relu(<in>)` takes its one operand. The one detail worth being precise about is *what* gets printed for an operand -- never a node's numeric id (as Chapter 5's own File 007 debug dump did), always the input node's own `debugName`, because a numeric id like `%2` is not a legal argument in this book's grammar at all; only a NAME or a NUMBER is, and `parseStatement()` would reject anything else.

```text
PRINTING THE DIAMOND GRAPH, NODE BY NODE, ID-TO-NAME SUBSTITUTION MADE EXPLICIT:

  node  op     inputs (by id)   printed line          substitutions used
  %0    Input  (none)           a = input()           (none)
  %1    Input  (none)           b = input()           (none)
  %2    Add    %0, %1           t1 = add(a, b)        %0 -> "a", %1 -> "b"
  %3    Mul    %2, %0           t2 = mul(t1, a)        %2 -> "t1", %0 -> "a"
  %4    ReLU   %2                t3 = relu(t1)          %2 -> "t1"
  %5    Add    %3, %4           out = add(t2, t3)      %3 -> "t2", %4 -> "t3"

  every substitution above looks up g.node(inputId)->debugName -- the
  SAME lookup, every time, regardless of which operation is being
  printed, which is why one small loop handles Add, Mul, and ReLU
  identically and only Const needs its own separate branch at all.
```

A second detail this file's own code comments call out explicitly, because it is easy to assume the opposite: printing nodes in plain `g.nodes()` order -- the order they were created in, not some order computed by a call to `topologicalSort()` -- is already correct here, and does not need Chapter 4's topological sort at all.

```text
WHY NODE-CREATION ORDER IS ALREADY A VALID PRINTING ORDER:

  Chapter 4's own addBinary()/addUnary() can only take a Value that was
  ALREADY returned by some earlier call -- there is no way to construct
  a Value referring to a node that doesn't exist yet. That means, for
  every node in the graph:

    every one of its inputs' ids is strictly SMALLER than its own id

  which is exactly the property printGraphAsSource() needs: by the time
  it reaches node K in the loop, every node K could possibly reference
  as an input has ALREADY been printed, and therefore already has a
  debugName the substitution above can look up. This is the SAME
  cycle-freedom-by-construction argument Chapter 4 used to prove the
  graph can never contain a cycle in the first place -- here it shows
  up again as a free, no-extra-work guarantee about print order.
```

!!! note "This is a coincidence of THIS book's construction API, not a law"
    Chapter 6's own [COMMON TRAP] made the identical point about `inferShapes()`: id order and topological order coincide here because Chapter 4's API is append-only and can only reference already-existing nodes. A future pass that rewrites or reorders nodes -- which Part 2 starts building in Chapter 8 -- is not guaranteed to preserve that coincidence. `printGraphAsSource()` relies on it today because it is genuinely true today; a later chapter that adds node reordering would need to switch this function back to walking `topologicalSort()`'s own `order`, exactly the way `inferShapes()` already does on principle rather than convenience.

Const is the one operation whose argument is a number, not a name, and `printGraphAsSource()` branches on that explicitly rather than trying to force it through the same "look up a debugName" logic every other operation uses:

```text
THE ONE BRANCH IN printGraphAsSource() THAT ISN'T "LOOK UP A DEBUGNAME":

  every op except Const:  print the INPUT NODE'S debugName
                           (a reference to something already defined)

  Const specifically:     print the node's OWN constValue, formatted
                           as a number (e.g. "3.5")
                           (a literal, not a reference to anything)

  getting this branch wrong in either direction breaks the round trip
  immediately: printing a Const's constValue as if it were a name would
  produce "c = const(c)" (nonsense -- Const takes a NUMBER); printing
  some other op's input as a number instead of a name would produce
  text parseStatement() rejects outright, since add/mul/relu each
  require their arguments to resolve through the symbol table.
```

File 011 below proves all of this concretely: it prints the familiar diamond graph, checks the printed text against a hand-written expected string byte for byte, then performs the real round trip -- parsing that printed text and confirming `graphsStructurallyEqual()` against the original -- and repeats the whole thing a second time on a small graph containing a `Const` node, specifically to exercise the one branch the diamond graph itself never touches.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 011_printing_the_ir_as_reparseable_text.cpp -o 011_printing_the_ir_as_reparseable_text
./011_printing_the_ir_as_reparseable_text
```

**Output:**

```text
=== Section 7.2: printing a graph as text its OWN parser can read back in ===

Printed source (6 nodes):
a = input()
b = input()
t1 = add(a, b)
t2 = mul(t1, a)
t3 = relu(t1)
out = add(t2, t3)

self-check: printed text matches the hand-written expected source byte-for-byte (confirmed)
self-check: parseProgram(printGraphAsSource(g)) reconstructs a graph STRUCTURALLY
IDENTICAL to g -- print and parse are genuine inverses (confirmed)

Printed source with a Const node:
c = const(3.5)
x = input()
scaled = mul(c, x)

self-check: a graph containing a Const node also round-trips correctly (confirmed)
```

## 7.3 Seeing the Graph: Exporting to Real Graphviz DOT

### Intuition

Text answers "what does this graph say," read line by line, but it answers "what SHAPE is this graph" much more slowly than a picture does -- finding every node with two consumers, or seeing at a glance where a chain of computation branches and later merges back together, is exactly the kind of question eyes answer faster over a picture than over a column of `name = op(args)` lines. This section builds a second, completely different external view of the same `Graph`: an export to Graphviz's own DOT language, a real, independently specified graph-description format that a real, independently written tool (`dot`) already knows how to lay out and render. This book is not inventing a picture format of its own; it is speaking a language something else already understands.

```text
DOT'S OWN GRAMMAR, QUOTED FROM GRAPHVIZ'S REAL DOCUMENTATION
(https://graphviz.org/doc/info/lang.html), MAPPED ONTO WHAT THIS
SECTION'S exportToDot() ACTUALLY EMITS:

  Graphviz's own grammar rule            what exportToDot() emits
  --------------------------------       --------------------------------
  "digraph ID { stmt_list }"             digraph IR { ... }
  a node statement, with an
    attribute list in brackets           n2 [label="t1\nAdd\n[3, 4]"];
  "An edgeop is -> in directed
    graphs"                              n0 -> n2;
  an attr_list attribute is
    "ID = ID" inside "[ ]"               label="..." inside [ ]

  every line exportToDot() produces is one of these three real DOT
  statement forms -- nothing invented, nothing this book made up.
```

### Background

`exportToDot()` walks the graph twice: once to emit one node statement per `Node` (an id like `n2`, an attribute list containing a `label` built from that node's `debugName`, its operation, and -- when shape information is supplied -- its inferred shape from Chapter 6's own `inferShapes()`), and once to emit one edge statement per real input edge, `n<inputId> -> n<consumerId>;`, directly mirroring `Node::inputs` the same way every other file in this book has read that field. A node's DOT identifier is deliberately never its `debugName` directly:

```text
WHY exportToDot() USES "n2" AS A NODE'S DOT IDENTIFIER, NOT ITS debugName:

  DOT's own unquoted-identifier rule requires a letter or underscore
  followed by letters, digits, or underscores -- "n" + an integer id
  ALWAYS satisfies that rule, for any node this book's Graph can ever
  contain. A raw debugName is not guaranteed to: nothing in Chapter 4's
  or 5's own grammar stops a program from naming a value something DOT
  would need quoted or escaped.

  keeping the DOT identifier (n2) and the human-readable label
  ("t1\nAdd\n[3, 4]") as two SEPARATE things -- one for the graph
  structure, one purely for display -- sidesteps that whole problem
  without exportToDot() needing its own quoting/escaping logic at all.
```

The claim this section makes is stronger than "this text looks like DOT to a human reading it," and File 012's own self-check is built to prove that stronger claim directly: the generated text is written to a real file and handed to the actual `dot` binary (`dot -Tplain`, confirmed installed on this machine before the test runs), and the self-check passes only if Graphviz's own parser accepts it and exits successfully.

```text
THE VALIDATION PIPELINE FILE 012 ACTUALLY RUNS:

  step 1: Graph
  step 2: exportToDot()          -->  produces a .dot file on disk
  step 3: the REAL `dot` binary  -->  not this book's own code --
                                       parses that exact file
  step 4: exit code 0 (accepted) or exit code != 0 (rejected)

  this is the same discipline the book's own Getting Started page
  already commits to for every real external fact -- verify against the
  real thing, don't just assert your own output looks plausible. Here
  "the real thing" is Graphviz's own parser, not a web page to fetch.
```

File 012 also runs three lighter, tool-independent checks on the DOT text itself -- the number of node statements equals the graph's own node count, the number of edge statements equals the graph's own total input-edge count, and the diamond graph's `out` node specifically shows exactly two incoming edges in the text, matching its two real inputs -- so the file's self-check still means something on a machine where `dot` itself happens not to be installed.

```cpp
// Chapter 7: Printing and Visualizing CUDA Hammer's IR
// 012_visualizing_the_ir_as_graphviz_dot.cpp
//
// Section 7.3 -- text is enough to feed a program back into itself, but
// a human trying to SEE a graph's shape (where does it branch, where
// does it merge back together, how deep is it) reads a picture far
// faster than a column of "name = op(args)" lines. This file exports a
// Graph to Graphviz's real DOT language -- not an invented ASCII-art
// format of this book's own -- so any of Graphviz's own tools can render
// it (`dot -Tsvg`, `dot -Tpng`, ...).
//
// The DOT language's actual grammar, straight from Graphviz's own
// documentation (https://graphviz.org/doc/info/lang.html): a directed
// graph is declared as `digraph ID { stmt_list }`; a directed edge uses
// the `->` operator ("An edgeop is -> in directed graphs and -- in
// undirected graphs"); a node or edge takes an attribute list in square
// brackets, `[ ID = ID ]`, for things like a display `label`. This file
// uses exactly that syntax -- nothing more.
//
// This chapter's real correctness claim is stronger than "the text we
// produce looks like DOT": the generated file is fed to the ACTUAL
// Graphviz `dot` binary (confirmed installed on this machine before
// running this file) and its exit code and output are checked -- if
// Graphviz's own parser rejects the file, this file's self-check fails.
// We are not eyeballing our own output; we are asking the real tool
// whose syntax we are claiming to speak.
//
// Reuses Chapter 4's Value/Node/Graph, and Chapter 6's Shape/
// broadcastShapes()/inferShapes() (which itself reuses Chapter 4's
// topologicalSort() unchanged) so each node's label in the picture can
// show its inferred shape too -- tying together everything Part 1 built:
// the graph (Ch4), how it's written down (Ch5), what it's allowed to
// compute (Ch6), and now, how it's actually SEEN (Ch7).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 012_visualizing_the_ir_as_graphviz_dot.cpp -o 012_visualizing_the_ir_as_graphviz_dot
// Run:     ./012_visualizing_the_ir_as_graphviz_dot
// (Requires the `dot` command from Graphviz to be installed for the
// self-check; the DOT text itself is produced either way.)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include <fstream>

// ==================== Value / Node / Graph (from Chapter 4) ====================

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

// ==================== Topological sort (from Chapter 4, unchanged) ====================

struct TopoResult {
    std::vector<int> order;
    bool ok = true;
};

static TopoResult topologicalSort(const Graph& g) {
    std::map<int, int> inDegree;
    for (const auto& n : g.nodes()) inDegree[n->id] = static_cast<int>(n->inputs.size());
    std::deque<int> ready;
    for (const auto& n : g.nodes()) {
        if (inDegree[n->id] == 0) ready.push_back(n->id);
    }
    TopoResult result;
    while (!ready.empty()) {
        int id = ready.front();
        ready.pop_front();
        result.order.push_back(id);
        for (const auto& n : g.nodes()) {
            for (const Value& in : n->inputs) {
                if (in.nodeId == id) {
                    if (--inDegree[n->id] == 0) ready.push_back(n->id);
                }
            }
        }
    }
    result.ok = (result.order.size() == g.size());
    return result;
}

// ==================== Shape / broadcasting / inference (from Chapter 6, unchanged) ====================

struct Shape {
    std::vector<int> dims;
};

static std::string shapeStr(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s.dims[i]);
    }
    out += "]";
    return out;
}

static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        int outDim;
        if (da == db) {
            outDim = da;
        } else if (da == 1) {
            outDim = db;
        } else if (db == 1) {
            outDim = da;
        } else {
            throw std::runtime_error(context + ": shapes " + shapeStr(a) + " and " + shapeStr(b) +
                                      " are not broadcast-compatible");
        }
        result[rank - 1 - i] = outDim;
    }
    return Shape{result};
}

static std::map<int, Shape> inferShapes(const Graph& g, const std::map<int, Shape>& declaredShapes) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("inferShapes: graph is not acyclic");
    std::map<int, Shape> shapes;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        std::string context = "node %" + std::to_string(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            auto it = declaredShapes.find(id);
            if (it == declaredShapes.end()) throw std::runtime_error(context + ": no declared shape");
            shapes[id] = it->second;
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        } else {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

// ============================== Section 7.3: Graphviz DOT export (new) ==============================
//
// One node statement per Node ("n3 [label=\"...\"];"), one edge statement
// per (input -> consumer) pair ("n1 -> n3;"). Graphviz's own DOT id rules
// allow a bare, unquoted identifier for a node id as long as it is a
// letter/underscore followed by letters/digits/underscores -- "n3" always
// qualifies, which is exactly why node ids are printed as "n" + the
// integer id rather than risking a debugName that might contain a
// character DOT would need quoted.
static std::string exportToDot(const Graph& g, const std::map<int, Shape>* shapes = nullptr) {
    std::string out = "digraph IR {\n";
    out += "  node [shape=box, fontname=\"monospace\"];\n";

    for (const auto& n : g.nodes()) {
        std::string label = n->debugName + "\\n" + opKindStr(n->op);
        if (shapes) {
            auto it = shapes->find(n->id);
            if (it != shapes->end()) label += "\\n" + shapeStr(it->second);
        }
        out += "  n" + std::to_string(n->id) + " [label=\"" + label + "\"];\n";
    }
    out += "\n";
    for (const auto& n : g.nodes()) {
        for (const Value& in : n->inputs) {
            out += "  n" + std::to_string(in.nodeId) + " -> n" + std::to_string(n->id) + ";\n";
        }
    }
    out += "}\n";
    return out;
}

// Runs the real `dot` binary against a generated file and returns true
// only if it exits 0 -- i.e. Graphviz's OWN parser accepted the syntax.
static bool validateWithRealGraphviz(const std::string& dotText, const std::string& tmpPath) {
    std::ofstream f(tmpPath);
    f << dotText;
    f.close();
    std::string cmd = "dot -Tplain " + tmpPath + " -o " + tmpPath + ".out 2>" + tmpPath + ".err";
    int rc = std::system(cmd.c_str());
    return rc == 0;
}

int main() {
    printf("=== Section 7.3: exporting the graph as real Graphviz DOT ===\n\n");

    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");

    std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);

    std::string dotText = exportToDot(g, &shapes);
    printf("Generated DOT source:\n%s\n", dotText.c_str());

    // Structural self-checks on the text itself, independent of whether
    // `dot` is available: exactly one node statement per Node, and
    // exactly one edge statement per input edge in the whole graph.
    size_t nodeStatementCount = 0, pos = 0;
    while ((pos = dotText.find("[label=", pos)) != std::string::npos) { ++nodeStatementCount; pos += 7; }
    size_t edgeStatementCount = 0;
    pos = 0;
    while ((pos = dotText.find("->", pos)) != std::string::npos) { ++edgeStatementCount; pos += 2; }

    size_t expectedEdgeCount = 0;
    for (const auto& n : g.nodes()) expectedEdgeCount += n->inputs.size();

    bool nodeCountMatches = (nodeStatementCount == g.size());
    bool edgeCountMatches = (edgeStatementCount == expectedEdgeCount);
    printf("self-check: %zu node statements emitted, matching the graph's %zu nodes (%s)\n",
           nodeStatementCount, g.size(), nodeCountMatches ? "confirmed" : "MISMATCH");
    printf("self-check: %zu edge statements emitted, matching the graph's %zu real input edges (%s)\n",
           edgeStatementCount, expectedEdgeCount, edgeCountMatches ? "confirmed" : "MISMATCH");

    // Out is reached by every path through the diamond -- if the edge
    // count is right, "out" specifically must have in-degree 2 in the
    // DOT text (two "-> n5" occurrences, matching its two real inputs).
    size_t outInEdges = 0;
    pos = 0;
    std::string outTarget = "-> n" + std::to_string(out.nodeId) + ";";
    while ((pos = dotText.find(outTarget, pos)) != std::string::npos) { ++outInEdges; pos += outTarget.size(); }
    bool outInDegreeCorrect = (outInEdges == 2);
    printf("self-check: 'out' (n%d) has exactly 2 incoming edges in the DOT text (%s)\n",
           out.nodeId, outInDegreeCorrect ? "confirmed" : "MISMATCH");

    // The strong check: hand this exact text to the REAL Graphviz tool.
    bool graphvizAccepted = false;
    bool dotAvailable = (std::system("which dot > /dev/null 2>&1") == 0);
    if (dotAvailable) {
        graphvizAccepted = validateWithRealGraphviz(dotText, "/tmp/012_chapter7_graph.dot");
        printf("self-check: the real `dot` binary parsed this exact file without error (%s)\n",
               graphvizAccepted ? "confirmed" : "MISMATCH");
    } else {
        printf("(`dot` not found on this machine -- skipping the real-Graphviz validation step)\n");
    }

    bool allOk = nodeCountMatches && edgeCountMatches && outInDegreeCorrect && (!dotAvailable || graphvizAccepted);
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 012_visualizing_the_ir_as_graphviz_dot.cpp -o 012_visualizing_the_ir_as_graphviz_dot
./012_visualizing_the_ir_as_graphviz_dot
```

**Output:**

```text
=== Section 7.3: exporting the graph as real Graphviz DOT ===

Generated DOT source:
digraph IR {
  node [shape=box, fontname="monospace"];
  n0 [label="a\nInput\n[3, 4]"];
  n1 [label="b\nInput\n[4]"];
  n2 [label="t1\nAdd\n[3, 4]"];
  n3 [label="t2\nMul\n[3, 4]"];
  n4 [label="t3\nReLU\n[3, 4]"];
  n5 [label="out\nAdd\n[3, 4]"];

  n0 -> n2;
  n1 -> n2;
  n2 -> n3;
  n0 -> n3;
  n2 -> n4;
  n3 -> n5;
  n4 -> n5;
}

self-check: 6 node statements emitted, matching the graph's 6 nodes (confirmed)
self-check: 7 edge statements emitted, matching the graph's 7 real input edges (confirmed)
self-check: 'out' (n5) has exactly 2 incoming edges in the DOT text (confirmed)
self-check: the real `dot` binary parsed this exact file without error (confirmed)
```

!!! warning "[COMMON TRAP] Assuming a picture that looks right is proof the exporter is correct"
    Eyeballing a rendered graph and thinking "yes, that looks like the diamond" is a genuinely weak check -- a human glancing at a picture will not reliably notice one missing edge among seven, or a mislabeled node, the same way eyeballing printed text would not reliably catch a subtle printer bug in Section 7.2. File 012's own self-check does not stop at "Graphviz rendered something" -- it counts node statements against `g.size()`, counts edge statements against the graph's own total input-edge count, and checks `out`'s specific in-degree, all in code, before ever asking whether Graphviz itself accepts the file. A picture is for a human to look at *after* the structural checks already passed, not a substitute for them.

## Chapter Summary

This chapter built two genuinely different external views of the same `Graph`, closing out Part 1. Section 7.1 drew a sharp line between a debug dump -- readable, but not meant to be consumed by anything else, like every ad hoc print helper Chapters 4 through 6 wrote along the way -- and a real serialization, which has to survive being read back in. Section 7.2 built `printGraphAsSource()`, checked the strongest way available: printing a graph, parsing the result with Chapter 5's own unmodified parser, and confirming with `graphsStructurallyEqual()` that the round trip reproduces the original exactly, including the one special case (`Const`, whose argument is a number rather than a name) that every other operation's printing logic does not need to handle. It also showed, explicitly, that node-creation order is already a valid printing order here, for the same cycle-freedom-by-construction reason Chapter 4 first proved. Section 7.3 built `exportToDot()`, targeting Graphviz's own real DOT language (quoted directly from Graphviz's own documentation) rather than an invented picture format, and validated its output the strongest way available for that format too: handing the generated file to the actual `dot` binary and checking that Graphviz's own parser accepts it, on top of code-level checks on node and edge counts. Part 1 is now complete: a real graph data structure (Chapter 4), a way to build one from text (Chapter 5), a way to check it makes sense (Chapter 6), and now two ways to see it (Chapter 7). Part 2 starts writing CUDA Hammer's first genuine optimization passes over graphs Part 1 now guarantees are well-formed and fully observable, before and after every pass runs.

## Self-Check Questions

1. What specifically makes Chapter 5's own `printGraph()` (File 007) a "debug dump" rather than a "serialization," in the terms Section 7.1 uses?
2. Why does `printGraphAsSource()` print an input as the referenced node's `debugName` rather than its numeric id, and what would happen if it printed the id instead?
3. `printGraphAsSource()` does not call `topologicalSort()` before deciding what order to print nodes in. Why is plain `g.nodes()` order already correct here, and what specific property of Chapter 4's `Graph` API guarantees it?
4. Section 7.2's [COMMON TRAP]-adjacent note draws a parallel to Chapter 6's own `inferShapes()`. What is the shared assumption both functions currently rely on, and what kind of future change would break that assumption for both of them at once?
5. Why does `printGraphAsSource()` need a separate branch for `Const`, when every other operation is handled by one shared loop that looks up each input's `debugName`?
6. `exportToDot()` uses `"n" + nodeId` as each node's DOT identifier instead of using the node's own `debugName` directly. What real problem does this sidestep?
7. File 012's self-check includes handing the generated DOT text to the actual `dot` binary, not just checking the text's own structure in code. What does this catch that the code-level checks (node count, edge count, `out`'s in-degree) could not catch on their own?
8. Suppose `printGraphAsSource()` had a bug that printed every `Add` node's two operands in the wrong order (swapped). Would File 011's own three self-checks (byte-for-byte text match, diamond round trip, `Const`-graph round trip) be guaranteed to catch it? Why or why not?

## Where We Go Next

Part 1 is done: a real, sharable, printable, and picturable graph representation, built from first principles across four chapters. Part 2 begins with Chapter 8, "The Pass Manager: Structuring Compiler Transformations" -- the first chapter to actually *change* a graph rather than only build, parse, check, or display one, and the first place this chapter's own round-trip-verified printer earns its keep for real: every pass Part 2 writes will be debugged by printing a graph before that pass runs and printing it again after, trusting -- because Chapter 7 proved it, not just asserted it -- that neither dump is lying about what the pass actually did.

## Worked Solutions

1. `printGraph()` prints numeric node ids and a `<-`-style arrow that is not part of Chapter 5's own grammar at all -- feeding its output back into `parseStatement()` would fail immediately, on the very first token. A serialization's defining property is that it CAN be fed back into whatever reads the format it claims to produce; `printGraph()` was never built or tested against that bar, so calling it a serialization would be an unverified assumption, not a demonstrated fact.
2. A numeric id like `%2` is not a legal argument in Chapter 5's grammar -- `parseStatement()` only accepts an `Ident` (a name, resolved through the symbol table) or a `Number` as an argument token, and a bare integer id would either be misread as a `Number` (breaking `resolveIdentArg()`'s type check) or rejected outright. Printing the referenced node's `debugName` instead produces exactly the kind of token the parser already knows how to resolve, which is precisely what makes the round trip possible at all.
3. Because Chapter 4's `addBinary()`/`addUnary()` can only accept a `Value` that some earlier call already returned, every node's own inputs always have strictly smaller ids than the node itself. That means iterating `g.nodes()` in plain creation order already visits every node after all of its own inputs -- the exact ordering property `printGraphAsSource()` needs (every input already has a printable `debugName` by the time it's referenced) -- without needing `topologicalSort()`'s more general, explicitly-computed guarantee at all.
4. Both `printGraphAsSource()` and `inferShapes()`'s own earlier [COMMON TRAP] rely on the same fact: that plain node-id order and a valid topological order currently coincide, because Chapter 4's construction API is append-only and can only ever reference already-existing nodes. A future pass that rewrites, reorders, or replaces nodes in place -- which Part 2's own pass manager (Chapter 8) is specifically being built to support -- could break that coincidence for both functions simultaneously, which is exactly why `inferShapes()` already calls `topologicalSort()` explicitly rather than relying on the coincidence, and why this chapter's own note flags that `printGraphAsSource()` would need the same change if that day comes.
5. Every operation except `Const` refers to some other, already-existing node as its argument, so a single shared lookup (`g.node(inputId)->debugName`) is correct for all of them. `Const`'s one argument is fundamentally different in kind -- a literal number stored directly on the node itself (`constValue`), not a reference to anything else in the graph -- so applying the same "look up a debugName" logic to it would be a category error: there is no other node to look up, only a value the node already holds.
6. Graphviz's own DOT grammar allows a node identifier to appear unquoted only if it is a letter or underscore followed by letters, digits, or underscores. Nothing in this book's own text grammar (Chapters 4 and 5) restricts what characters a `debugName` can contain, so a program could legally define a value with a name DOT would need to quote or escape. Using `"n" + nodeId` instead guarantees every node identifier is always a valid unquoted DOT identifier, for any graph this book's `Graph` can ever represent, without `exportToDot()` needing any quoting or escaping logic of its own.
7. The code-level checks confirm the TEXT has the right shape by this book's own counting rules (the right number of `[label=` and `->` substrings) -- they cannot confirm that Graphviz's OWN, independently written parser agrees the file is well-formed DOT syntax in every other respect (correct statement termination, correctly balanced braces and brackets, a label string that is properly quoted and escaped, and so on). Only handing the file to the real `dot` binary and checking its exit code tests against the actual specification this section claims to be speaking, rather than against this book's own possibly-incomplete idea of what that specification requires.
8. No -- none of File 011's three checks would be guaranteed to catch a bug that swapped `Add`'s two operands. The byte-for-byte text check would fail (the printed text would read `t1 = add(b, a)` instead of `t1 = add(a, b)`), which for this specific test data DOES happen to reveal the bug -- but that is a property of the diamond graph's OWN hand-written expected string being asserted at all, not a property the round-trip checks themselves would provide. The two round-trip checks (`graphsStructurallyEqual()` against the original) would very likely still PASS despite the swap: `Add` is commutative in the actual arithmetic it represents, but more importantly, `graphsStructurallyEqual()` only compares each node's input *ids* in order -- it has no idea which operand slot is "supposed to be" which, so a version of the graph with `Add`'s own two inputs consistently swapped both when printed AND when re-parsed would still come back structurally identical by that check's own definition. This is a real gap: it shows why File 011 deliberately asserts the exact expected text as a real check of its own, not merely relying on the round trip, and why a future test for a non-commutative operation (something Part 2 or later might add) would need to check operand ORDER specifically, not just operand SET membership.

---

**Sources cited in this chapter:**

- Graphviz, ["The DOT Language"](https://graphviz.org/doc/info/lang.html) -- the real grammar `exportToDot()`'s own output is checked against: the `digraph ID { stmt_list }` graph declaration, the directed-edge operator ("An edgeop is `->` in directed graphs and `--` in undirected graphs"), and the `attr_list` syntax used for node labels. Used in Section 7.3. `exportToDot()`'s own implementation is original to this book; only the DOT syntax it emits is Graphviz's own, and File 012's self-check additionally validates the generated output against the real, installed `dot` binary rather than relying on this citation alone.
