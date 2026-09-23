# 11. Algebraic Simplification

**What you will understand:** the book's fourth genuinely *optimizing* transform pass -- algebraic simplification, which uses basic identities from ordinary arithmetic (adding zero, multiplying by one, multiplying by zero, applying `ReLU` twice) to rewrite a computation into something cheaper without computing a new answer at all -- and how, combined with Chapters 9 and 10's own passes through Chapter 8's unmodified `PassManager`, Part 2 closes with four independent optimizations working together on one graph.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph` and `topologicalSort()`, Chapter 7's `printGraphAsSource()`, Chapter 8's `TransformPass` and `runPasses()`, and Chapter 9's `constantFoldPass()`, `deadCodeEliminationPass()`, and `evaluate()`, and Chapter 10's `commonSubexpressionEliminationPass()` -- this chapter adds a fourth pass to that same pipeline, and reuses all three of Chapter 9 and 10's passes directly in Section 11.3.

---

Chapter 9's `constantFoldPass()` asks "is this computation's answer already fully known?" and, when it is, computes that answer once and emits it as a `Const`. Chapter 10's `commonSubexpressionEliminationPass()` asks "has this exact computation already been done somewhere else?" and, when it has, reuses the earlier result. This chapter's pass asks a third, narrower question that neither of those two can answer: "does ordinary arithmetic already guarantee this computation changes nothing at all, regardless of what its operand turns out to be?" Adding `0` to *any* number -- a compile-time constant or a genuine runtime `Input` nobody could ever fold away -- always returns that same number. Multiplying by `1` does too. A graph containing `add(x, 0)`, where `x` is a real runtime value, has nothing left to fold and nothing to deduplicate -- but the addition itself is still provably pointless, and algebraic simplification is the pass that notices.

```text
WHAT THIS CHAPTER ADDS TO CHAPTERS 8, 9, AND 10's OWN INFRASTRUCTURE:

  Ch8's passes:        Ch9's passes:            Ch10's pass:              THIS chapter's pass:

    identityPass          constantFoldPass         commonSubexpression      algebraicSimplificationPass
    canonicalizeNode-       deadCodeElimination-    EliminationPass           (rewrites toward an
      Names                  Pass                    (reuses an existing      operand ALREADY there,
                                                       computation)            using arithmetic identities
                                                                               like x+0=x, x*1=x, x*0=0)

  all five have the EXACT signature Chapter 8 already defined --
  Graph(const Graph&) -- so all five plug straight into runPasses()
  with ZERO changes to Chapter 8's own PassManager, across four
  separate chapters.
```

## 11.1 Algebraic Simplification: The Additive Identity

### Intuition

Walking zero extra steps in any direction leaves you exactly where you started -- it is not a *small* step, it is not a step that happens to cancel out later, it is simply not a step at all, and a sensible person planning a route would never bother writing it down. `add(x, 0)` is the graph equivalent of writing down that zero step: whatever `x` turns out to be, adding zero to it changes nothing, for every possible value `x` could ever take, not just the ones a compiler happens to already know. Constant folding (Chapter 9) can only replace a computation once *every* operand is a known constant; `add(x, 0)` never qualifies, because `x` itself might be a genuine runtime input. Algebraic simplification does not need `x` to be known at all -- it only needs to know that `0` is the additive identity, a fact about arithmetic that holds regardless of what the other operand is.

```text
WHAT algebraicSimplificationPass() SEES, THAT constantFoldPass() CANNOT USE:

  constantFoldPass() folds add(x, 0)     only when x is ALSO a Const --
                                          it computes a brand new answer,
                                          and needs every input to do it.

  algebraicSimplificationPass() simplifies add(x, 0) for ANY x at all --
                                          it doesn't compute anything; it
                                          just points at x, which was
                                          ALREADY sitting right there.
```

### Background

`algebraicSimplificationPass()` is structurally the twin of `constantFoldPass()`: it rebuilds the graph in topological order, and at every `Add`, checks its own already-rebuilt operands against the additive identity. Where `constantFoldPass()` responds to a match by *computing* a new answer and building a fresh `Const` node, this pass responds by pointing directly at an operand that already exists in the new graph -- no computation, no new node of any kind.

```text
algebraicSimplificationPass() AT Add (this section's own identity):

  add(x, 0)  -->  IF the RIGHT operand is Const(0.0):  result is the LEFT operand, unchanged
  add(0, x)  -->  IF the LEFT operand is Const(0.0):   result is the RIGHT operand, unchanged
  add(x, y)  -->  otherwise: rebuild the Add exactly as it was

  in every "simplifies" case, NOTHING is computed and NO new node is
  built -- the result is simply a Value that was already sitting in the
  new graph before this node was even reached.
```

Because the check reads `oldIdToNewValue` -- the map from old ids to the *new* graph's own nodes, populated as the walk proceeds -- a simplification discovered early is immediately visible to anything downstream that consumes it, exactly the "check the new graph, not the old one" discipline `constantFoldPass()` used to fold an entire chain in a single pass, and `commonSubexpressionEliminationPass()` used to cascade a merge in a single pass. Section 11.2 shows this cascading behavior firing three levels deep in one graph; this section keeps things simpler, exercising the identity at more than one position (`x + 0` and `0 + x` both) and confirming a non-identity `Add` is left completely untouched.

```cpp
// Chapter 11: Algebraic Simplification
// 021_algebraic_simplification_the_additive_identity.cpp
//
// Section 11.1 -- algebraic simplification, this book's FOURTH
// TransformPass, plugging into the exact same Chapter 8 machinery
// constantFoldPass() (9.1), deadCodeEliminationPass() (9.2), and
// commonSubexpressionEliminationPass() (10.1) already plug into, with
// zero changes to that machinery required, for the fourth chapter in a
// row.
//
// The three passes so far each answer a different question: "is this
// computation's answer already fully known?" (folding), "does anything
// still need this?" (dead code elimination), "has this exact
// computation already been done somewhere else?" (CSE). Algebraic
// simplification asks a fourth: "does basic arithmetic already tell me
// this computation is POINTLESS, regardless of what its operand turns
// out to be?" Adding 0 to ANY number -- known or not, a runtime Input
// or a compile-time Const -- always returns that same number, unchanged.
// A graph can easily contain `add(x, 0)` where `x` is a genuine runtime
// value nobody could fold away -- but the ADDITION itself is still
// provably pointless, and this pass is the one that notices.
//
// algebraicSimplificationPass() is structurally the twin of Section
// 9.1's constantFoldPass(): it rebuilds the graph in topological order,
// and at Add and Mul nodes, checks its own (already-rebuilt) operands
// against a small table of algebraic identities. Where constantFoldPass()
// responds to a match by COMPUTING a brand new answer and emitting a
// fresh Const node, this pass responds by pointing straight at an
// operand THAT ALREADY EXISTS -- no computation, no new node, not even
// a new Const. This section covers the additive identity element: `x +
// 0` and `0 + x` both simplify to plain `x`, for any `x` at all.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 021_algebraic_simplification_the_additive_identity.cpp -o 021_algebraic_simplification_the_additive_identity
// Run:     ./021_algebraic_simplification_the_additive_identity
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

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

// ==================== evaluate() (from Chapter 9, unchanged) ====================

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");

    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) {
            v = inputValuesByName.at(n->debugName);
        } else if (n->op == OpKind::Const) {
            v = n->constValue;
        } else if (n->op == OpKind::Add) {
            v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        } else if (n->op == OpKind::Mul) {
            v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        } else { // ReLU
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

// ============================== Section 11.1: algebraicSimplificationPass() (new) ==============================
//
// This file only exercises the ADD identity element (0 is what "doing
// nothing" means for addition). Section 11.2 adds Mul's own identity
// (1), Mul's absorbing element (0), and ReLU's idempotence -- all
// through the exact same function, reproduced unchanged in every file
// in this chapter, the same "one real pass, reused verbatim" discipline
// every earlier chapter in Part 2 has followed.
using TransformPass = std::function<Graph(const Graph&)>;

static Graph algebraicSimplificationPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("algebraicSimplificationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        Value newValue;

        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->op == OpKind::ReLU) {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            const Node* inNode = result.node(in.nodeId);
            if (inNode->op == OpKind::ReLU) {
                // relu(relu(x)) -> relu(x): applying ReLU to something
                // ReLU already clamped changes nothing further.
                newValue = in;
            } else {
                newValue = result.addUnary(OpKind::ReLU, in, n->debugName);
            }
        } else { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            const Node* lhsNode = result.node(lhs.nodeId);
            const Node* rhsNode = result.node(rhs.nodeId);
            bool lhsIsZero = (lhsNode->op == OpKind::Const && lhsNode->constValue == 0.0f);
            bool rhsIsZero = (rhsNode->op == OpKind::Const && rhsNode->constValue == 0.0f);
            bool lhsIsOne  = (lhsNode->op == OpKind::Const && lhsNode->constValue == 1.0f);
            bool rhsIsOne  = (rhsNode->op == OpKind::Const && rhsNode->constValue == 1.0f);

            if (n->op == OpKind::Add) {
                if (lhsIsZero) newValue = rhs;          // 0 + x -> x
                else if (rhsIsZero) newValue = lhs;     // x + 0 -> x
                else newValue = result.addBinary(OpKind::Add, lhs, rhs, n->debugName);
            } else { // Mul
                if (lhsIsZero) newValue = lhs;          // 0 * x -> 0 (the zero itself)
                else if (rhsIsZero) newValue = rhs;     // x * 0 -> 0
                else if (lhsIsOne) newValue = rhs;      // 1 * x -> x
                else if (rhsIsOne) newValue = lhs;      // x * 1 -> x
                else newValue = result.addBinary(OpKind::Mul, lhs, rhs, n->debugName);
            }
        }

        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) printf(" [value=%g]", n->constValue);
        printf("\n");
    }
}

int main() {
    printf("=== Section 11.1: algebraicSimplificationPass(), the additive identity element ===\n\n");

    // t1 = x + 0, t2 = 0 + x -- both should simplify to just x, pointing
    // straight at x's own node, with NO new node built at all. t3 = x +
    // y is the control: neither operand is Const(0), so it must survive
    // untouched. t12 and out both combine already-simplified operands,
    // so the pass has to work correctly through TWO more levels after
    // the leaf-level simplification, not just at the very first Add.
    Graph g;
    Value x = g.addInput("x");
    Value y = g.addInput("y");
    Value zero = g.addConst(0.0f, "zero");
    Value t1 = g.addBinary(OpKind::Add, x, zero, "t1");     // x + 0 -> x
    Value t2 = g.addBinary(OpKind::Add, zero, x, "t2");     // 0 + x -> x
    Value t3 = g.addBinary(OpKind::Add, x, y, "t3");        // control: no identity applies
    Value t12 = g.addBinary(OpKind::Add, t1, t2, "t12");
    g.addBinary(OpKind::Add, t12, t3, "out");

    printf("Graph with two additive-identity Adds ('t1', 't2') and one real Add ('t3'):\n");
    printGraph(g);

    Graph simplified = algebraicSimplificationPass(g);
    printf("\nAfter algebraicSimplificationPass():\n");
    printGraph(simplified);

    bool twoNodesDropped = (simplified.size() == g.size() - 2);
    bool t1Gone = true, t2Gone = true;
    for (const auto& n : simplified.nodes()) {
        if (n->debugName == "t1") t1Gone = false;
        if (n->debugName == "t2") t2Gone = false;
    }
    printf("\nself-check: exactly two identity Adds were dropped, %zu -> %zu (%s)\n",
           g.size(), simplified.size(), twoNodesDropped ? "confirmed" : "MISMATCH");
    printf("self-check: neither 't1' nor 't2' appears anywhere in the simplified graph (%s)\n",
           (t1Gone && t2Gone) ? "confirmed" : "MISMATCH");

    const Node* t3InSimplified = findNodeByName(simplified, "t3");
    bool t3StillAdd = (t3InSimplified->op == OpKind::Add);
    printf("self-check: 't3' (x + y, no identity applies) is still a real Add (%s)\n",
           t3StillAdd ? "confirmed" : "MISMATCH");

    const Node* t12InSimplified = findNodeByName(simplified, "t12");
    bool t12BothOperandsAreX = (t12InSimplified->inputs[0].nodeId == t12InSimplified->inputs[1].nodeId);
    printf("self-check: 't12' now reads the SAME node (x) as both of its own operands (%s)\n",
           t12BothOperandsAreX ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"x", 4.0f}, {"y", 9.0f}};
    float originalOut = evaluate(g, inputs).at("out");
    float simplifiedOut = evaluate(simplified, inputs).at("out");
    bool sameAnswer = (originalOut == simplifiedOut);
    printf("self-check: evaluate(original, x=4,y=9).out = %g, evaluate(simplified, ...).out = %g (%s)\n",
           originalOut, simplifiedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = twoNodesDropped && t1Gone && t2Gone && t3StillAdd && t12BothOperandsAreX && sameAnswer;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 021_algebraic_simplification_the_additive_identity.cpp -o 021_algebraic_simplification_the_additive_identity
./021_algebraic_simplification_the_additive_identity
```

**Output:**

```text
=== Section 11.1: algebraicSimplificationPass(), the additive identity element ===

Graph with two additive-identity Adds ('t1', 't2') and one real Add ('t3'):
  %0 = Input(x)
  %1 = Input(y)
  %2 = Const(zero) [value=0]
  %3 = Add(t1)
  %4 = Add(t2)
  %5 = Add(t3)
  %6 = Add(t12)
  %7 = Add(out)

After algebraicSimplificationPass():
  %0 = Input(x)
  %1 = Input(y)
  %2 = Const(zero) [value=0]
  %3 = Add(t3)
  %4 = Add(t12)
  %5 = Add(out)

self-check: exactly two identity Adds were dropped, 8 -> 6 (confirmed)
self-check: neither 't1' nor 't2' appears anywhere in the simplified graph (confirmed)
self-check: 't3' (x + y, no identity applies) is still a real Add (confirmed)
self-check: 't12' now reads the SAME node (x) as both of its own operands (confirmed)
self-check: evaluate(original, x=4,y=9).out = 21, evaluate(simplified, ...).out = 21 (confirmed)
```

!!! note "'zero' survives the pass, even though nothing references it anymore"
    Look at the simplified graph's own output: `zero` is still sitting there as node `%2`, even though neither `t1` nor `t2` -- the only two nodes that ever read it -- exist anymore. `algebraicSimplificationPass()` rebuilds every leaf in topological order unconditionally, exactly the way `constantFoldPass()` (Section 9.1) always kept `c1` and `c2` around even after a fold made them irrelevant. Neither pass retroactively goes back and removes a node just because something ELSE it built later stopped needing it -- that is specifically `deadCodeEliminationPass()`'s job (Section 9.2), and Section 11.3 puts the two back to back on purpose.

## 11.2 The Absorbing Identity and an Idempotent Unary Op

### Intuition

Multiplying anything by zero is a different kind of "doing nothing" than adding zero. `x + 0` still gives you `x` back -- the identity element leaves the OTHER operand completely intact. `x * 0`, by contrast, gives you `0` back, no matter what `x` was -- the zero *absorbs* the other operand entirely, discarding it, whatever it happened to be. `1`, meanwhile, plays the same role for multiplication that `0` plays for addition: `x * 1` returns `x` unchanged, operand intact. And `ReLU` has an identity property of its own that has nothing to do with `0` or `1` at all: clamping every negative value to zero and leaving everything else untouched is a property that, once applied, cannot be made to matter again by applying it a second time -- `relu(relu(x))` and `relu(x)` are the exact same number, for every possible `x`, because there is nothing left for the second `relu` to clamp that the first one didn't already clamp.

```text
FOUR IDENTITIES, TWO DIFFERENT SHAPES:

  "returns an OPERAND, unchanged"          "returns something NEW, discarding an operand"

    add(x, 0)  -->  x                        mul(x, 0)  -->  0  (x is thrown away entirely)
    add(0, x)  -->  x                        mul(0, x)  -->  0  (x is thrown away entirely)
    mul(x, 1)  -->  x
    mul(1, x)  -->  x                      "applying it again changes nothing further"

                                              relu(relu(x))  -->  relu(x)
```

### Background

`mul(x, 0)`'s own shape matters for what comes later in this chapter: unlike every other identity here, it does not merely skip building a NEW node -- it discards the OTHER operand outright, along with everything that operand depended on. If `x` were itself an expensive, multi-node computation, `mul(x, 0)` throws all of that work away in one step, the instant simplification reaches it -- and whatever `x` was built from becomes real, newly-created dead code, since nothing in the simplified graph references it anymore. This chapter's own pass, like `constantFoldPass()` before it, never goes back and cleans that up; that observation is Section 11.3's own starting point.

```text
mul(x, 0) DISCARDS x -- AND EVERYTHING x DEPENDED ON:

  before:                        after:

    a   b                          a   b        -- now UNREFERENCED,
     \ /                            \ /             but still PRESENT
    y=mul(a,b)                    y=mul(a,b)         (only DCE removes it)
       \                                        
    w=mul(y,0)   -->   just "0"    w  -->  (gone -- aliases 0 directly)
```

Because `algebraicSimplificationPass()` checks the graph being rebuilt (not the original), a simplification discovered early in a walk is visible to whatever consumes it later -- letting several simplifications cascade in a single pass, the same discipline this chapter has followed since Section 11.1. The file below builds a graph where that cascade runs three levels deep: two separate `mul(_, 0)`s both collapse to the *same* zero constant; an `Add` that then combines those two now-identical zeros collapses *again*, to that same zero; and an `Add` that combines *that* zero with something else collapses a *third* time, to whatever that something else was.

```text
A THREE-LEVEL CASCADE, DISCOVERED IN ONE PASS:

  level 1:  t3=mul(x,0)  -->  aliases zero          t4=mul(0,y)  -->  aliases zero
  level 2:  s2=add(t3,t4)  =  add(zero,zero)  -->  ITSELF an identity  -->  aliases zero
  level 3:  s3=add(s1,s2)  =  add(s1,zero)  -->  ITSELF an identity  -->  aliases s1

  three separate identity matches, chained together, in a SINGLE walk
  of algebraicSimplificationPass() -- no repeated passes needed.
```

```cpp
// Chapter 11: Algebraic Simplification
// 022_multiplicative_identity_absorbing_zero_and_relu_idempotence.cpp
//
// Section 11.2 -- the same algebraicSimplificationPass() from Section
// 11.1, reproduced unchanged, exercising the identities Section 11.1
// did not: Mul's own identity element (`x * 1` and `1 * x` both
// simplify to plain `x`, the multiplicative twin of Section 11.1's
// additive-zero identity), Mul's ABSORBING element (`x * 0` and `0 * x`
// both simplify to plain `0` -- a genuinely different kind of identity
// from the other three: it does not return one of the two operands
// unchanged, it returns the CONSTANT operand itself, discarding the
// other operand -- and whatever produced that other operand -- entirely,
// regardless of what it was), and ReLU's idempotence (`relu(relu(x))`
// simplifies to plain `relu(x)`, since ReLU applied twice clamps
// nothing that applying it once didn't already clamp).
//
// The graph this file builds is deliberately busy: it is built so that
// EVERY identity this pass knows fires at least once, and so that
// several of those firings CASCADE into further firings, discovered in
// the very same topological walk -- the identical "check the new graph,
// not the old one" discipline that let constantFoldPass() (9.1) fold an
// entire chain in one pass and commonSubexpressionEliminationPass()
// (10.1) discover a multi-level merge in one pass, now shown a third
// time for a third kind of pass. Two Mul-by-zero nodes both collapse to
// the SAME shared zero constant; an Add that then combines those two
// zero-aliases collapses AGAIN, to that same zero; and an Add that then
// combines THAT zero-alias with something else collapses a THIRD time,
// to whatever that something else was. Three cascade levels, one pass.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 022_multiplicative_identity_absorbing_zero_and_relu_idempotence.cpp -o 022_multiplicative_identity_absorbing_zero_and_relu_idempotence
// Run:     ./022_multiplicative_identity_absorbing_zero_and_relu_idempotence
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

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

// ==================== evaluate() (from Chapter 9, unchanged) ====================

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");

    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) {
            v = inputValuesByName.at(n->debugName);
        } else if (n->op == OpKind::Const) {
            v = n->constValue;
        } else if (n->op == OpKind::Add) {
            v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        } else if (n->op == OpKind::Mul) {
            v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        } else { // ReLU
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

// ==================== algebraicSimplificationPass() (from Section 11.1, unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;

static Graph algebraicSimplificationPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("algebraicSimplificationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        Value newValue;

        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->op == OpKind::ReLU) {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            const Node* inNode = result.node(in.nodeId);
            if (inNode->op == OpKind::ReLU) {
                newValue = in;
            } else {
                newValue = result.addUnary(OpKind::ReLU, in, n->debugName);
            }
        } else { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            const Node* lhsNode = result.node(lhs.nodeId);
            const Node* rhsNode = result.node(rhs.nodeId);
            bool lhsIsZero = (lhsNode->op == OpKind::Const && lhsNode->constValue == 0.0f);
            bool rhsIsZero = (rhsNode->op == OpKind::Const && rhsNode->constValue == 0.0f);
            bool lhsIsOne  = (lhsNode->op == OpKind::Const && lhsNode->constValue == 1.0f);
            bool rhsIsOne  = (rhsNode->op == OpKind::Const && rhsNode->constValue == 1.0f);

            if (n->op == OpKind::Add) {
                if (lhsIsZero) newValue = rhs;
                else if (rhsIsZero) newValue = lhs;
                else newValue = result.addBinary(OpKind::Add, lhs, rhs, n->debugName);
            } else { // Mul
                if (lhsIsZero) newValue = lhs;
                else if (rhsIsZero) newValue = rhs;
                else if (lhsIsOne) newValue = rhs;
                else if (rhsIsOne) newValue = lhs;
                else newValue = result.addBinary(OpKind::Mul, lhs, rhs, n->debugName);
            }
        }

        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) printf(" [value=%g]", n->constValue);
        printf("\n");
    }
}

int main() {
    printf("=== Section 11.2: multiplicative identity, absorbing zero, and ReLU idempotence, cascading in one pass ===\n\n");

    Graph g;
    Value x = g.addInput("x");
    Value y = g.addInput("y");
    Value one = g.addConst(1.0f, "one");
    Value zero = g.addConst(0.0f, "zero");
    Value t1 = g.addBinary(OpKind::Mul, x, one, "t1");     // x * 1 -> x
    Value t2 = g.addBinary(OpKind::Mul, one, x, "t2");     // 1 * x -> x
    Value t3 = g.addBinary(OpKind::Mul, x, zero, "t3");    // x * 0 -> 0
    Value t4 = g.addBinary(OpKind::Mul, zero, y, "t4");    // 0 * y -> 0
    Value t5 = g.addBinary(OpKind::Mul, x, y, "t5");       // control: no identity applies
    Value r1 = g.addUnary(OpKind::ReLU, x, "r1");
    Value r2 = g.addUnary(OpKind::ReLU, r1, "r2");         // relu(relu(x)) -> relu(x)
    Value s1 = g.addBinary(OpKind::Add, t1, t2, "s1");     // CASCADE level 1: becomes add(x, x)
    Value s2 = g.addBinary(OpKind::Add, t3, t4, "s2");     // CASCADE level 1: becomes add(zero, zero) -- itself an identity!
    Value s3 = g.addBinary(OpKind::Add, s1, s2, "s3");     // CASCADE level 2: becomes add(s1, zero) -- itself an identity!
    Value s4 = g.addBinary(OpKind::Add, t5, r2, "s4");     // becomes add(t5, r1)
    g.addBinary(OpKind::Add, s3, s4, "out");                // becomes add(s1, s4)

    printf("Graph exercising every identity this pass knows, with three cascade levels:\n");
    printGraph(g);

    Graph simplified = algebraicSimplificationPass(g);
    printf("\nAfter algebraicSimplificationPass():\n");
    printGraph(simplified);

    // Six nodes elided: t1, t2 (mul identity), t3, t4 (mul absorbing
    // zero), r2 (ReLU idempotence), s2 AND s3 (both cascades). 16 -> 9.
    bool sizeCorrect = (simplified.size() == 9);
    printf("\nself-check: seven nodes were elided across three cascade levels, %zu -> %zu (%s)\n",
           g.size(), simplified.size(), sizeCorrect ? "confirmed" : "MISMATCH");

    bool allEludedNamesGone = true;
    for (const char* name : {"t1", "t2", "t3", "t4", "r2", "s2", "s3"}) {
        for (const auto& n : simplified.nodes()) {
            if (n->debugName == name) allEludedNamesGone = false;
        }
    }
    printf("self-check: none of 't1','t2','t3','t4','r2','s2','s3' appear in the simplified graph (%s)\n",
           allEludedNamesGone ? "confirmed" : "MISMATCH");

    const Node* t5InSimplified = findNodeByName(simplified, "t5");
    const Node* r1InSimplified = findNodeByName(simplified, "r1");
    const Node* s1InSimplified = findNodeByName(simplified, "s1");
    const Node* s4InSimplified = findNodeByName(simplified, "s4");
    const Node* outInSimplified = findNodeByName(simplified, "out");
    bool survivorsCorrect =
        (t5InSimplified->op == OpKind::Mul) &&
        (r1InSimplified->op == OpKind::ReLU) &&
        (s1InSimplified->op == OpKind::Add && s1InSimplified->inputs[0].nodeId == s1InSimplified->inputs[1].nodeId) &&
        (s4InSimplified->op == OpKind::Add &&
         s4InSimplified->inputs[0].nodeId == t5InSimplified->id &&
         s4InSimplified->inputs[1].nodeId == r1InSimplified->id) &&
        (outInSimplified->op == OpKind::Add &&
         outInSimplified->inputs[0].nodeId == s1InSimplified->id &&
         outInSimplified->inputs[1].nodeId == s4InSimplified->id);
    printf("self-check: 't5','r1','s1','s4','out' all survive as real nodes, correctly rewired (%s)\n",
           survivorsCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"x", 4.0f}, {"y", 5.0f}};
    float originalOut = evaluate(g, inputs).at("out");
    float simplifiedOut = evaluate(simplified, inputs).at("out");
    bool sameAnswer = (originalOut == simplifiedOut);
    printf("self-check: evaluate(original, x=4,y=5).out = %g, evaluate(simplified, ...).out = %g (%s)\n",
           originalOut, simplifiedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = sizeCorrect && allEludedNamesGone && survivorsCorrect && sameAnswer;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 022_multiplicative_identity_absorbing_zero_and_relu_idempotence.cpp -o 022_multiplicative_identity_absorbing_zero_and_relu_idempotence
./022_multiplicative_identity_absorbing_zero_and_relu_idempotence
```

**Output:**

```text
=== Section 11.2: multiplicative identity, absorbing zero, and ReLU idempotence, cascading in one pass ===

Graph exercising every identity this pass knows, with three cascade levels:
  %0 = Input(x)
  %1 = Input(y)
  %2 = Const(one) [value=1]
  %3 = Const(zero) [value=0]
  %4 = Mul(t1)
  %5 = Mul(t2)
  %6 = Mul(t3)
  %7 = Mul(t4)
  %8 = Mul(t5)
  %9 = ReLU(r1)
  %10 = ReLU(r2)
  %11 = Add(s1)
  %12 = Add(s2)
  %13 = Add(s3)
  %14 = Add(s4)
  %15 = Add(out)

After algebraicSimplificationPass():
  %0 = Input(x)
  %1 = Input(y)
  %2 = Const(one) [value=1]
  %3 = Const(zero) [value=0]
  %4 = ReLU(r1)
  %5 = Mul(t5)
  %6 = Add(s1)
  %7 = Add(s4)
  %8 = Add(out)

self-check: seven nodes were elided across three cascade levels, 16 -> 9 (confirmed)
self-check: none of 't1','t2','t3','t4','r2','s2','s3' appear in the simplified graph (confirmed)
self-check: 't5','r1','s1','s4','out' all survive as real nodes, correctly rewired (confirmed)
self-check: evaluate(original, x=4,y=5).out = 32, evaluate(simplified, ...).out = 32 (confirmed)
```

!!! warning "[COMMON TRAP] Assuming 'x * 0 simplifies to 0' is free"
    It is free in the sense that no arithmetic runs and no new node is built -- but it is not free in the sense of "nothing else changes." Every node `x` was built from is now unreferenced, exactly like `zero` at the end of Section 11.1, only potentially much larger: if `x` had been an entire multi-layer subexpression instead of a single leaf, `mul(x, 0)` would orphan the whole thing in one step. This chapter's own pass does not clean that up -- it was never designed to; that is what `deadCodeEliminationPass()` is for. Section 11.3 combines both on purpose, and shows the difference it makes.

## 11.3 Folding, CSE, Simplification, and Dead Code Elimination Together: Closing Part 2's Pipeline

### Intuition

Section 9.3 asked whether constant folding could expose new dead code, and proved it could. Section 10.3 asked a different-angle version of the same question -- whether folding could expose a common-subexpression opportunity CSE could never find on its own -- and proved that too. This section asks the question from a third angle: can folding expose an algebraic-simplification opportunity that simplification could never find on its own? The graph below is built so the answer, once again, is yes -- and once again, this chapter does not just assert it: it measures it, running `algebraicSimplificationPass()` completely alone on the unfolded graph first, and confirming directly that it finds nothing.

```text
THE SAME "COULD RUNNING THIS AGAIN FIND MORE" QUESTION, A THIRD ANGLE:

  Section 9.3's angle:          Section 10.3's angle:          THIS section's angle:

    fold --> dead code            fold --> two DIFFERENT-       fold --> an Add becomes
    dce  --> cleans it up          LOOKING nodes become          EXACTLY Const(0)
                                    IDENTICAL                    simplify --> uses that 0
                                   cse --> merges them,            to discard an entire
                                    then CASCADES                  subexpression
                                                                  dce --> cleans up what
                                                                   simplification just
                                                                   orphaned
```

### Background

The graph this section builds gives `constantFoldPass()`, `commonSubexpressionEliminationPass()`, `algebraicSimplificationPass()`, and `deadCodeEliminationPass()` each a genuine, separate job to do, in that order. `c1 + c2` (values `3` and `-3`) folds to `Const(0)`, named `z`. Before folding runs, `z` is a plain `Add` node -- `algebraicSimplificationPass()` only ever checks whether an operand *is already* a `Const`, never what it would eventually compute to, so `mul(y, z)` cannot be recognized as "multiply by zero" until folding has actually turned `z` into one. `y` itself is a real, otherwise-unused subexpression (`mul(a, b)`) -- once simplification uses the now-folded `z` to discard `mul(y, z)` entirely, `y` (and the `a`/`b` that fed it) become genuinely new dead code, exactly the hazard Section 11.2's own `[COMMON TRAP]` described. A literal duplicate (`sum2`, computing the same thing as `sum1`) gives `commonSubexpressionEliminationPass()` real work too, merging before simplification even runs.

```text
BEFORE FOLDING:  z is an Add -- mul(y, z) has no Const operand at all.

    c1  c2                z is NOT yet a Const --
     \  /                 algebraicSimplificationPass()
    z=add(c1,c2)           has NOTHING to check it against.

  AFTER FOLDING:  z is Const(0) -- NOW mul(y, z) matches the absorbing identity.

    z=const(0)             mul(y, z)  -->  aliases z directly,
                            discarding y (and a, b) entirely.
```

```cpp
// Chapter 11: Algebraic Simplification
// 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline.cpp
//
// Section 11.3 -- constantFoldPass() (9.1), commonSubexpressionEliminationPass()
// (10.1), algebraicSimplificationPass() (11.1/11.2), and
// deadCodeEliminationPass() (9.2), all four run back to back through
// Chapter 8's own, still completely unmodified runPasses(). Four
// TransformPasses, from three different chapters, closing out Part 2.
//
// PART A -- a control measurement, the same "prove it, don't assert it"
// discipline Section 9.3 and Section 10.3 both used for their own
// order-matters claims: algebraicSimplificationPass() run ALONE on the
// ORIGINAL (unfolded) graph below finds nothing to simplify at the one
// spot that matters, because the relevant operand is still an Add node,
// not yet the Const(0) constant folding will turn it into.
//
// PART B -- the full four-pass pipeline. The graph is built so that
// constant folding creates a zero, common subexpression elimination
// merges a literal duplicate, algebraic simplification uses that zero
// to discard an entire otherwise-unrelated subexpression, and dead code
// elimination cleans up everything that discarding left behind --
// shrinking a real 13-node graph down to 4, with evaluate() confirming
// the answer never changed.
//
// PART C -- a compact, deliberately separate demonstration of a real
// hazard: when algebraic simplification happens to elide the LITERAL
// last node a graph ever added, Chapter 9's own "the last node is the
// graph's own output" convention -- which deadCodeEliminationPass()
// depends on completely -- silently ends up pointing at the wrong node.
// This is not a new bug introduced by this chapter; it is Section 9.2's
// own honestly-stated limitation (Worked Solution #4: "a graph needing
// ... an output that is not the most recently created node, could not
// be expressed under this convention at all") coming true for a
// concrete, demonstrable reason. Part B's own graph was deliberately
// built to avoid this exact trap; Part C shows precisely why that care
// was necessary.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline.cpp -o 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline
// Run:     ./023_folding_cse_simplification_and_dce_closing_part_2s_pipeline
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <algorithm>
#include <functional>
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

// ==================== printGraphAsSource() (from Chapter 7, unchanged) ====================

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
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", n->constValue);
            out += buf;
        } else {
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                out += g.node(n->inputs[i].nodeId)->debugName;
            }
        }
        out += ")\n";
    }
    return out;
}

// ==================== evaluate() (from Chapter 9, unchanged) ====================

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");

    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) {
            v = inputValuesByName.at(n->debugName);
        } else if (n->op == OpKind::Const) {
            v = n->constValue;
        } else if (n->op == OpKind::Add) {
            v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        } else if (n->op == OpKind::Mul) {
            v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        } else { // ReLU
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== constantFoldPass() (from Section 9.1, unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;

static Graph constantFoldPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("constantFoldPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        Value newValue;

        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->op == OpKind::ReLU) {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            const Node* inNode = result.node(in.nodeId);
            if (inNode->op == OpKind::Const) {
                newValue = result.addConst(std::max(0.0f, inNode->constValue), n->debugName);
            } else {
                newValue = result.addUnary(OpKind::ReLU, in, n->debugName);
            }
        } else { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            const Node* lhsNode = result.node(lhs.nodeId);
            const Node* rhsNode = result.node(rhs.nodeId);
            if (lhsNode->op == OpKind::Const && rhsNode->op == OpKind::Const) {
                float folded = (n->op == OpKind::Add) ? (lhsNode->constValue + rhsNode->constValue)
                                                        : (lhsNode->constValue * rhsNode->constValue);
                newValue = result.addConst(folded, n->debugName);
            } else {
                newValue = result.addBinary(n->op, lhs, rhs, n->debugName);
            }
        }

        oldIdToNewValue[oldId] = newValue;
    }

    return result;
}

// ==================== commonSubexpressionEliminationPass() (from Section 10.1, unchanged) ====================

static std::string cseKey(const Node* n, const std::map<int, Value>& oldIdToNewValue) {
    char buf[128];
    switch (n->op) {
        case OpKind::Input:
            return "input:" + n->debugName;
        case OpKind::Const:
            snprintf(buf, sizeof(buf), "const:%.9g", n->constValue);
            return buf;
        case OpKind::ReLU: {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            snprintf(buf, sizeof(buf), "relu:%d", in.nodeId);
            return buf;
        }
        default: { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            snprintf(buf, sizeof(buf), "%s:%d,%d", n->op == OpKind::Add ? "add" : "mul", lhs.nodeId, rhs.nodeId);
            return buf;
        }
    }
}

static Graph commonSubexpressionEliminationPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("commonSubexpressionEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    std::map<std::string, Value> keyToExistingValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string key = cseKey(n, oldIdToNewValue);

        auto found = keyToExistingValue.find(key);
        if (found != keyToExistingValue.end()) {
            oldIdToNewValue[oldId] = found->second;
            continue;
        }

        Value newValue;
        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->inputs.size() == 1) {
            newValue = result.addUnary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId), n->debugName);
        } else {
            newValue = result.addBinary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId),
                                         oldIdToNewValue.at(n->inputs[1].nodeId), n->debugName);
        }
        keyToExistingValue[key] = newValue;
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ==================== algebraicSimplificationPass() (from Section 11.1/11.2, unchanged) ====================

static Graph algebraicSimplificationPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("algebraicSimplificationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        Value newValue;

        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->op == OpKind::ReLU) {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            const Node* inNode = result.node(in.nodeId);
            if (inNode->op == OpKind::ReLU) {
                newValue = in;
            } else {
                newValue = result.addUnary(OpKind::ReLU, in, n->debugName);
            }
        } else { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            const Node* lhsNode = result.node(lhs.nodeId);
            const Node* rhsNode = result.node(rhs.nodeId);
            bool lhsIsZero = (lhsNode->op == OpKind::Const && lhsNode->constValue == 0.0f);
            bool rhsIsZero = (rhsNode->op == OpKind::Const && rhsNode->constValue == 0.0f);
            bool lhsIsOne  = (lhsNode->op == OpKind::Const && lhsNode->constValue == 1.0f);
            bool rhsIsOne  = (rhsNode->op == OpKind::Const && rhsNode->constValue == 1.0f);

            if (n->op == OpKind::Add) {
                if (lhsIsZero) newValue = rhs;
                else if (rhsIsZero) newValue = lhs;
                else newValue = result.addBinary(OpKind::Add, lhs, rhs, n->debugName);
            } else { // Mul
                if (lhsIsZero) newValue = lhs;
                else if (rhsIsZero) newValue = rhs;
                else if (lhsIsOne) newValue = rhs;
                else if (rhsIsOne) newValue = lhs;
                else newValue = result.addBinary(OpKind::Mul, lhs, rhs, n->debugName);
            }
        }

        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ==================== deadCodeEliminationPass() (from Section 9.2, unchanged) ====================

static std::set<int> computeLiveNodeIds(const Graph& g, int rootId) {
    std::set<int> live;
    std::deque<int> worklist{rootId};
    while (!worklist.empty()) {
        int id = worklist.front();
        worklist.pop_front();
        if (live.count(id)) continue;
        live.insert(id);
        for (const Value& in : g.node(id)->inputs) {
            worklist.push_back(in.nodeId);
        }
    }
    return live;
}

static Graph deadCodeEliminationPass(const Graph& g) {
    if (g.size() == 0) throw std::runtime_error("deadCodeEliminationPass: an empty graph has no output node");
    int rootId = static_cast<int>(g.size()) - 1;
    std::set<int> live = computeLiveNodeIds(g, rootId);

    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("deadCodeEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
        if (!live.count(oldId)) continue;

        const Node* n = g.node(oldId);
        Value newValue;
        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->inputs.size() == 1) {
            newValue = result.addUnary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId), n->debugName);
        } else {
            newValue = result.addBinary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId),
                                         oldIdToNewValue.at(n->inputs[1].nodeId), n->debugName);
        }
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ==================== PassManager (from Section 8.3, unchanged) ====================

struct NamedPass {
    std::string name;
    TransformPass fn;
};

static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass (%zu nodes) ---\n%s\n", initial.size(), printGraphAsSource(initial).c_str());

    Graph current = passes[0].fn(initial);
    {
        TopoResult topo = topologicalSort(current);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + passes[0].name +
                                      "' produced an invalid (non-acyclic) graph");
        }
    }
    printf("--- after pass '%s' (%zu nodes) ---\n%s\n", passes[0].name.c_str(), current.size(),
           printGraphAsSource(current).c_str());

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& p = passes[i];
        Graph next = p.fn(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + p.name +
                                      "' produced an invalid (non-acyclic) graph");
        }
        printf("--- after pass '%s' (%zu nodes) ---\n%s\n", p.name.c_str(), next.size(),
               printGraphAsSource(next).c_str());
        current = std::move(next);
    }
    return current;
}

int main() {
    // ---------------- PART A: control measurement ----------------
    printf("=== Section 11.3, Part A: algebraicSimplificationPass() ALONE, before any folding ===\n\n");

    // "z" folds to Const(0) under constantFoldPass() -- but here, BEFORE
    // folding runs, it is still a plain Add node. "waste" = mul(y, z)
    // cannot be recognized as "y * 0" while z is still an Add: the
    // simplification pass only ever checks whether an operand IS
    // already a Const, never what it would eventually evaluate to.
    Graph original;
    Value c1 = original.addConst(3.0f, "c1");
    Value c2 = original.addConst(-3.0f, "c2");
    Value z  = original.addBinary(OpKind::Add, c1, c2, "z");     // 3 + (-3) -- folds to Const(0)
    Value a  = original.addInput("a");
    Value b  = original.addInput("b");
    Value y  = original.addBinary(OpKind::Mul, a, b, "y");       // an otherwise-unused subexpression
    Value waste = original.addBinary(OpKind::Mul, y, z, "waste"); // y * z -- NOT foldable/simplifiable yet
    Value x1 = original.addInput("x1");
    Value x2 = original.addInput("x2");
    Value sum1 = original.addBinary(OpKind::Add, x1, x2, "sum1");
    Value sum2 = original.addBinary(OpKind::Add, x1, x2, "sum2"); // literal duplicate of sum1
    Value combined = original.addBinary(OpKind::Add, waste, sum1, "combined");
    original.addBinary(OpKind::Mul, combined, sum2, "out");

    Graph simplifyAloneOnOriginal = algebraicSimplificationPass(original);
    bool simplifyAloneFindsNothing = (simplifyAloneOnOriginal.size() == original.size());
    printf("size: %zu -> %zu\n", original.size(), simplifyAloneOnOriginal.size());
    printf("self-check: simplification alone finds NOTHING before folding runs (%s)\n\n",
           simplifyAloneFindsNothing ? "confirmed" : "MISMATCH");

    // ---------------- PART B: the full four-pass pipeline ----------------
    printf("=== Section 11.3, Part B: [constantFoldPass, commonSubexpressionEliminationPass, algebraicSimplificationPass, deadCodeEliminationPass] ===\n\n");

    Graph finalResult = runPasses(original, {
        {"constantFoldPass", constantFoldPass},
        {"commonSubexpressionEliminationPass", commonSubexpressionEliminationPass},
        {"algebraicSimplificationPass", algebraicSimplificationPass},
        {"deadCodeEliminationPass", deadCodeEliminationPass},
    });

    bool finalSizeCorrect = (finalResult.size() == 4);
    printf("self-check: the full pipeline reduced the graph from %zu nodes to %zu nodes (%s)\n",
           original.size(), finalResult.size(), finalSizeCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"a", 2.0f}, {"b", 7.0f}, {"x1", 3.0f}, {"x2", 4.0f}};
    float originalOut = evaluate(original, inputs).at("out");
    float finalOut = evaluate(finalResult, inputs).at("out");
    bool sameAnswer = (originalOut == finalOut) && (originalOut == 49.0f);  // (x1+x2)^2 = 7^2 = 49
    printf("self-check: evaluate(original, ...).out = %g, evaluate(final, ...).out = %g, both == 49 (%s)\n",
           originalOut, finalOut, sameAnswer ? "confirmed" : "MISMATCH");

    // ---------------- PART C: the last-node-elision hazard ----------------
    printf("\n=== Section 11.3, Part C: when simplification elides the LITERAL last node ===\n\n");

    // A tiny, deliberately minimal graph: "out" itself is a pure
    // additive-identity Add, and it is also the LAST node ever added.
    Graph tiny;
    Value tx = tiny.addInput("x");
    Value tzero = tiny.addConst(0.0f, "zero");
    tiny.addBinary(OpKind::Add, tx, tzero, "out");  // x + 0 -- IS the last node, and WILL be elided

    Graph tinySimplified = algebraicSimplificationPass(tiny);
    printf("Before: %zu nodes, last node is 'out'\n", tiny.size());
    printf("After:  %zu nodes\n", tinySimplified.size());
    printf("%s\n", printGraphAsSource(tinySimplified).c_str());

    // "out" (x + 0) elides to alias x directly -- x was already added
    // FIRST, so it is NOT the new graph's last node. The new graph's
    // actual last node is "zero" -- which is emphatically NOT what "out"
    // used to mean. Treating "the last node is the output" here, the
    // way deadCodeEliminationPass() always does, would silently select
    // the WRONG node.
    bool tinyShrankByOne = (tinySimplified.size() == tiny.size() - 1);
    const Node& tinyLastNode = *tinySimplified.nodes().back();
    bool lastNodeIsWrong = (tinyLastNode.debugName == "zero");
    printf("self-check: the simplified graph's own LAST node is '%s', not the real output (%s)\n",
           tinyLastNode.debugName.c_str(), lastNodeIsWrong ? "confirmed -- this IS the hazard" : "MISMATCH");
    printf("self-check: the graph shrank by exactly one node, as expected, %zu -> %zu (%s)\n",
           tiny.size(), tinySimplified.size(), tinyShrankByOne ? "confirmed" : "MISMATCH");
    printf("(Part B's own graph was deliberately built so its final Mul survives un-elided,\n");
    printf("precisely to avoid the hazard this small graph exists purely to demonstrate.)\n");

    bool allOk = simplifyAloneFindsNothing && finalSizeCorrect && sameAnswer &&
                 tinyShrankByOne && lastNodeIsWrong;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline.cpp -o 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline
./023_folding_cse_simplification_and_dce_closing_part_2s_pipeline
```

**Output:**

```text
=== Section 11.3, Part A: algebraicSimplificationPass() ALONE, before any folding ===

size: 13 -> 13
self-check: simplification alone finds NOTHING before folding runs (confirmed)

=== Section 11.3, Part B: [constantFoldPass, commonSubexpressionEliminationPass, algebraicSimplificationPass, deadCodeEliminationPass] ===

--- before any pass (13 nodes) ---
c1 = const(3)
c2 = const(-3)
z = add(c1, c2)
a = input()
b = input()
y = mul(a, b)
waste = mul(y, z)
x1 = input()
x2 = input()
sum1 = add(x1, x2)
sum2 = add(x1, x2)
combined = add(waste, sum1)
out = mul(combined, sum2)

--- after pass 'constantFoldPass' (13 nodes) ---
c1 = const(3)
c2 = const(-3)
a = input()
b = input()
x1 = input()
x2 = input()
z = const(0)
y = mul(a, b)
sum1 = add(x1, x2)
sum2 = add(x1, x2)
waste = mul(y, z)
combined = add(waste, sum1)
out = mul(combined, sum2)

--- after pass 'commonSubexpressionEliminationPass' (12 nodes) ---
c1 = const(3)
c2 = const(-3)
a = input()
b = input()
x1 = input()
x2 = input()
z = const(0)
y = mul(a, b)
sum1 = add(x1, x2)
waste = mul(y, z)
combined = add(waste, sum1)
out = mul(combined, sum1)

--- after pass 'algebraicSimplificationPass' (10 nodes) ---
c1 = const(3)
c2 = const(-3)
a = input()
b = input()
x1 = input()
x2 = input()
z = const(0)
y = mul(a, b)
sum1 = add(x1, x2)
out = mul(sum1, sum1)

--- after pass 'deadCodeEliminationPass' (4 nodes) ---
x1 = input()
x2 = input()
sum1 = add(x1, x2)
out = mul(sum1, sum1)

self-check: the full pipeline reduced the graph from 13 nodes to 4 nodes (confirmed)
self-check: evaluate(original, ...).out = 49, evaluate(final, ...).out = 49, both == 49 (confirmed)

=== Section 11.3, Part C: when simplification elides the LITERAL last node ===

Before: 3 nodes, last node is 'out'
After:  2 nodes
x = input()
zero = const(0)

self-check: the simplified graph's own LAST node is 'zero', not the real output (confirmed -- this IS the hazard)
self-check: the graph shrank by exactly one node, as expected, 3 -> 2 (confirmed)
(Part B's own graph was deliberately built so its final Mul survives un-elided,
precisely to avoid the hazard this small graph exists purely to demonstrate.)
```

!!! warning "[COMMON TRAP] Assuming 'the last node is the output' survives every pass unscathed"
    Part C is not a hypothetical. It is the direct, concrete fallout of a limitation Section 9.2 stated plainly when it first adopted the "last node is the output" convention: that convention assumes the output is always the most recently created node, and nothing about `Graph` enforces that assumption -- it holds only because every example graph through Chapter 10 happened to be built that way. Algebraic simplification is the first pass in this book that can make that assumption false all on its own: eliding a node OUTRIGHT, rather than folding it into a new one or simply declining to re-add an already-unreferenced one, is exactly the kind of change most likely to remove the specific node "the last node is the output" depends on. Part B's own graph was built with this risk in mind -- its final `Mul` combines two non-constant operands and can never itself match an identity, so it always survives. That carefulness is not automatic, and this book still has no general fix for it: a real "designated output" concept, tracked explicitly rather than inferred from array position, remains open future work, the same honest kind of unresolved question Chapter 9 and Chapter 10 each left standing in their own closing sections.

## Chapter Summary

This chapter added Part 2's fourth and final optimizing pass, `algebraicSimplificationPass()`, plugging into Chapter 8's `TransformPass`/`PassManager` infrastructure with no changes to that infrastructure at all -- the fourth chapter in a row to do so. Section 11.1 built the pass's core shape, structurally the twin of `constantFoldPass()`: at each `Add`, it checks its own already-rebuilt operands against the additive identity (`x + 0 = x`, `0 + x = x`), responding to a match not by computing a new answer but by pointing directly at an operand that already exists. Section 11.2 extended the same pass to Mul's identity element (`x * 1 = x`, `1 * x = x`), Mul's genuinely different absorbing element (`x * 0 = 0`, `0 * x = 0`, which discards the other operand -- and everything it depended on -- entirely), and ReLU's idempotence (`relu(relu(x)) = relu(x)`), and demonstrated all four cascading three levels deep in a single topological walk, the same "check the new graph, not the old one" discipline every real pass in Part 2 has now shared. Section 11.3 closed Part 2 by composing all four of this Part's passes -- folding, CSE, simplification, and dead code elimination -- through Chapter 8's still-unmodified `runPasses()`, proving by direct measurement that folding can expose a simplification opportunity simplification could never find alone, that the simplification which follows creates yet more dead code for the final pass to clean up, and, in a compact separate demonstration, that simplification's own outright elisions make Chapter 9's "last node is the output" convention a genuinely live hazard rather than a theoretical one -- an honest, unresolved question this book carries forward rather than papering over.

## Self-Check Questions

1. `algebraicSimplificationPass()` is described as "structurally the twin" of `constantFoldPass()`. What do the two passes do identically, and what is the one key difference in how each one responds to a match?
2. Why can `algebraicSimplificationPass()` simplify `add(x, 0)` even when `x` is a genuine runtime `Input`, while `constantFoldPass()` could never fold that same node?
3. `mul(x, 0)` and `add(x, 0)` are both algebraic identities, but Section 11.2 describes them as two "genuinely different shapes." What is the difference, and why does it matter for what happens to `x` afterward?
4. In Section 11.2's own cascading test, `s2 = add(t3, t4)` and `s3 = add(s1, s2)` each get elided, even though neither one was originally written as an obvious identity. Why does the pass catch both anyway?
5. In Section 11.3, `algebraicSimplificationPass()` run ALONE on the unfolded original graph finds nothing at the `waste = mul(y, z)` node, even though `z` will eventually be worth exactly `0`. Why can't simplification see that on its own?
6. After `constantFoldPass()`, `commonSubexpressionEliminationPass()`, and `algebraicSimplificationPass()` have all run in Section 11.3's pipeline, `a`, `b`, and `y` all become dead code. Trace, briefly, the specific chain of events that made that happen.
7. Section 11.3's Part C shows a 3-node graph where, after simplification, the new graph's own last node is `zero` rather than the value the original `out` represented. Why does this happen, and why does Part B's own pipeline avoid it?
8. This chapter closes Part 2 with four independently-built optimizing passes, none of which was ever modified to work with any of the others. What single design decision, made all the way back in Chapter 8, is responsible for that?

## Where We Go Next

Part 2 closes with four independently useful, independently tested optimizations -- constant folding, dead code elimination, common subexpression elimination, and algebraic simplification -- composed through infrastructure that has needed zero changes since Chapter 8 first built it, across four separate chapters. Every pass in Part 2 has answered a question about a SINGLE graph in isolation: is this computation already known, is it needed, has it been done before, does arithmetic already make it pointless. Part 3, "Operator Fusion," opens a different kind of question entirely -- not whether a computation is necessary, but whether several NECESSARY computations could be merged into a single kernel to avoid paying for intermediate results no one actually needs to see. Chapter 12, "Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels," begins Part 3 by establishing, with real counted evidence in the same spirit as Chapter 1's own opening argument, exactly what fusion is trying to save.

## Worked Solutions

1. Both passes rebuild the graph in topological order and, at certain nodes, check whether their own already-rebuilt operands match a specific pattern -- both read `oldIdToNewValue` (the NEW graph, not the old one) to do it, letting either pass discover a chain or cascade of matches in a single walk. The one key difference: `constantFoldPass()` responds to a match by COMPUTING a brand new value and building a fresh `Const` node to hold it; `algebraicSimplificationPass()` responds to a match by pointing directly at an operand THAT ALREADY EXISTS in the new graph, building no new node of any kind.
2. `constantFoldPass()` can only fold a computation once EVERY operand involved is a known constant -- it needs concrete numbers to actually compute an answer with. `algebraicSimplificationPass()` never computes anything; it only needs to know that ONE specific operand is the identity element for the operation involved (`0` for `Add`, `1` for `Mul`), a fact about arithmetic itself that holds no matter what the OTHER operand is, known or not. `x` in `add(x, 0)` can be a completely unknown runtime value, and the simplification is still valid.
3. `add(x, 0)` returns `x` -- the OTHER operand -- completely unchanged; nothing about `x` or whatever produced it is discarded. `mul(x, 0)` returns `0` -- a DIFFERENT value than either original operand -- and in doing so, discards `x` (and everything `x` depended on) entirely. The additive identity preserves the non-identity operand; the absorbing element replaces it outright, which is exactly what creates new dead code for `deadCodeEliminationPass()` to find later.
4. Because `algebraicSimplificationPass()` checks each node's operands against the NEW graph being built, not the original one. By the time `s2 = add(t3, t4)` is processed, `t3` and `t4` have ALREADY been remapped (via `oldIdToNewValue`) to both point at the same `zero` node -- so `s2`'s own check sees `add(zero, zero)`, which matches the additive identity itself, and elides. The same thing happens one level further at `s3 = add(s1, s2)`: by the time `s3` is checked, `s2` has already been remapped to `zero`, so `s3`'s check sees `add(s1, zero)`, which matches too. Neither elision was "planned" by the original graph's author -- both are discovered purely by the pass reading its own already-updated state as the walk proceeds.
5. `algebraicSimplificationPass()` only ever checks what an operand ALREADY IS in the graph being rebuilt -- it has no arithmetic of its own, and never evaluates a node to discover what it would eventually compute to. Before `constantFoldPass()` runs, `z` is a plain `Add` node, not a `Const` -- and `algebraicSimplificationPass()`'s check for "is this operand `Const(0.0)`" fails immediately on an `Add` node, regardless of what that `Add` would eventually equal. Only after folding has actually turned `z` into a real `Const(0)` node does the check succeed.
6. `constantFoldPass()` turns `z` (originally `add(c1, c2)`) into `Const(0)`, but leaves `y = mul(a, b)` untouched -- it has no constant operands to fold. `commonSubexpressionEliminationPass()` does its own separate job (merging `sum2` into `sum1`) and leaves `y` untouched too. Only `algebraicSimplificationPass()`, seeing `waste = mul(y, z)` with `z` now `Const(0)`, elides `waste` entirely -- pointing directly at `z` and discarding `y` as an operand completely. Once `waste` no longer references `y`, and nothing else in the graph ever did either, `y` becomes unreachable from the graph's own output -- and `a`/`b`, which only ever fed `y`, become unreachable right along with it. `deadCodeEliminationPass()`, running last, is what actually removes all three.
7. `out = add(x, 0)` in Part C's tiny graph matches the additive identity and elides, aliasing directly to `x` -- which was already added to the new graph EARLIER, as the very first node, long before `out` was ever reached. Because an elided node is never added to the new graph at all, the new graph's actual LAST node ends up being whatever real node happened to be built most recently among the SURVIVORS -- here, `zero`, which was added right before the now-vanished `out`. Part B's own pipeline avoids this because its final `out = mul(combined, sum2)` combines two genuinely non-constant operands (after CSE, `combined` and `sum1`) -- a combination no identity in this chapter's table ever matches, so it is guaranteed to survive as a real node and remain the graph's actual last entry.
8. Chapter 8's decision to give every `TransformPass` the exact same signature -- `Graph(const Graph&)`, reading one graph and returning a brand new one, with no other parameters and no shared mutable state between passes. Because every pass in Part 2 (identity, canonicalization, folding, dead code elimination, common subexpression elimination, algebraic simplification) commits to that one signature, `runPasses()` can thread any sequence of them together, in any order, without needing to know anything about what a specific pass does internally -- the infrastructure and the passes stay completely decoupled, which is exactly why four chapters' worth of independently-written passes could be combined in Section 11.3 without editing a single line of Chapter 8's own code.

---

**Sources cited in this chapter:**

None. `algebraicSimplificationPass()` and its combination with Chapters 9 and 10's own passes through Chapter 8's `runPasses()` are original designs for this book's own `Graph`, built entirely from vocabulary and machinery Chapters 4, 7, 8, 9, and 10 already established. The algebraic identities themselves (additive identity, multiplicative identity, the absorbing element of multiplication, idempotence) are ordinary, general mathematical facts, not specific to any one real compiler or requiring external citation.
