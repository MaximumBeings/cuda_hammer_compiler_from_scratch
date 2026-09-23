# 9. Constant Folding and Dead Code Elimination

**What you will understand:** the book's first two genuinely *optimizing* transform passes -- constant folding (replacing a computation with its already-known answer) and dead code elimination (dropping a node nothing actually depends on) -- and why running them in the wrong order leaves real, findable waste on the table.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph` and `topologicalSort()`, Chapter 7's `printGraphAsSource()`, and Chapter 8's `TransformPass` and `runPasses()` -- this chapter plugs two brand-new passes directly into that last piece of infrastructure, completely unchanged.

---

Chapter 8 built the machinery -- a `TransformPass` signature, a first real pass (`canonicalizeNodeNames()`), and a `PassManager` that runs a sequence of passes safely -- but neither of Chapter 8's own passes made a graph *better* in any measurable sense; canonicalization only changes names. This chapter builds the first two passes that actually do: constant folding, which notices when a computation's answer is already fully determined and replaces the computation with that answer directly, and dead code elimination, which notices when a node's result is never actually used and drops it. Both passes plug into Chapter 8's `TransformPass` signature -- `Graph(const Graph&)` -- with no changes to that signature at all, which is exactly the payoff of Chapter 8 having settled on that design two chapters early.

```text
WHAT THIS CHAPTER ADDS TO CHAPTER 8's OWN INFRASTRUCTURE:

  Chapter 8's passes:              THIS chapter's passes:

    identityPass            -->      constantFoldPass
      (changes nothing)                (replaces a computation whose
    canonicalizeNodeNames             answer is already known with
      (changes NAMES only)            that answer, directly)

                                    deadCodeEliminationPass
                                      (drops a node nothing actually
                                       depends on)

  both have the EXACT signature Chapter 8 already defined --
  Graph(const Graph&) -- so both plug straight into runPasses() with
  ZERO changes to Chapter 8's own PassManager.
```

## 9.1 Constant Folding: Replacing Computation with Its Answer

### Intuition

A recipe that says "preheat the oven to 350 plus 25 degrees" is asking the cook to do arithmetic that has nothing to do with cooking -- the number 375 was always the actual answer, and writing "350 plus 25" instead of just "375" adds a small, pointless step every single time the recipe is followed. A graph can end up with the exact same kind of pointless step: `Const(2)` added to `Const(3)` is *always* going to be `5`, for every single execution of that graph, for every possible input the graph might receive elsewhere -- because neither operand depends on anything that could vary. Computing that addition again every time the graph runs is real, avoidable, repeated work for an answer that was already fully determined the moment the graph was built. Constant folding is the pass that notices this and does the arithmetic once, right now, at compile time, instead of leaving it to be redone at every single run.

Chapter 2's own File 004 already built a constant-folding pass -- but for a much simpler IR, a flat sequence of three-address-code instructions with no graph structure at all. This section builds the real thing for CUDA Hammer's own `Graph`, reusing Chapter 2's core idea (if every operand is already known, replace the computation with the answer) while building it entirely from this book's own `Value`/`Node`/`Graph`/`topologicalSort()` machinery.

```text
THE CORE IDEA, UNCHANGED SINCE CHAPTER 2, NOW APPLIED TO A REAL GRAPH:

  Chapter 2's File 004 (a flat instruction list):       THIS chapter (a real Graph):

    t0 = 2                                                c1 = const(2)
    t1 = 3                                                c2 = const(3)
    t2 = t0 + t1     -->  folds to  -->  t2 = 5            t1 = add(c1, c2)  -->  folds to  -->  t1 = const(5)

  same idea (both operands already known --> compute now, not later),
  completely different IR underneath it.
```

### Background

`constantFoldPass()` rebuilds the graph in topological order -- structurally the same walk Chapter 8's `canonicalizeNodeNames()` already used -- but at every `Add`, `Mul`, and `ReLU`, it checks whether its own already-rebuilt operands turned out to be `Const` nodes *in the new graph*. When they are, it computes the real arithmetic answer immediately and emits a single `Const` node instead of the original operation. When they aren't, it rebuilds the node exactly as it was, unchanged.

```text
constantFoldPass() AT EACH KIND OF NODE:

  Input, Const   -->  rebuilt exactly as-is (nothing to fold -- a leaf
                       has no operands to check in the first place)

  Add, Mul       -->  IF both (already-rebuilt) operands are Const:
                         compute the real answer now, emit ONE Const
                       ELSE:
                         rebuild the Add/Mul exactly as it was

  ReLU           -->  IF its (already-rebuilt) operand is Const:
                         compute max(0, value) now, emit ONE Const
                       ELSE:
                         rebuild the ReLU exactly as it was
```

The detail that makes this fold an entire *chain* of constant computation in a single pass, rather than needing to run repeatedly until nothing changes, is exactly which graph the Const check looks at: the check reads `result.node(...)`, the *new* graph being built, not the old one. A node folded earlier in the same walk is already sitting in the new graph as a `Const` by the time anything downstream of it gets checked -- so that downstream node sees a `Const` operand too, and folds in turn.

```cpp
// Chapter 9: Constant Folding and Dead Code Elimination
// 015_constant_folding_replaces_computation_with_its_answer.cpp
//
// Section 9.1 -- constant folding, this book's first genuinely
// OPTIMIZING transform pass, plugging directly into Chapter 8's own
// TransformPass signature and design.
//
// Chapter 2's File 004 already built a constant-folding pass -- but for
// a completely different, much simpler IR (a flat sequence of
// three-address-code instructions over an arithmetic expression, with
// no notion of Input/Const/Add/Mul/ReLU nodes or a graph structure at
// all). This file builds the real thing for CUDA Hammer's own Graph,
// reusing Chapter 2's own CORE IDEA -- if every operand a computation
// needs is already known at compile time, replace the computation with
// its answer -- but built entirely from this chapter's own Value/Node/
// Graph/topologicalSort()/TransformPass machinery.
//
// constantFoldPass() walks the input graph in topological order (via
// Chapter 4's own topologicalSort(), the same discipline Chapters 6, 7,
// and 8 all already established) and rebuilds every node into a new
// Graph, exactly the way Chapter 8's canonicalizeNodeNames() did --
// except at Add, Mul, and ReLU, it first checks whether its own
// (already-rebuilt) inputs turned out to be Const nodes. If they did,
// it computes the real arithmetic answer right now and emits a single
// Const node instead of the original operation. Because this check
// looks at the NEW graph's own nodes (not the old graph's), a whole
// CHAIN of constant computations folds down to one single Const node
// in a single pass -- no need to run the pass repeatedly until nothing
// changes, for the graphs this chapter's own tests use.
//
// Correctness here means something stronger than "the fold happened" --
// it means the folded graph computes the EXACT SAME ANSWER as the
// original, for every actual input. This file adds a small evaluate()
// interpreter (new here, playing the same verification role Chapter
// 2's own IR interpreter played for ITS constant folding pass) and
// checks the original and folded graphs agree, for concrete input
// values, at the node both graphs still share by id (constantFoldPass
// never drops or reorders nodes -- only Section 9.2's dead code
// elimination will do that).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 015_constant_folding_replaces_computation_with_its_answer.cpp -o 015_constant_folding_replaces_computation_with_its_answer
// Run:     ./015_constant_folding_replaces_computation_with_its_answer
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

// ============================== evaluate() (new): a small interpreter ==============================
//
// Plays the same role Chapter 2's own IR interpreter played for ITS
// constant folding pass: an independent way to compute what a graph
// ACTUALLY produces for concrete input values, so folding can be
// checked against real execution rather than merely trusted by
// construction. Walks topological order, exactly like inferShapes()
// (Chapter 6) and every transform pass in this chapter.
//
// Deliberately keyed by NAME (a node's own debugName), never by node
// id, for both the caller-supplied input values AND the returned
// result -- a real lesson this file's own tests surfaced directly
// while being written: constantFoldPass() rebuilds a graph in
// TOPOLOGICAL order, and topological order is not guaranteed to match
// the ORIGINAL graph's id order once a graph has more than one
// simultaneously-ready leaf (this file's own test graph does -- c1,
// c2, c3, AND x are all leaves, so Kahn's algorithm visits all four
// before it can visit anything that depends on them, which does not
// necessarily preserve their original relative id order once other,
// non-leaf nodes were created in between). A node's id can silently
// shift across a rebuild; comparing two graphs' evaluate() results by
// id would be comparing the wrong nodes half the time. Comparing by
// NAME sidesteps the problem entirely, since every pass in this
// chapter (unlike Chapter 8's own canonicalizeNodeNames()) preserves
// debugName exactly.
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

// ============================== Section 9.1: constantFoldPass() (new) ==============================
//
// Rebuilds the graph node by node, in topological order -- structurally
// the same walk Chapter 8's canonicalizeNodeNames() used -- except at
// Add, Mul, and ReLU, it checks whether its (already-rebuilt) operands
// are Const nodes IN THE NEW GRAPH. Because a node folded earlier in
// this same walk becomes a Const in the new graph immediately, a later
// node that consumes it sees a Const operand too -- letting an entire
// chain of constant computation fold down to one Const in a single
// topological pass, with no need to iterate until nothing changes.
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

// Finds a node by its own debugName -- the SAME "look up by name, not
// id" discipline evaluate() above already needs, for the identical
// reason: after constantFoldPass() rebuilds a graph in topological
// order, a node's NEW id is not guaranteed to match its OLD id (this
// file's own three-constant-plus-one-input test graph is a genuine
// case where it does not -- see evaluate()'s own comment). Any code
// that wants to find "the node that used to be called t1" after a
// rebuild has to search by name, never reuse the old Value's nodeId.
static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) printf(" [value=%g]", n->constValue);
        printf("\n");
    }
}

int main() {
    printf("=== Section 9.1: constantFoldPass(), folding a whole constant chain in one pass ===\n\n");

    // c1 + c2 -> foldable; (c1+c2) * c3 -> ALSO foldable, using the
    // already-folded result; the final add brings in a real runtime
    // input "x", so folding necessarily stops there.
    Graph original;
    Value c1 = original.addConst(2.0f, "c1");
    Value c2 = original.addConst(3.0f, "c2");
    Value c3 = original.addConst(4.0f, "c3");
    Value t1 = original.addBinary(OpKind::Add, c1, c2, "t1");   // 2 + 3 = 5
    Value t2 = original.addBinary(OpKind::Mul, t1, c3, "t2");   // 5 * 4 = 20 (chained fold)
    Value x  = original.addInput("x");
    original.addBinary(OpKind::Add, t2, x, "out");  // 20 + x -- NOT foldable (looked up by name below)

    printf("Original graph:\n");
    printGraph(original);

    Graph folded = constantFoldPass(original);
    printf("\nAfter constantFoldPass():\n");
    printGraph(folded);

    // Looked up by NAME, not by the original t1/t2/out Values' own
    // nodeId -- constantFoldPass() rebuilds in topological order, and
    // (as evaluate()'s own comment explains) this graph's ids genuinely
    // shift across that rebuild, since c1/c2/c3 AND x are all
    // simultaneously-ready leaves. Using the stale old ids here would
    // silently check the WRONG nodes in the new graph.
    const Node* t1InFolded = findNodeByName(folded, "t1");
    const Node* t2InFolded = findNodeByName(folded, "t2");
    const Node* outInFolded = findNodeByName(folded, "out");
    bool t1Folded = t1InFolded->op == OpKind::Const && t1InFolded->constValue == 5.0f;
    bool t2Folded = t2InFolded->op == OpKind::Const && t2InFolded->constValue == 20.0f;
    bool outNotFolded = outInFolded->op == OpKind::Add;
    printf("\nself-check: t1 (2+3) folded to Const(5) (%s)\n", t1Folded ? "confirmed" : "MISMATCH");
    printf("self-check: t2 ((2+3)*4), using t1's ALREADY-folded value, folded to Const(20) (%s)\n",
           t2Folded ? "confirmed" : "MISMATCH");
    printf("self-check: 'out' (20 + x) stays a real Add -- x is not a compile-time constant (%s)\n",
           outNotFolded ? "confirmed" : "MISMATCH");

    // The claim that actually matters: folding did not change what the
    // graph COMPUTES. Evaluate both graphs at a real input value --
    // keyed and looked up by NAME, not id, for the reason explained
    // above evaluate()'s own definition.
    std::map<std::string, float> inputs = {{"x", 6.5f}};
    float originalOut = evaluate(original, inputs).at("out");
    float foldedOut = evaluate(folded, inputs).at("out");
    bool sameAnswer = (originalOut == foldedOut);
    printf("self-check: evaluate(original, x=6.5).out = %g, evaluate(folded, x=6.5).out = %g (%s)\n",
           originalOut, foldedOut, sameAnswer ? "confirmed" : "MISMATCH");

    // A second, independent case: ReLU of a constant.
    Graph reluGraph;
    Value negC = reluGraph.addConst(-2.5f, "negC");
    reluGraph.addUnary(OpKind::ReLU, negC, "reluOut");  // looked up by name below
    Graph reluFolded = constantFoldPass(reluGraph);
    const Node* reluOutInFolded = findNodeByName(reluFolded, "reluOut");
    bool reluFoldedCorrectly = reluOutInFolded->op == OpKind::Const && reluOutInFolded->constValue == 0.0f;
    printf("self-check: relu(const(-2.5)) folds to Const(0) (%s)\n",
           reluFoldedCorrectly ? "confirmed" : "MISMATCH");

    bool allOk = t1Folded && t2Folded && outNotFolded && sameAnswer && reluFoldedCorrectly;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 015_constant_folding_replaces_computation_with_its_answer.cpp -o 015_constant_folding_replaces_computation_with_its_answer
./015_constant_folding_replaces_computation_with_its_answer
```

**Output:**

```text
=== Section 9.1: constantFoldPass(), folding a whole constant chain in one pass ===

Original graph:
  %0 = Const(c1) [value=2]
  %1 = Const(c2) [value=3]
  %2 = Const(c3) [value=4]
  %3 = Add(t1)
  %4 = Mul(t2)
  %5 = Input(x)
  %6 = Add(out)

After constantFoldPass():
  %0 = Const(c1) [value=2]
  %1 = Const(c2) [value=3]
  %2 = Const(c3) [value=4]
  %3 = Input(x)
  %4 = Const(t1) [value=5]
  %5 = Const(t2) [value=20]
  %6 = Add(out)

self-check: t1 (2+3) folded to Const(5) (confirmed)
self-check: t2 ((2+3)*4), using t1's ALREADY-folded value, folded to Const(20) (confirmed)
self-check: 'out' (20 + x) stays a real Add -- x is not a compile-time constant (confirmed)
self-check: evaluate(original, x=6.5).out = 26.5, evaluate(folded, x=6.5).out = 26.5 (confirmed)
self-check: relu(const(-2.5)) folds to Const(0) (confirmed)
```

!!! warning "[COMMON TRAP] Assuming a rebuilt node's new id always matches its old id"
    Look closely at the two printed blocks above: in the ORIGINAL graph, `x` is node `%5`, printed AFTER `t1` (`%3`) and `t2` (`%4`). In the FOLDED graph, `x` is node `%3`, printed BEFORE the folded versions of `t1` and `t2`. Its id genuinely moved. This is not a bug -- it is `topologicalSort()` working exactly as designed: `c1`, `c2`, `c3`, and `x` are ALL leaves (nodes with no inputs of their own), so Kahn's algorithm makes all four immediately "ready" and visits them before anything that depends on them, regardless of what order they happened to be created in relative to `t1` and `t2`. Chapters 6, 7, and 8 each warned, in the abstract, that a rebuilt graph's ids are not guaranteed to match the original's -- this file is where that warning stops being abstract: `findNodeByName()` and `evaluate()`'s own name-keyed design exist specifically because looking a node up by its OLD `Value.nodeId` after a rebuild would silently check the wrong node here, not a hypothetical elsewhere.

## 9.2 Dead Code Elimination: Dropping What Nothing Needs

### Intuition

A shipping warehouse that never checks which crates anyone actually ordered will happily keep building and storing crates nobody will ever pick up -- the resources spent building, labeling, and storing those crates are real, and completely wasted, the moment nothing downstream ever asks for them. A graph can accumulate the exact same kind of waste: a node computed correctly, from real inputs, that nothing else in the graph -- and nothing outside it -- ever actually consumes. Dead code elimination is the pass that finds nodes like this and drops them, finally putting Chapter 8's own "rebuild rather than mutate" design to the purpose it was built for: `Graph` still has no way to remove a node from an existing graph, so "drop a dead node" has only ever meant "don't add it to the new graph being built."

### Background

Deciding whether a node is dead requires answering a question this book's `Graph` has never had a way to ask: which node's value does the *caller* actually want? Every example graph since Chapter 4 has ended with a node named `"out"`, purely by convention -- nothing in `Graph` itself records that fact. Without SOME notion of "this is the value the graph is actually for," dead code elimination has no anchor to reason from at all: every node would trivially qualify as "not used by anything outside the graph," since the graph has no concept of an outside in the first place.

```text
WHY "DEAD" HAS NO MEANING WITHOUT A DESIGNATED OUTPUT:

  a Graph with NO known output:  every node is "unused by anything
                                   outside the graph" -- but so is
                                   EVERY other node, including the one
                                   the caller actually wanted. Nothing
                                   distinguishes "genuinely dead" from
                                   "the actual point of the graph."

  a Graph WITH a known output:   exactly one node is the reason the
                                   graph exists at all -- everything
                                   THAT node transitively depends on is
                                   live; everything else is dead.
```

This section picks the simplest convention that fits every example graph this book has built through Chapter 8: the *last* node added to the graph is treated as that graph's own output. This is a real simplification, not a hidden one -- a language with multiple return values, or an explicit list of "these values must stay live," would need something more general than "whichever node happens to be last." For the single-output graphs this book has used everywhere so far, this convention has one very deliberate payoff: `deadCodeEliminationPass()` keeps the exact signature every other `TransformPass` in this book has, `Graph(const Graph&)` -- no extra "which node is the output" parameter ever needs to be threaded through Chapter 8's own `runPasses()`.

```text
LIVENESS, COMPUTED BY WALKING BACKWARD FROM THE OUTPUT:

  topologicalSort() (Chapter 4) walks FORWARD -- from nodes with no
  inputs, toward nodes that depend on them -- to find a valid order to
  COMPUTE a graph in.

  computeLiveNodeIds() (this section) walks BACKWARD -- from the
  designated output, through each node's OWN `inputs` field, toward
  whatever it depends on -- to find every node that output actually
  NEEDS. Same field (`inputs`), opposite direction, completely
  different question being answered.

  a node reached by this backward walk is LIVE. everything else --
  reachable from NOTHING the output depends on -- is DEAD.
```

One detail worth being precise about: a node with more than one live consumer (this book's own diamond graph's `t1`, needed by both `t2` and `t3`) is still visited, and kept, exactly once. `computeLiveNodeIds()` uses a `std::set<int>` specifically because liveness is a yes/no question about a node, never a count of how many paths reach it -- the backward walk simply skips a node it has already marked live, the same way `identityPass()` and `constantFoldPass()` skip nothing and instead correctly reuse a single, already-built `Value` whenever more than one downstream node references the same input.

```cpp
// Chapter 9: Constant Folding and Dead Code Elimination
// 016_dead_code_elimination_from_the_graphs_own_output.cpp
//
// Section 9.2 -- dead code elimination, the first real use of Chapter
// 8's own "rebuild rather than mutate" design, put to exactly the
// purpose Chapter 8 built it for: dropping nodes nothing actually
// depends on.
//
// Dead code elimination needs one thing Chapter 4's Graph has never
// had a concept of: an actual OUTPUT. Every example graph in this book
// so far has ended with a node named "out," by convention, but nothing
// in Graph itself has ever recorded "this is the value the caller
// actually cares about" -- Graph is just a set of nodes and edges.
// Without knowing which node's value the graph is FOR, "is this node
// dead" has no answer at all: every node in a graph with no designated
// output is trivially "used by nothing outside the graph," which would
// make deadCodeEliminationPass delete everything.
//
// This file picks the simplest possible convention that fits every
// example graph this book has built so far: the LAST node added to the
// graph (the highest id) is treated as the graph's own output. This is
// a real simplification, stated plainly rather than hidden -- a
// compiler for a language with multiple return values, or explicit
// "these N values are live outputs" annotations, would need something
// more general. For a graph with exactly one output, which is every
// graph this book has built through Chapter 8, "the last node is the
// output" is enough, and it has one very convenient side effect: it
// keeps deadCodeEliminationPass's own signature EXACTLY `Graph(const
// Graph&)`, identical to every other TransformPass this chapter and
// Chapter 8 define -- no extra "which node is the output" parameter
// ever needs to be threaded through Chapter 8's own PassManager.
//
// "Dead," precisely: a node is LIVE if it is the designated output, or
// if some other LIVE node depends on it (directly or transitively).
// Everything else is dead. This is computed by walking BACKWARD from
// the output through each node's own `inputs` -- the same field every
// other pass in this book already reads, just walked in the opposite
// direction from topologicalSort()'s own forward walk.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 016_dead_code_elimination_from_the_graphs_own_output.cpp -o 016_dead_code_elimination_from_the_graphs_own_output
// Run:     ./016_dead_code_elimination_from_the_graphs_own_output
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
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

static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

// ============================== Section 9.2: liveness and DCE (new) ==============================

using TransformPass = std::function<Graph(const Graph&)>;

// Walks BACKWARD from rootId through each node's own `inputs` -- the
// opposite direction from topologicalSort()'s forward walk -- marking
// every node reached along the way as live. A node with two live
// consumers (like this book's own diamond graph's own t1) is still
// visited and inserted only once, since `live` is a set: reachability
// doesn't care HOW MANY paths lead to a node, only whether at least
// one does.
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

// Rebuilds the graph keeping only LIVE nodes -- a dead node is simply
// never re-added, exactly the "rebuild rather than mutate" pattern
// Chapter 8 established, now finally used for the one thing it was
// always meant to make possible: this book's Graph still has no way to
// REMOVE a node from an existing graph, so dropping one has always
// meant "don't add it to the new graph in the first place."
static Graph deadCodeEliminationPass(const Graph& g) {
    if (g.size() == 0) throw std::runtime_error("deadCodeEliminationPass: an empty graph has no output node");
    int rootId = static_cast<int>(g.size()) - 1;  // convention: the last node IS the graph's own output
    std::set<int> live = computeLiveNodeIds(g, rootId);

    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("deadCodeEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
        if (!live.count(oldId)) continue;  // DEAD -- skipped, never added to the new graph

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

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
    }
}

int main() {
    printf("=== Section 9.2: deadCodeEliminationPass(), dropping a node nothing depends on ===\n\n");

    // The familiar diamond graph, PLUS one genuinely dead node ("junk")
    // inserted in the middle -- computed from real graph nodes (a, b),
    // so it LOOKS legitimate, but nothing that "out" (the last node,
    // this graph's own designated output) depends on ever reaches it.
    Graph withJunk;
    Value a  = withJunk.addInput("a");
    Value b  = withJunk.addInput("b");
    Value t1 = withJunk.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = withJunk.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = withJunk.addUnary(OpKind::ReLU, t1, "t3");
    withJunk.addBinary(OpKind::Add, a, b, "junk");             // DEAD: computed, never consumed
    withJunk.addBinary(OpKind::Add, t2, t3, "out");            // the graph's own output (last node)

    printf("Graph with one dead node ('junk'):\n");
    printGraph(withJunk);

    Graph cleaned = deadCodeEliminationPass(withJunk);
    printf("\nAfter deadCodeEliminationPass():\n");
    printGraph(cleaned);

    bool junkRemoved = (cleaned.size() == withJunk.size() - 1);
    bool everyRemainingNodeIsLive = true;
    for (const auto& n : cleaned.nodes()) {
        if (n->debugName == "junk") everyRemainingNodeIsLive = false;
    }
    printf("\nself-check: exactly one node was dropped, %zu -> %zu (%s)\n",
           withJunk.size(), cleaned.size(), junkRemoved ? "confirmed" : "MISMATCH");
    printf("self-check: 'junk' does not appear anywhere in the cleaned graph (%s)\n",
           everyRemainingNodeIsLive ? "confirmed" : "MISMATCH");

    // t1 has TWO live consumers (t2 and t3) -- confirm it was kept
    // exactly ONCE, not duplicated, and that both of its consumers
    // still correctly reference it in the new graph.
    const Node* t1InCleaned = findNodeByName(cleaned, "t1");
    const Node* t2InCleaned = findNodeByName(cleaned, "t2");
    const Node* t3InCleaned = findNodeByName(cleaned, "t3");
    int t1OccurrenceCount = 0;
    for (const auto& n : cleaned.nodes()) if (n->debugName == "t1") ++t1OccurrenceCount;
    bool t2ReferencesT1 = t2InCleaned->inputs[0].nodeId == t1InCleaned->id;
    bool t3ReferencesT1 = t3InCleaned->inputs[0].nodeId == t1InCleaned->id;
    printf("self-check: 't1' (2 live consumers) appears exactly once in the cleaned graph,\n");
    printf("and both 't2' and 't3' still correctly reference that ONE copy (%s)\n",
           (t1OccurrenceCount == 1 && t2ReferencesT1 && t3ReferencesT1) ? "confirmed" : "MISMATCH");

    // Control case: a graph with NO dead nodes at all should come back
    // completely unchanged in size -- DCE must not misfire on clean input.
    Graph clean;
    Value ca = clean.addInput("a");
    Value cb = clean.addInput("b");
    Value ct1 = clean.addBinary(OpKind::Add, ca, cb, "t1");
    Value ct2 = clean.addBinary(OpKind::Mul, ct1, ca, "t2");
    Value ct3 = clean.addUnary(OpKind::ReLU, ct1, "t3");
    clean.addBinary(OpKind::Add, ct2, ct3, "out");
    Graph stillClean = deadCodeEliminationPass(clean);
    bool nothingRemoved = (stillClean.size() == clean.size());
    printf("\nself-check: a graph with NO dead nodes comes back the same size, %zu -> %zu\n",
           clean.size(), stillClean.size());
    printf("(DCE does not misfire on already-clean input) (%s)\n",
           nothingRemoved ? "confirmed" : "MISMATCH");

    bool allOk = junkRemoved && everyRemainingNodeIsLive && t1OccurrenceCount == 1 &&
                 t2ReferencesT1 && t3ReferencesT1 && nothingRemoved;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 016_dead_code_elimination_from_the_graphs_own_output.cpp -o 016_dead_code_elimination_from_the_graphs_own_output
./016_dead_code_elimination_from_the_graphs_own_output
```

**Output:**

```text
=== Section 9.2: deadCodeEliminationPass(), dropping a node nothing depends on ===

Graph with one dead node ('junk'):
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1)
  %3 = Mul(t2)
  %4 = ReLU(t3)
  %5 = Add(junk)
  %6 = Add(out)

After deadCodeEliminationPass():
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1)
  %3 = Mul(t2)
  %4 = ReLU(t3)
  %5 = Add(out)

self-check: exactly one node was dropped, 7 -> 6 (confirmed)
self-check: 'junk' does not appear anywhere in the cleaned graph (confirmed)
self-check: 't1' (2 live consumers) appears exactly once in the cleaned graph,
and both 't2' and 't3' still correctly reference that ONE copy (confirmed)

self-check: a graph with NO dead nodes comes back the same size, 6 -> 6
(DCE does not misfire on already-clean input) (confirmed)
```

!!! note "Why 'junk' had to be built from real graph nodes, not just dropped in as an orphan"
    `junk = add(a, b)` is not a trivially-obviously-useless node -- it reads two perfectly real, live inputs and computes a perfectly real sum. That is deliberate: a dead code eliminator that only caught nodes referencing nothing at all would be nearly useless in practice, since real dead code almost always comes from a computation that LOOKS legitimate in isolation but simply never gets used by anything the program actually needed. Building `junk` from genuine, live inputs is what makes this test a fair one -- `deadCodeEliminationPass()` has to notice that `junk` itself has no live consumer, not merely that its own inputs are suspicious.

## 9.3 Combining Both Passes Through Chapter 8's Pass Manager

### Intuition

Paying off a mortgage early does not just eliminate the payment itself -- it can also free up money that was going toward mortgage-related insurance, which in turn might free up something else, in a chain that was never visible while the mortgage was still active. Constant folding has exactly this same second-order effect on a graph: folding `(c1 + c2) * c3` down to a single `Const(20)` does not just simplify that one computation -- it also *severs* every edge that used to point back to `c1`, `c2`, `c3`, and the intermediate `t1`, since the new `Const(20)` node has no inputs at all. Those four nodes were genuinely needed before the fold; after it, nothing in the graph depends on any of them anymore. Folding does not know or care about this side effect -- it rebuilds every single node in the graph, whether or not anything downstream still needs it. Noticing and cleaning up the wreckage is dead code elimination's job, and it can only do that job if it runs *after* folding has created the mess for it to find.

```text
WHAT FOLDING ALONE LEAVES BEHIND (SAME GRAPH AS SECTION 9.1):

  before folding:  c1, c2, c3 --> t1 --> t2 --> out
                                          (x also feeds out)

  after folding:   c1, c2, c3    t1    t2=Const(20) --> out
                   (still built,  (still built,   (x also feeds out)
                    but nothing    but nothing
                    references     references
                    them anymore)  it anymore)

  constantFoldPass() rebuilds EVERY node, live or not -- it has no
  notion of "live" at all, that is Section 9.2's own job entirely.
```

### Background

Both `constantFoldPass()` and `deadCodeEliminationPass()` already have the exact signature Chapter 8's `TransformPass` requires -- neither needed a single change to plug directly into Chapter 8's own `runPasses()`. File 017 below reuses `runPasses()` completely unchanged, running `[constantFoldPass, deadCodeEliminationPass]` over File 015's own seven-node graph, and prints a node count alongside every before/after dump so the size reduction is visible at each step, not just at the end.

```text
THE PIPELINE, NODE COUNTS MADE EXPLICIT AT EVERY STEP:

  before any pass:                7 nodes  (c1, c2, c3, t1, t2, x, out)
  after constantFoldPass:         7 nodes  (t1, t2 became Const --
                                             SAME count, folding never
                                             drops a node, it only
                                             changes what a node computes)
  after deadCodeEliminationPass:  3 nodes  (c1, c2, c3, and the old t1
                                             are all gone -- NOTHING in
                                             the folded graph still
                                             depends on them)

  the size reduction happens ENTIRELY in the second pass -- folding's
  own contribution was making that reduction POSSIBLE, not causing it
  directly.
```

The order-matters claim is worth proving directly, not just asserting: File 017 also runs `deadCodeEliminationPass()` completely alone, on the *original*, unfolded graph. In the unfolded graph, `c1`, `c2`, `c3`, and `t1` are all still genuinely needed -- `t1` really does compute `Add(c1, c2)`, and `t2` really does need that result -- so DCE alone finds every single node live and removes nothing at all. The 7-node-to-3-node reduction earlier is not something either pass could produce by itself; it specifically requires folding to run first, creating the dead nodes, and DCE to run second, finding them.

```text
WHY ORDER SPECIFICALLY MATTERS HERE (NOT JUST "BOTH PASSES SHOULD RUN"):

  DCE alone, on the UNFOLDED graph:
    c1, c2, c3, t1 are all genuinely, actually needed by t2 -->
    nothing is dead --> 7 nodes in, 7 nodes out, NO reduction at all

  constantFoldPass, THEN deadCodeEliminationPass:
    fold first: t1, t2 become Const nodes with no inputs at all -->
    c1, c2, c3, and the OLD t1 are now unreachable from "out" -->
    DCE finds them --> 7 nodes in, 3 nodes out

  the difference is not "DCE didn't run" -- it's that DCE was asked
  the right question ("what's live NOW") only in the second ordering.
```

```cpp
// Chapter 9: Constant Folding and Dead Code Elimination
// 017_folding_and_dce_together_through_chapter_8s_pass_manager.cpp
//
// Section 9.3 -- running constantFoldPass and deadCodeEliminationPass
// TOGETHER, through Chapter 8's own unmodified runPasses(), and seeing
// why the ORDER genuinely matters: folding does not just simplify
// existing computation, it actively CREATES new dead code, which only
// becomes visible -- and only gets cleaned up -- once DCE runs AFTER it.
//
// Both of this chapter's own passes already have the exact signature
// Chapter 8's TransformPass requires, `Graph(const Graph&)` -- neither
// one needed any change at all to plug directly into Chapter 8's own
// runPasses(). This file is the payoff of that design choice: Part 2's
// first real, multi-pass optimization pipeline, built entirely by
// composing two independently-tested passes through infrastructure
// that was finished two chapters ago.
//
// Reuses File 015's constantFoldPass() and evaluate(), File 016's
// deadCodeEliminationPass(), Chapter 7's printGraphAsSource(), and
// Chapter 8's runPasses()/NamedPass -- all completely unchanged.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 017_folding_and_dce_together_through_chapter_8s_pass_manager.cpp -o 017_folding_and_dce_together_through_chapter_8s_pass_manager
// Run:     ./017_folding_and_dce_together_through_chapter_8s_pass_manager
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

// ==================== evaluate() (from File 015, unchanged) ====================

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
        } else {
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== TransformPass / constantFoldPass() (from Section 9.1) ====================

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
        } else {
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

// ==================== deadCodeEliminationPass() (from Section 9.2) ====================

static std::set<int> computeLiveNodeIds(const Graph& g, int rootId) {
    std::set<int> live;
    std::deque<int> worklist{rootId};
    while (!worklist.empty()) {
        int id = worklist.front();
        worklist.pop_front();
        if (live.count(id)) continue;
        live.insert(id);
        for (const Value& in : g.node(id)->inputs) worklist.push_back(in.nodeId);
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

// ============================== Section 9.3: runPasses() (from Chapter 8, unchanged) ==============================

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
            throw std::runtime_error("runPasses: pass '" + passes[0].name + "' produced an invalid graph");
        }
    }
    printf("--- after pass '%s' (%zu nodes) ---\n%s\n",
           passes[0].name.c_str(), current.size(), printGraphAsSource(current).c_str());

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& p = passes[i];
        Graph next = p.fn(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + p.name + "' produced an invalid graph");
        }
        printf("--- after pass '%s' (%zu nodes) ---\n%s\n",
               p.name.c_str(), next.size(), printGraphAsSource(next).c_str());
        current = std::move(next);
    }
    return current;
}

int main() {
    printf("=== Section 9.3: constant folding creates dead code; DCE only helps if it runs AFTER ===\n\n");

    // File 015's own graph: a foldable constant chain (c1+c2)*c3,
    // combined with a real runtime input x. "out" is this graph's own
    // designated output (the last node).
    Graph original;
    Value c1 = original.addConst(2.0f, "c1");
    Value c2 = original.addConst(3.0f, "c2");
    Value c3 = original.addConst(4.0f, "c3");
    Value t1 = original.addBinary(OpKind::Add, c1, c2, "t1");
    Value t2 = original.addBinary(OpKind::Mul, t1, c3, "t2");
    Value x  = original.addInput("x");
    original.addBinary(OpKind::Add, t2, x, "out");

    printf(">>> Pipeline: [constantFoldPass, deadCodeEliminationPass]\n\n");
    Graph finalResult = runPasses(original, {
        {"constantFoldPass", constantFoldPass},
        {"deadCodeEliminationPass", deadCodeEliminationPass},
    });

    // After folding, t2 becomes a plain Const(20) with NO inputs at
    // all -- which means c1, c2, c3, AND the old t1 are now reachable
    // from NOTHING "out" depends on, even though constantFoldPass
    // itself still faithfully re-emitted all of them (it rebuilds
    // EVERY node, it does not know or care which ones anything else
    // still needs -- that is precisely DCE's job, not folding's).
    printf("self-check: the graph shrank from %zu nodes to %zu nodes -- folding severed the\n",
           original.size(), finalResult.size());
    printf("dependency chain back to c1/c2/c3/t1, and DCE then dropped all four (%s)\n",
           (finalResult.size() == 3) ? "confirmed" : "MISMATCH");

    bool onlyThreeNamesRemain = finalResult.size() == 3;
    bool namesAreOutT2X = onlyThreeNamesRemain; // checked precisely below
    if (onlyThreeNamesRemain) {
        std::set<std::string> remainingNames;
        for (const auto& n : finalResult.nodes()) remainingNames.insert(n->debugName);
        namesAreOutT2X = remainingNames == std::set<std::string>{"out", "t2", "x"};
    }
    printf("self-check: exactly {t2, x, out} remain -- c1, c2, c3, and the original t1 are\n");
    printf("all gone (%s)\n", namesAreOutT2X ? "confirmed" : "MISMATCH");

    // The claim that actually matters, same as File 015: two whole
    // optimization passes later, the graph still computes the exact
    // same answer for a real input.
    std::map<std::string, float> inputs = {{"x", 6.5f}};
    float originalOut = evaluate(original, inputs).at("out");
    float finalOut = evaluate(finalResult, inputs).at("out");
    bool sameAnswer = (originalOut == finalOut);
    printf("self-check: evaluate(original, x=6.5).out = %g, evaluate(final, x=6.5).out = %g (%s)\n",
           originalOut, finalOut, sameAnswer ? "confirmed" : "MISMATCH");

    // The order-matters claim: running DCE BEFORE folding would find
    // EVERY node live (c1/c2/c3/t1 all still genuinely feed "out" in
    // the UNFOLDED graph), so it would remove nothing at all -- proving
    // the size reduction above specifically requires folding to run
    // FIRST, not just requires both passes to run in some order.
    Graph dceFirst = deadCodeEliminationPass(original);
    bool dceAloneRemovedNothing = (dceFirst.size() == original.size());
    printf("\nself-check: running deadCodeEliminationPass() ALONE, before any folding, removes\n");
    printf("NOTHING (%zu -> %zu nodes) -- every node is genuinely live in the UNFOLDED graph (%s)\n",
           original.size(), dceFirst.size(), dceAloneRemovedNothing ? "confirmed" : "MISMATCH");

    bool allOk = (finalResult.size() == 3) && namesAreOutT2X && sameAnswer && dceAloneRemovedNothing;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 017_folding_and_dce_together_through_chapter_8s_pass_manager.cpp -o 017_folding_and_dce_together_through_chapter_8s_pass_manager
./017_folding_and_dce_together_through_chapter_8s_pass_manager
```

**Output:**

```text
=== Section 9.3: constant folding creates dead code; DCE only helps if it runs AFTER ===

>>> Pipeline: [constantFoldPass, deadCodeEliminationPass]

--- before any pass (7 nodes) ---
c1 = const(2)
c2 = const(3)
c3 = const(4)
t1 = add(c1, c2)
t2 = mul(t1, c3)
x = input()
out = add(t2, x)

--- after pass 'constantFoldPass' (7 nodes) ---
c1 = const(2)
c2 = const(3)
c3 = const(4)
x = input()
t1 = const(5)
t2 = const(20)
out = add(t2, x)

--- after pass 'deadCodeEliminationPass' (3 nodes) ---
x = input()
t2 = const(20)
out = add(t2, x)

self-check: the graph shrank from 7 nodes to 3 nodes -- folding severed the
dependency chain back to c1/c2/c3/t1, and DCE then dropped all four (confirmed)
self-check: exactly {t2, x, out} remain -- c1, c2, c3, and the original t1 are
all gone (confirmed)
self-check: evaluate(original, x=6.5).out = 26.5, evaluate(final, x=6.5).out = 26.5 (confirmed)

self-check: running deadCodeEliminationPass() ALONE, before any folding, removes
NOTHING (7 -> 7 nodes) -- every node is genuinely live in the UNFOLDED graph (confirmed)
```

!!! warning "[COMMON TRAP] Assuming a single run of each pass is always enough"
    This chapter's own examples fold and eliminate dead code completely in one pass each -- but that is a property of THESE particular graphs, not a guarantee `constantFoldPass()` or `deadCodeEliminationPass()` makes in general. A more elaborate graph could easily need folding, then DCE, then MORE folding that DCE's own removals happened to expose, and so on. A real pass manager typically runs a fixed sequence like this one to a "fixed point" -- repeating it until an entire pass over the graph produces no change at all -- rather than assuming two passes, run once each, are always the end of the story. Chapter 8's own `runPasses()` does not build that repeat-to-a-fixed-point logic; it runs exactly the sequence it's given, exactly once. Extending it to iterate is a natural next step this book leaves as an open question rather than a solved one -- worth keeping in mind heading into Chapter 10's common subexpression elimination, which will raise the same "could running this again find more" question from a different angle.

## Chapter Summary

This chapter built Part 2's first two genuinely optimizing passes, both plugging into Chapter 8's `TransformPass`/`PassManager` infrastructure with no changes to that infrastructure at all. Section 9.1 built `constantFoldPass()`, reusing Chapter 2's own core constant-folding idea (if every operand is already known, compute the answer now instead of at every run) but built entirely for CUDA Hammer's real `Graph`, folding an entire chain of constant computation in a single topological pass by checking each operand against the NEW graph being built, not the old one -- and, while testing it, surfacing a genuine instance of the "a rebuilt graph's ids don't have to match the original's" warning Chapters 6 through 8 had each only stated in the abstract. Section 9.2 built `deadCodeEliminationPass()`, the first real use of Chapter 8's own "rebuild rather than mutate" design for the purpose it was built for, using the simplest workable convention (the last node is the graph's own output) to compute liveness by walking backward through each node's `inputs`, and dropping anything unreachable from that output. Section 9.3 combined both through Chapter 8's own unmodified `runPasses()`, proving directly -- not just asserting -- that order matters: constant folding actively creates new dead code by severing dependency edges, and dead code elimination can only clean up what folding exposes when it runs afterward, shrinking a real seven-node graph down to three while `evaluate()` confirms the computed answer never changed. Chapter 10 continues Part 2 with common subexpression elimination, the next real optimization to plug into this same, now well-exercised, pass infrastructure.

## Self-Check Questions

1. Why does `constantFoldPass()` check whether an operand is `Const` in the NEW graph being built, rather than checking the OLD graph's own node? What specific capability would be lost if it checked the old graph instead?
2. This file's own tests surfaced a real case where a rebuilt graph's node ids do not match the original graph's ids. What specific property of the test graph caused that, and why didn't Chapter 8's own diamond-graph examples ever show the same thing?
3. Why does `deadCodeEliminationPass()` need SOME notion of "the graph's own output" to do its job at all, rather than being able to compute liveness from the graph's structure alone?
4. What real limitation does the "the last node is the graph's own output" convention have, and why was it still a reasonable choice for this chapter to make?
5. `computeLiveNodeIds()` walks backward through each node's `inputs`, the exact same field `topologicalSort()` reads to walk FORWARD. Why does the same field support both a forward and a backward walk, and what does each walk actually answer?
6. Why does `t1`, in File 016's own diamond-plus-junk graph, appear exactly once in the cleaned output despite having two live consumers (`t2` and `t3`)?
7. File 017 runs `deadCodeEliminationPass()` alone, on the UNFOLDED graph, and confirms it removes nothing. What does that specific test prove that running the full `[constantFoldPass, deadCodeEliminationPass]` pipeline, by itself, would not have proven?
8. Section 9.3's own [COMMON TRAP] points out that a more elaborate graph could need folding and DCE to alternate more than once each. Why doesn't `constantFoldPass()` and `deadCodeEliminationPass()`, run once each in sequence, already guarantee a fully optimized result for every possible graph?

## Where We Go Next

Part 2 now has two independently useful, independently tested optimizations, composed through infrastructure that needed no changes at all to support them. Chapter 10, "Common Subexpression Elimination for Tensor Graphs," continues in exactly this pattern: a third `TransformPass`, plugging into the same `runPasses()`, finding computations that are not merely foldable to a constant but structurally IDENTICAL to some other computation already present elsewhere in the graph -- and reusing the existing result instead of computing the same thing twice.

## Worked Solutions

1. Checking the NEW graph lets a node folded EARLIER in the same walk (say, `t1` becoming `Const(5)`) be seen as a `Const` by whatever consumes it LATER in that same walk (`t2`, computing `t1 * c3`) -- letting an entire chain fold in one pass. Checking the OLD graph instead would only ever see `t1` as the `Add` it originally was, since the old graph is never modified -- `t2` would then see a non-Const operand and fail to fold, even though the value it needs is, in fact, already fully known. The pass would need to run repeatedly, checking the old graph state after each run, to achieve what checking the new graph achieves in one pass.
2. The test graph has FOUR simultaneously-ready leaves (`c1`, `c2`, `c3`, and `x`) created in an order where `x` was added AFTER two non-leaf nodes (`t1`, `t2`) that themselves depend on the earlier constants. `topologicalSort()`'s Kahn's-algorithm walk visits every zero-in-degree node (every leaf) before it can visit anything that depends on them, so all four leaves get visited together, in their OWN relative order, before `t1`/`t2` -- which shifts `x`'s position earlier than its original id suggested. Chapter 8's own diamond graph never showed this because its own two leaves (`a`, `b`) were BOTH created first, before any non-leaf node existed -- so topological order and creation order happened to coincide for that specific graph, the exact coincidence Chapters 6 through 8 kept warning not to rely on.
3. Because "dead" only has meaning relative to something the graph is trying to produce -- without a designated output, every node is equally "not referenced by anything outside the graph," since the graph has no formal notion of an "outside" at all. There would be no way to distinguish a node nothing needs from the one node that is the entire reason the graph was built, so liveness could not be computed from structure alone; it needs an explicit starting point to walk backward from.
4. The real limitation: it assumes exactly one output per graph, and specifically that the output is always the LAST node added -- a graph needing multiple live outputs, or an output that is not the most recently created node, could not be expressed under this convention at all. It was still a reasonable choice for this chapter because every example graph built through Chapter 8 already follows exactly this shape (one output, always named "out," always added last), and the payoff is real: `deadCodeEliminationPass()` keeps the plain `Graph(const Graph&)` signature every other `TransformPass` in this book has, requiring no change to Chapter 8's own `PassManager` to support it.
5. `inputs` records, for every node, which OTHER nodes it directly depends on -- that is simply what an edge in this graph means. `topologicalSort()` reads it forward (from a node, find what must come before it) to answer "in what order can every node be correctly computed." `computeLiveNodeIds()` reads the exact same field backward (from a node, find what it itself depends on) to answer a completely different question: "starting from the one node that matters, what else does REACHING it actually require." Same data, two different traversal directions, two different questions.
6. `computeLiveNodeIds()` uses a `std::set<int>` to track which nodes have been marked live, and checks `live.count(id)` before processing a node -- once `t1` has been visited (from either `t2` or `t3`, whichever the backward walk reaches first), it is already marked live, so reaching it again from the other consumer does nothing further. When `deadCodeEliminationPass()` later rebuilds the graph, it walks the SET of live ids (each id appearing once, by definition of a set) in topological order, so `t1` is rebuilt exactly once, producing exactly one new `Value` that both `t2` and `t3` are correctly remapped to reference.
7. Running the full pipeline once shows the END RESULT (a 7-node graph became a 3-node graph) but does not, by itself, distinguish between two different possible explanations: "DCE alone would have found this reduction regardless of what ran before it" versus "the reduction genuinely required folding to run first." Running `deadCodeEliminationPass()` alone on the unfolded graph and showing it removes NOTHING rules out the first explanation directly -- proving the size reduction is not something DCE could have found on its own, no matter when it ran, without folding having created the dead code for it to find first.
8. Because folding and DCE each only look at what's DIRECTLY in front of them in a single pass -- folding only combines operands that are ALREADY `Const` by the time it reaches a node, and DCE only removes what's unreachable from the output AS THE GRAPH STANDS when it runs. Neither pass re-examines its own earlier work in light of what the OTHER pass just did. A graph where DCE's own removal of some dead branch happened to leave behind a NEW foldable constant expression (one that folding's own single earlier pass had no way to see, because the relevant nodes were still entangled with now-removed ones) would need folding to run a second time to catch it -- which is exactly the scenario the chapter's own [COMMON TRAP] flags as a real, unaddressed gap in running each pass "once, in sequence."

---

**Sources cited in this chapter:**

None. `constantFoldPass()`, `deadCodeEliminationPass()`, and their combination through Chapter 8's `runPasses()` are original designs for this book's own `Graph`, built entirely from vocabulary and machinery Chapters 2, 4, 6, 7, and 8 already established. Constant folding's own core idea is the same one Chapter 2's File 004 already used for a much simpler IR -- an internal callback to this book's own earlier work, not an external citation.
