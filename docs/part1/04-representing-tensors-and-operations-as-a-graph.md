# 4. Representing Tensors and Operations as a Graph

**What you will understand:** why CUDA Hammer's own IR, starting in this chapter, is a graph rather than a tree or a straight-line list -- and the two real building blocks (`Node`, `Value`) every later chapter's passes, fusion, codegen, and autotuner all operate on. This is CUDA Hammer's own first real code: genuinely compiled, genuinely run C++, not a toy arithmetic language anymore.

**What you need to know first:** Chapter 2's vocabulary (IR, pass, codegen) and, specifically, that Chapter 2's own IR was always a tree (as an AST) or a straight-line list (as linear IR) -- this chapter explains exactly why a tensor computation cannot stay that way.

---

Part 1 starts CUDA Hammer itself: no more parsing arithmetic expression text, no more single-letter variables. From here forward, every chapter builds on the same graph data structure this chapter designs, compiles, and tests.

```text
CHAPTER 2's LINEAR IR (every temp is consumed by AT MOST one later
instruction -- guaranteed by construction, since it came from walking
PARSED TEXT, where each subexpression appears exactly once):

  t0 = a + b
  t1 = t0 * c          -- t0 is used here, and only here, ever

CHAPTER 4's GRAPH (built through an API, not parsed from text -- the
exact same computed Value can be handed to more than one later call):

  t1 = Add(a, b)
  t2 = Mul(t1, a)       -- t1 used here...
  t3 = ReLU(t1)         -- ...AND here: the SAME t1, not a second copy,
                           not recomputed -- this is what makes it a
                           graph instead of a tree or a straight line.
```

## 4.1 Why a Graph, Not a Tree or a Linear List

### Intuition

Picture a well shared by two houses through a single pipe that splits at a T-joint: the water is drawn from the ground once, and both houses draw from that same supply. Now picture the alternative -- each house paying to dig its own separate well, doing the identical work of reaching the same water table twice, just so each house can have "its own" supply instead of sharing one. A tree is the second arrangement: nothing in a tree can be reached by two different paths, so if two different places in a computation both need the same result, a tree has no way to represent that except by computing it twice, in two separate, unconnected parts of the tree.

```text
ONE well, shared by two houses (a graph):
  well --> pipe --> [house A tap]
                 --> [house B tap]
  the water in the pipe is drawn once; both houses draw from the SAME
  supply, never a second, separately-dug well.

TWO wells, one per house (what a TREE would require instead):
  well A --> pipe --> [house A tap]
  well B --> pipe --> [house B tap]
  digging a second well is exactly what "expanding to a tree" costs --
  the identical water, produced twice, just to give each consumer its
  own private copy.
```

### Background

Chapter 2's toy compiler was always a tree (as an AST) and then a straight-line list (as linear IR) for a real structural reason, not an arbitrary design choice: both were built by walking *parsed source text*, and a subexpression written once in source text produces exactly one AST node. Nothing in `lex()`/`Parser::parseExpr()`/`lower()` could ever make one computed result feed two different downstream operations, because there was never a way to write that sharing down in source text in the first place -- writing `a` twice in `a + a` just re-parses two separate `Variable` nodes that both happen to look the same name up in the same environment, not one shared computed value. A real tensor program does not have this restriction: a normalization's output routinely feeds both a residual-add and the next layer, and File 005 below builds exactly that shape by hand, directly through CUDA Hammer's own graph-construction API (`addInput`/`addBinary`/`addUnary`), never by parsing text.

```cpp
// Chapter 4: Representing Tensors and Operations as a Graph
// 005_graph_representation_and_sharing.cpp
//
// Section 4.1 -- why a tensor computation needs a graph, not a tree or a
// linear list -- and Section 4.2 -- Node and Value as the actual building
// blocks CUDA Hammer's IR is made of, from here through the rest of the
// book.
//
// Chapter 2's toy compiler always produced a tree (as an AST) and then a
// straight-line list (as linear IR), because both were built by walking
// PARSED TEXT: a subexpression written once in source produces exactly
// one AST node, so nothing in that pipeline could ever make one computed
// result feed two different downstream operations without writing that
// subexpression's text out twice (and recomputing it). A real tensor
// program does not have this restriction -- a value computed once is
// routinely consumed by more than one later operation (a normalization's
// output feeding both a residual-add and a following layer, for
// instance) -- so CUDA Hammer's own IR needs a data structure that lets
// one computed result be referenced by more than one consumer without
// duplicating the operation that produced it. That data structure is a
// graph: Value identifies a specific node's output, Node holds a list of
// Values as its own inputs (edges pointing backward at its producers),
// and Graph owns every Node.
//
// This file builds a small, genuinely shared DAG by hand (through
// CUDA Hammer's own graph-construction API, not by parsing text) and
// counts two real things: how many consumers each node's output
// actually has, and how many nodes an equivalent TREE (no sharing --
// every multi-consumer value inlined at every one of its use sites
// instead) would need to represent the identical computation. That
// second count is computed by a genuinely recursive, UNMEMOIZED
// expansion -- which, not coincidentally, redoes the shared node's own
// work once per consumer while computing it, the IR-size version of
// exactly the "eager execution recomputes what fusion would have
// shared" argument Chapter 1 already made about runtime memory traffic.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 005_graph_representation_and_sharing.cpp -o 005_graph_representation_and_sharing
// Run:     ./005_graph_representation_and_sharing
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>

// ============================ Value and Node ============================
//
// A Value never points at a Node directly (no pointer, no reference) --
// it is just a small, cheap-to-copy pair of integers: which node
// produced it, and which of that node's outputs. Every op in this
// chapter has exactly one output, so outputIndex is always 0 here, but
// keeping it explicit now means a future op with two real outputs (for
// instance, a max-with-argmax op producing both a value and an index)
// needs no change to Value's own shape later, only a new node kind that
// happens to report more than one output.
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
    std::vector<Value> inputs;  // edges INTO this node, i.e. this node's own producers
    float constValue = 0.0f;    // only meaningful when op == OpKind::Const
};

// Graph owns every Node through a vector<unique_ptr<Node>> -- Node
// objects are individually heap-allocated, so a Value's nodeId stays
// valid forever even if the owning vector itself reallocates (which is
// exactly why Value stores an id, resolved through Graph::node(), rather
// than a raw Node* that a vector reallocation could otherwise leave
// dangling if Node were stored by value instead of by unique_ptr).
class Graph {
public:
    Value addInput(const std::string& name) {
        return addNode(OpKind::Input, {}, name);
    }
    Value addConst(float v, const std::string& name) {
        Value out = addNode(OpKind::Const, {}, name);
        nodes_.back()->constValue = v;
        return out;
    }
    Value addUnary(OpKind op, Value in, const std::string& name) {
        return addNode(op, {in}, name);
    }
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) {
        return addNode(op, {lhs, rhs}, name);
    }

    const Node* node(int id) const { return nodes_.at(static_cast<size_t>(id)).get(); }
    size_t size() const { return nodes_.size(); }
    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }

    // Genuinely counted: how many other nodes' input lists reference
    // each node's single output, across the whole graph.
    std::map<int, int> consumerCounts() const {
        std::map<int, int> counts;
        for (const auto& n : nodes_) counts[n->id] = 0;
        for (const auto& n : nodes_) {
            for (const Value& in : n->inputs) {
                counts[in.nodeId]++;
            }
        }
        return counts;
    }

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

// Recursively expands a value into the node count an equivalent TREE
// would need -- every multi-consumer node's own subtree gets counted
// again, once per visit, exactly mirroring what inlining it at every use
// site (instead of sharing it) would actually cost. Deliberately NOT
// memoized: a memoized version would just be re-deriving the graph's own
// sharing, defeating the point of counting what NOT sharing costs.
static long long expandToTreeNodeCount(const Graph& g, Value v) {
    const Node* n = g.node(v.nodeId);
    long long count = 1;
    for (const Value& in : n->inputs) {
        count += expandToTreeNodeCount(g, in);
    }
    return count;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) {
            printf(" [value=%.1f]", n->constValue);
        }
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

int main() {
    printf("=== Section 4.1 / 4.2: a genuinely shared DAG, built through CUDA Hammer's own graph API ===\n\n");

    // Builds the diamond below directly through addInput/addBinary/addUnary
    // -- never by parsing text, so this is real graph CONSTRUCTION, not
    // Chapter 2's kind of parsing-driven tree building:
    //
    //   a, b = Input, Input
    //   t1   = Add(a, b)
    //   t2   = Mul(t1, a)      <- t1 used again here
    //   t3   = ReLU(t1)        <- and again here: t1 has TWO consumers
    //   out  = Add(t2, t3)
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");

    printf("Graph (%zu nodes):\n", g.size());
    printGraph(g);

    printf("\nConsumer counts (how many other nodes' inputs reference each node's output):\n");
    std::map<int, int> counts = g.consumerCounts();
    bool t1HasTwoConsumers = false;
    for (const auto& n : g.nodes()) {
        int c = counts[n->id];
        printf("  %%%d (%s %s): %d consumer%s\n", n->id, opKindStr(n->op).c_str(),
               n->debugName.c_str(), c, c == 1 ? "" : "s");
        if (n->id == t1.nodeId) t1HasTwoConsumers = (c == 2);
    }

    // Note that %0 (a) ALSO has 2 consumers -- and that alone would not
    // distinguish this from Chapter 2's tree, since a tree can "reuse" a
    // leaf value too (parsing "a + a" produces two separate Variable AST
    // nodes that both look `a` up, with no sharing required). What a tree
    // genuinely cannot do is share a COMPUTED, non-leaf result -- t1 is
    // an Add's own output, not a leaf, and its 2 consumers below (t2 and
    // t3) are exactly the case a tree would have to duplicate the entire
    // Add operation to represent.
    printf("\nself-check: t1 (a computed, non-leaf node) has exactly 2 consumers (%s) -- sharing\n",
           t1HasTwoConsumers ? "confirmed" : "MISMATCH");
    printf("a COMPUTED result this way, without recomputing it, is what a tree cannot do.\n");

    long long treeNodeCount = expandToTreeNodeCount(g, out);
    printf("\nReal DAG node count:                    %zu\n", g.size());
    printf("Equivalent unshared-tree node count:    %lld  (t1's own subtree -- {a, b, t1}, 3 nodes --\n",
           treeNodeCount);
    printf("                                          gets counted twice: once while expanding t2,\n");
    printf("                                          once more while expanding t3)\n");

    bool treeIsLarger = treeNodeCount > static_cast<long long>(g.size());
    printf("self-check: the unshared-tree expansion is strictly larger than the real DAG (%s)\n",
           treeIsLarger ? "confirmed" : "MISMATCH");

    return (t1HasTwoConsumers && treeIsLarger) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 005_graph_representation_and_sharing.cpp -o 005_graph_representation_and_sharing
./005_graph_representation_and_sharing
```

**Output:**

```text
=== Section 4.1 / 4.2: a genuinely shared DAG, built through CUDA Hammer's own graph API ===

Graph (6 nodes):
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1) <- %0, %1
  %3 = Mul(t2) <- %2, %0
  %4 = ReLU(t3) <- %2
  %5 = Add(out) <- %3, %4

Consumer counts (how many other nodes' inputs reference each node's output):
  %0 (Input a): 2 consumers
  %1 (Input b): 1 consumer
  %2 (Add t1): 2 consumers
  %3 (Mul t2): 1 consumer
  %4 (ReLU t3): 1 consumer
  %5 (Add out): 0 consumers

self-check: t1 (a computed, non-leaf node) has exactly 2 consumers (confirmed) -- sharing
a COMPUTED result this way, without recomputing it, is what a tree cannot do.

Real DAG node count:                    6
Equivalent unshared-tree node count:    10  (t1's own subtree -- {a, b, t1}, 3 nodes --
                                          gets counted twice: once while expanding t2,
                                          once more while expanding t3)
self-check: the unshared-tree expansion is strictly larger than the real DAG (confirmed)
```

!!! warning "[COMMON TRAP] Assuming any reused leaf name proves a graph is needed"
    `%0` (`a`) above also has 2 consumers, but that alone does not prove a tree could not represent this computation -- a tree can "reuse" a leaf just fine, since parsing `a + a` already produces two separate `Variable` AST nodes that both independently look `a` up, with no sharing required at all. What a tree genuinely cannot do is share a *computed, non-leaf* result: `t1` is an `Add` node's own output, and its two consumers (`t2` and `t3`) are exactly the case that would force a tree to duplicate the entire `Add` operation -- not just a name lookup -- to represent the same computation. The defining test for "does this need a graph" is whether a *non-leaf* node has more than one consumer, not whether any name appears more than once.

## 4.2 Nodes and Values: The Building Blocks

### Intuition

A library's own catalog does not hand a reader the book itself -- it hands them a call slip recording where the book lives: which shelf, which copy. Two different readers can hold identical call slips pointing at the exact same shelf entry, and neither one owns a private copy; neither slip can ever go stale from the book being moved, either, because the library's own shelving (not the slip) is what actually holds the book. CUDA Hammer's `Value` is that call slip, and `Graph` is the library's own shelves.

```text
A Value is a call slip, not the book itself:
  Value { nodeId: 7, outputIndex: 0 }    -- "shelf 7, copy 0"

Two DIFFERENT consumers can hold IDENTICAL call slips pointing at the
SAME shelf entry -- neither one owns a private copy, and neither slip
can ever go stale, because a slip only ever records WHERE the real
Node lives (by id, resolved through Graph::node()), never a direct
grip on the Node object itself.
```

### Background

File 005 above already contains this section's own code -- `Value{nodeId, outputIndex}`, `Node` (an op kind, a debug name, and a list of input `Value`s), and `Graph` (owning every `Node` through `vector<unique_ptr<Node>>`) are exactly the three types this section is about, just read now for their own design rather than for the sharing they enable. Two design choices are worth being explicit about. First, `outputIndex` is always `0` for every op this chapter defines, and yet it is still a real field on `Value` rather than being left out -- because a future op with genuinely more than one output (a max-with-argmax op producing both a value and an index, to name one CUDA Hammer will eventually need) needs no change to `Value`'s own shape when that day comes, only a new node kind that happens to report more than one output. Second, `Value` stores an integer `nodeId`, resolved through `Graph::node()`, rather than a raw `Node*` -- both would actually be safe here, since `Node` objects are individually heap-allocated through `unique_ptr` and so never move even if the owning `vector` itself reallocates, but an id is the choice CUDA Hammer keeps for the rest of the book: ids serialize trivially (Chapter 7's printer needs exactly this), and an id can be looked up and found *absent* cleanly, where a stale pointer cannot safely be checked for validity at all.

!!! warning "[COMMON TRAP] Treating outputIndex as dead weight because every op here only has one output"
    It is tempting to simplify `Value` down to just `nodeId` once, since nothing in this chapter's four op kinds (`Input`, `Const`, `Add`, `Mul`, `ReLU`) ever produces more than one output. Doing that would work for this chapter and then force a breaking change to every single call site across the whole rest of the book the first time a real multi-output op shows up -- exactly the kind of "premature simplification" a real IR design avoids by keeping the field now, at zero real cost, while it is trivial to add.

## 4.3 A Graph Must Be Acyclic (And How CUDA Hammer's API Guarantees It)

### Intuition

A course catalog's prerequisite chart is only usable if some valid enrollment order actually exists: Calculus 1 before Calculus 2, Calculus 2 before Linear Algebra, and so on. A broken catalog that required Course A as a prerequisite for Course B, and Course B as a prerequisite for Course A, would leave no student able to take either course first -- there is no valid starting point, because everything that could come first already requires something else to come first. A computation graph has exactly the same requirement: every node's inputs have to be genuinely computable before that node runs, which means the graph has to be acyclic, or there is no order in which CUDA Hammer could ever actually execute it.

```text
COURSE PREREQUISITES (a valid DAG -- a valid enrollment order always
exists, because nothing requires itself, directly or indirectly):
  Calc 1 --> Calc 2 --> Linear Algebra

A BROKEN CATALOG (a cycle -- no student could ever start):
  Course A --> Course B --> Course A
  A requires B, B requires A: there is no first course to take, which
  is exactly what Kahn's algorithm below detects by getting stuck with
  nodes left over that never reach in-degree zero.
```

### Background

CUDA Hammer's own `addInput`/`addConst`/`addUnary`/`addBinary` API can only ever take `Value`s that some *earlier* call already returned -- there is no way to reference a node that does not exist yet -- so a graph built only through that API can never contain a cycle, by construction, the same way Chapter 2's parser could never produce an AST node with itself as its own child. File 006 below proves the resulting acyclicity two ways. First, it runs a genuine topological sort (Kahn's algorithm: repeatedly remove a node with no unprocessed producers left, decrementing every consumer's own remaining-producer count) on File 005's own diamond graph, and cross-checks that order with a second, independently implemented method -- a positional check confirming every node really does appear after all of its own producers -- the same independent-cross-check discipline Chapter 2 used for its AST evaluator. Second, it reaches *past* the normal API -- directly mutating a `Node`'s own `inputs` vector after construction, something ordinary use of `Graph` never does -- to hand-build one small graph that genuinely contains a cycle, and confirms the same checker correctly reports it as invalid rather than silently producing a wrong order.

```cpp
// Chapter 4: Representing Tensors and Operations as a Graph
// 006_topological_order_and_cycle_detection.cpp
//
// Section 4.3 -- a graph must be acyclic, and how CUDA Hammer's own
// construction API guarantees that by design.
//
// Reuses File 005's own Value/Node/Graph classes unchanged (a real
// compiler's later stages never re-derive the earlier ones). The only
// new code here is a genuine topological sort (Kahn's algorithm) and an
// independent second check of its own output.
//
// CUDA Hammer's addInput/addConst/addUnary/addBinary API can only ever
// take Values that some EARLIER call already returned -- there is no way
// to reference a node that does not exist yet -- so a graph built
// through that API can never contain a cycle. This file proves that
// property two ways: first, by running Kahn's algorithm on a normally
// built graph and cross-checking its result with an independent,
// differently-implemented positional check; second, by deliberately
// reaching PAST the normal API (directly mutating a Node's own inputs
// vector, something ordinary use of Graph never does) to hand-construct
// one small graph that DOES contain a cycle, and confirming the same
// checker correctly reports it as invalid rather than silently
// producing a wrong order.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 006_topological_order_and_cycle_detection.cpp -o 006_topological_order_and_cycle_detection
// Run:     ./006_topological_order_and_cycle_detection
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>

// ==================== Value / Node / Graph (from File 005) ====================

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
    Node* mutableNode(int id) { return nodes_.at(static_cast<size_t>(id)).get(); }
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

// ======================= Kahn's algorithm (new in this file) =======================
//
// Repeatedly removes a node with in-degree 0 (no unprocessed producers
// left), appending it to the order and decrementing the in-degree of
// every node that reads it. If every node gets removed this way, the
// graph is acyclic and `order` is a valid topological order. If nodes
// remain stuck with in-degree > 0 once no more zero-in-degree nodes are
// left to remove, the graph contains a cycle -- ok is set to false.
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
        // Decrement the in-degree of every node that lists `id` as an input.
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

// Independent cross-check: for a claimed order to be valid, every node
// must appear AFTER every one of its own inputs' producer nodes. This is
// checked by position lookup, never by re-running Kahn's own in-degree
// bookkeeping -- a genuinely different method arriving at the same
// verdict, the same cross-check discipline Chapter 2 used (an
// independent AST evaluator, checked against the IR interpreter).
static bool independentlyValidatesOrder(const Graph& g, const std::vector<int>& order) {
    if (order.size() != g.size()) return false;
    std::map<int, int> position;
    for (size_t i = 0; i < order.size(); ++i) position[order[i]] = static_cast<int>(i);
    for (const auto& n : g.nodes()) {
        for (const Value& in : n->inputs) {
            if (position.at(in.nodeId) >= position.at(n->id)) return false;
        }
    }
    return true;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("    %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
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

int main() {
    printf("=== Section 4.3: topological order, and detecting a graph that has no valid one ===\n\n");

    // --- Part A: the normal diamond DAG from File 005, built ONLY through
    // the graph API -- this can never contain a cycle by construction. ---
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    g.addBinary(OpKind::Add, t2, t3, "out");

    TopoResult result = topologicalSort(g);
    printf("Part A -- valid DAG (%zu nodes, built only through the graph API):\n", g.size());
    printGraph(g);
    printf("  Kahn's algorithm order: ");
    for (int id : result.order) printf("%%%d ", id);
    printf("\n  Kahn's algorithm reports valid: %s\n", result.ok ? "yes" : "no");

    bool independentCheckA = independentlyValidatesOrder(g, result.order);
    printf("  independent positional cross-check agrees: %s\n\n",
           independentCheckA ? "confirmed" : "MISMATCH");

    // --- Part B: a SECOND, tiny graph, deliberately broken by reaching
    // past the normal API to hand-construct a cycle. This is never
    // something ordinary use of Graph can produce -- addBinary/addUnary
    // only ever accept Values that an earlier call already returned, so
    // a fresh Node cannot reference a Node that does not exist yet. The
    // only way to build a cycle at all is to add the nodes first and then
    // go back and mutate an EARLIER node's own inputs afterward, which is
    // exactly what happens below, purely to test that the checker itself
    // correctly rejects it. ---
    Graph broken;
    Value x = broken.addInput("x");             // node 0, no inputs yet
    Value y = broken.addUnary(OpKind::ReLU, x, "y"); // node 1, input: node 0
    // Reach past the API: force node 0 ("x") to also depend on node 1
    // ("y") after the fact -- x -> y -> x, a genuine cycle no normal
    // sequence of addInput/addUnary/addBinary calls could ever create.
    broken.mutableNode(x.nodeId)->inputs.push_back(y);

    TopoResult brokenResult = topologicalSort(broken);
    printf("Part B -- deliberately hand-broken graph (2 nodes, x -> y -> x cycle\n");
    printf("           forced by mutating node 0's inputs directly, bypassing the graph API):\n");
    printGraph(broken);
    printf("  Kahn's algorithm processed %zu of %zu nodes before getting stuck\n",
           brokenResult.order.size(), broken.size());
    printf("  Kahn's algorithm reports valid: %s\n", brokenResult.ok ? "yes" : "no");

    bool correctlyRejected = !brokenResult.ok;
    printf("\nself-check: the normal DAG is confirmed valid AND cross-checked (%s);\n",
           (result.ok && independentCheckA) ? "confirmed" : "MISMATCH");
    printf("the hand-broken cyclic graph is correctly REJECTED, not silently misordered (%s)\n",
           correctlyRejected ? "confirmed" : "MISMATCH");

    return (result.ok && independentCheckA && correctlyRejected) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 006_topological_order_and_cycle_detection.cpp -o 006_topological_order_and_cycle_detection
./006_topological_order_and_cycle_detection
```

**Output:**

```text
=== Section 4.3: topological order, and detecting a graph that has no valid one ===

Part A -- valid DAG (6 nodes, built only through the graph API):
    %0 = Input(a)
    %1 = Input(b)
    %2 = Add(t1) <- %0, %1
    %3 = Mul(t2) <- %2, %0
    %4 = ReLU(t3) <- %2
    %5 = Add(out) <- %3, %4
  Kahn's algorithm order: %0 %1 %2 %3 %4 %5 
  Kahn's algorithm reports valid: yes
  independent positional cross-check agrees: confirmed

Part B -- deliberately hand-broken graph (2 nodes, x -> y -> x cycle
           forced by mutating node 0's inputs directly, bypassing the graph API):
    %0 = Input(x) <- %1
    %1 = ReLU(y) <- %0
  Kahn's algorithm processed 0 of 2 nodes before getting stuck
  Kahn's algorithm reports valid: no

self-check: the normal DAG is confirmed valid AND cross-checked (confirmed);
the hand-broken cyclic graph is correctly REJECTED, not silently misordered (confirmed)
```

!!! warning "[COMMON TRAP] Assuming the graph API needs its own separate cycle check"
    It's tempting to think `Graph` itself needs to actively guard against cycles -- some kind of check inside `addBinary`/`addUnary` refusing a bad call. It does not, and File 006's Part B shows exactly why: every normal call to that API can only take `Value`s some *earlier* call already returned, so there is no sequence of ordinary calls that could ever construct a cycle in the first place -- the guarantee comes from the API's own shape, not from a runtime check bolted onto it. Part B's cycle only exists because the code deliberately reached past the API (`mutableNode()`, a method File 005's own `Graph` does not even expose) specifically to build an invalid test case. A pass written later in this book that only ever calls the public `addBinary`/`addUnary`/`addUnary` API can trust the graph it builds is acyclic, for free.

## Chapter Summary

CUDA Hammer's own IR is a graph, not a tree or a straight-line list, because a real tensor computation needs one computed value to be consumed by more than one later operation without recomputing it -- something Chapter 2's toy compiler structurally could never need, since it was always built by walking parsed text where each subexpression already appears exactly once. `Value` (a `{nodeId, outputIndex}` pair, never a raw pointer) identifies a specific node's output the way a library call slip identifies a shelf entry rather than gripping the book itself; `Node` holds an op kind, a debug name, and a list of input `Value`s as its own edges; `Graph` owns every `Node`. File 005 built a real six-node diamond DAG through that API and genuinely counted two things: that a computed, non-leaf node (`t1`) can have more than one consumer, and that flattening this same six-node graph into an equivalent tree (no sharing at all) would cost 10 nodes instead of 6 -- a real, counted representation-size cost of not sharing, the IR-size sibling of Chapter 1's own runtime memory-traffic argument. File 006 proved the graph this API builds is always acyclic two ways: a genuine topological sort (Kahn's algorithm), cross-checked by an independent positional validator, succeeds on a normally-built graph; the same checker correctly detects a cycle in a graph deliberately hand-broken by reaching past the normal API, rather than silently producing a wrong order. Chapter 5 builds a real frontend on top of this exact `Graph`/`Node`/`Value` foundation -- a way to construct these graphs from something closer to how a real user would describe a tensor computation, rather than the direct API calls this chapter used by hand.

## Self-Check Questions

1. Why could Chapter 2's `lower()` never produce an IR temp with more than one consumer, structurally, regardless of what expression was parsed?
2. File 005's `%0` (`a`) has 2 consumers, same as `t1`. Why does the chapter's own [COMMON TRAP] say this does not, by itself, prove a graph is needed?
3. What does `Value` actually store, and why does `Graph` resolve a `Value` to a real `Node` through an id lookup (`Graph::node()`) rather than storing a `Node*` directly inside `Value`?
4. `outputIndex` is always `0` in this chapter's own code. What is the concrete argument for keeping it as a real field anyway, rather than removing it until it's needed?
5. File 005's `expandToTreeNodeCount()` is deliberately not memoized. What would memoizing it actually defeat the purpose of measuring?
6. Explain, in your own words, why CUDA Hammer's `addInput`/`addConst`/`addUnary`/`addBinary` API can never construct a cycle through ordinary use, without needing a runtime check inside those functions.
7. File 006's Part B needed a method (`mutableNode()`) that File 005's own `Graph` does not expose in its own public interface. Why was that necessary to build a cyclic test case at all?
8. If `topologicalSort()`'s own in-degree bookkeeping had a bug that caused it to report `ok = true` on a graph that actually had a cycle, would File 006's `independentlyValidatesOrder()` check catch that bug? Why or why not?

## Where We Go Next

This chapter designed and tested the `Value`/`Node`/`Graph` data structures every later chapter in this book builds on, and proved the one structural guarantee (acyclicity) everything downstream gets to assume for free. Chapter 5 builds CUDA Hammer's own frontend: a way to construct these same graphs that looks less like the direct `addBinary`/`addUnary` calls this chapter used by hand, and more like how a real user actually wants to describe a tensor computation.

## Worked Solutions

1. `lower()` walks a parsed AST, and an AST node is only ever reachable from the one parent that owns it (`std::unique_ptr<Expr> left, right` -- exclusive ownership, not shared). Every IR instruction `lower()` emits corresponds to exactly one AST node, so an IR temp can only ever be read by the one instruction lowered from that AST node's one parent -- there is structurally no second parent that could reference the same temp a second time, because the AST itself never allows a node to have two parents.
2. Because a tree can already "reuse" a leaf without any sharing infrastructure at all -- parsing `a + a` produces two separate `Variable` AST nodes that both independently look the name `a` up in the environment, with no shared node and no edge-sharing required. What a tree genuinely cannot represent is a *computed, non-leaf* result (like `t1`, an `Add` node's own output) being read by two different consumers without literally duplicating the `Add` operation itself -- that specific case, not mere name reuse, is the real test for whether a graph is needed.
3. `Value` stores `{nodeId, outputIndex}` -- two plain integers, not a pointer. `Graph` resolves it through `Graph::node()` (an id lookup into its own `vector<unique_ptr<Node>>`) rather than storing a `Node*` directly because an id-based lookup can be checked for validity cleanly (does this id exist in the graph, yes or no) and serializes trivially as plain data (exactly what Chapter 7's printer needs) -- a stale or dangling pointer offers neither property, even though in this specific implementation a `Node*` would technically stay valid too, since `Node` objects are individually heap-allocated through `unique_ptr` and so never move when the owning `vector` reallocates.
4. Every op this chapter defines happens to have exactly one output, but a real op with two genuine outputs (a max-with-argmax op producing both a value and an index, for instance) is a real, foreseeable future need. Keeping `outputIndex` as a real field now, at zero cost since it is just always `0`, means that future op needs no change to `Value`'s own shape and no update to every call site across the rest of the book that already constructs a `Value` -- only a new node kind that happens to report more than one output.
5. Memoizing `expandToTreeNodeCount()` would make it stop counting what an actual unshared tree would cost, and start silently re-deriving the graph's own sharing instead (a memoized call for `t1` would compute its subtree size once and reuse that cached answer for both `t2` and `t3` -- which is exactly what the real DAG already does, not what a tree would have to do). The whole point of the function is to answer "what would this cost WITHOUT the graph's sharing," so it has to actually re-walk `t1`'s subtree in full every time it is reached, once per consumer, the same redundant work a real unshared tree -- or File 1's own EAGER execution mode -- would actually perform.
6. Every call to `addUnary`/`addBinary` takes one or more `Value`s as arguments, and the only `Value`s that exist to pass in are ones some *earlier* call to `addInput`/`addConst`/`addUnary`/`addBinary` already returned. A brand-new node being constructed right now cannot possibly receive, as one of its own inputs, a `Value` pointing at itself or at any node that will only be created later -- that `Value` simply does not exist yet to be passed in. The impossibility is structural, built into what arguments the API's own functions can even accept, not something any function body needs to actively check for.
7. Because the public API (`addInput`/`addConst`/`addUnary`/`addBinary`) can only ever construct edges pointing backward at already-existing nodes, exactly the property that makes cycles impossible through ordinary use (per the answer to Question 6). The only way to build a genuine cycle at all is to modify an *already-existing* node's own `inputs` list after the fact, reaching past what the public API allows -- which is precisely why File 005's own `Graph` does not expose a way to do this, and File 006 has to add `mutableNode()` specifically, and only, to construct this one deliberately invalid test case.
8. Yes. `independentlyValidatesOrder()` never looks at `topologicalSort()`'s own in-degree counters at all -- it takes the claimed order as a plain list of ids and checks, by position lookup alone, whether every node in that list genuinely appears after all of its own inputs' producer nodes. A bug in the in-degree bookkeeping that caused `topologicalSort()` to report `ok = true` on a genuinely cyclic graph would still have to produce *some* concrete order, and that order would necessarily place at least one node before one of its own producers (since a true topological order cannot exist for a cyclic graph) -- which is exactly the condition `independentlyValidatesOrder()` checks for, using a completely different method than the one that produced the (buggy) order in the first place.

---

**Sources cited in this chapter:** none -- CUDA Hammer's own `Value`/`Node`/`Graph` design is original to this book, not modeled on any specific external project's own source code (Chapter 3 already surveyed, and cited, how three real production compilers represent a comparable graph-shaped IR).
