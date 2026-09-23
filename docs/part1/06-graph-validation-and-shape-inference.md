# 6. Graph Validation and Shape Inference

**What you will understand:** given a `Graph` of `Input`/`Const`/`Add`/`Mul`/`ReLU` nodes -- built by hand in Chapter 4, or parsed from text in Chapter 5 -- how to determine whether it is actually a *valid* tensor computation, and how to compute the shape of every node's own output without a user ever stating it for a computed value. This is the last piece Part 1 needs: Part 2's optimization passes all assume they are working on a graph that has already been checked this way.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph` and its topological sort (`topologicalSort()`, reused completely unchanged in this chapter), and Chapter 2's vocabulary for what a "pass" is.

---

Neither Chapter 4 nor Chapter 5 ever gave a `Node` a size. A graph built by hand through `addBinary`, or parsed from a line of text like `t2 = mul(t1, a)`, is entirely a statement about *which values depend on which other values* -- it says nothing about how many numbers any of those values actually holds. That is deliberate: shape is not a structural property of the graph the way an edge is, it is something that has to be *computed*, separately, and this chapter builds exactly that computation as its own pass -- reading an already-built graph, producing a result, and never modifying `Node`'s own definition to do it.

```text
WHAT CHAPTERS 4 AND 5 BUILT (edges only -- no notion of size anywhere):

  Graph:  %0 = Input(a)        %2 = Add(t1), inputs %0, %1
          %1 = Input(b)        %3 = Mul(t2), inputs %2, %0
                                ...

WHAT THIS CHAPTER ADDS (a SEPARATE side table, keyed by node id):

  Shapes: %0 -> [3, 4]          %2 -> [3, 4]   (computed, not stated)
          %1 -> [4]             %3 -> [3, 4]   (computed, not stated)

  the Graph above is not touched or modified at all -- Shapes is a new,
  independent map that a PASS (this chapter's own inferShapes()) builds
  by reading the Graph, the same relationship Chapter 2's own passes had
  to the IR they read.
```

## 6.1 What "Valid" Means for a Tensor Graph

### Intuition

Two spreadsheets can only be added cell by cell if their columns line up -- row 3, column B in the first sheet has to correspond to *some* specific cell in the second sheet for "add these two sheets together" to mean anything at all. If both sheets are exactly the same size, the correspondence is obvious: cell for cell, in place. Real tensor programs need something more flexible than "exactly the same size, always" -- a single per-channel bias value, shaped like one row, routinely gets added to every row of a much larger table, and forcing the programmer to first manually copy that one row into a full-sized table just to make the shapes match would be pure busywork. Broadcasting is the rule that makes "obviously supposed to line up, even though the sizes aren't identical" precise instead of vague.

Picture the one-row bias case concretely: a table of monthly sales with 12 rows (one per month) and 4 columns (one per product), and a single 4-column row of per-product adjustments to add to *every* month. Nothing needs to be copied 12 times -- the single row is understood to apply to each of the 12 rows in turn, because it lines up with them column for column.

```text
A 12-ROW TABLE, PLUS A SINGLE 4-COLUMN ROW APPLIED TO EVERY ROW OF IT:

  sales, shape [12, 4]:          adjustment, shape [4]:
    row 1:  [ 10, 20, 30, 40 ]     [ +1, +2, +3, +4 ]
    row 2:  [ 11, 19, 29, 41 ]     (this ONE row lines up with EVERY
    ...                            row of sales above, column for
    row 12: [  9, 21, 28, 39 ]     column -- it is never copied 12
                                    times to make the shapes "match")

  result, shape [12, 4]:
    row 1:  [ 11, 22, 33, 44 ]
    row 2:  [ 12, 21, 32, 45 ]
    ...
```

Chapter 1's own EAGER-vs-FUSED chains never needed this idea at all -- every array in that chapter was already the same size as every other array it combined with, by construction. A real tensor graph cannot assume that; broadcasting is the precise rule that decides which *different* sizes are still allowed to combine, and what size the result has when they do.

### Background

Two shapes are combined by NumPy, PyTorch, and TensorFlow all using the exact same rule, and it is worth stating precisely rather than gesturing at loosely, since File 009's own `broadcastShapes()` implements it exactly. NumPy's own documentation states the comparison direction plainly: "When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimension and works its way left." -- meaning two shapes are lined up starting from their *last* dimension, not their first, and a shape with fewer dimensions than the other is treated as if it had extra leading dimensions of size `1` to pad it out to the same length. Once two shapes are lined up this way, NumPy's own documentation states the compatibility test just as plainly: "Two dimensions are compatible when 1. they are equal, or 2. one of them is 1." When one of an aligned pair is `1`, the result takes the *other* one's size -- the size-1 dimension is the one that conceptually "repeats" to match its partner, the way the single adjustment row above conceptually repeats against every one of the sales table's 12 rows.

Walking `[3, 4]` and `[4]` through this rule step by step makes the trailing-alignment idea concrete:

```text
ALIGNING [3, 4] AGAINST [4], FROM THE TRAILING DIMENSION LEFTWARD:

  shape A:     3    4
  shape B:  (pad)   4
             ^^^^
             B has no second-from-right dimension at all, so it is
             treated as 1 here -- NOT an error, just "not specified."

  position 0 (trailing):  A=4, B=4  -->  equal            -->  result: 4
  position 1 (next left): A=3, B=1  -->  B is 1            -->  result: 3
                                          (A's size, 3, wins)

  final result shape:  [3, 4]
```

And the reverse case -- `[3, 1]` against `[1, 4]`, where *both* shapes contribute a real, non-1 dimension at different positions -- shows broadcasting is symmetric, not something only the "smaller" shape does:

```text
ALIGNING [3, 1] AGAINST [1, 4] -- BOTH SIDES BROADCAST, AT DIFFERENT POSITIONS:

  position 0 (trailing):  A=1, B=4  -->  A is 1  -->  result: 4  (B's size wins here)
  position 1 (next left): A=3, B=1  -->  B is 1  -->  result: 3  (A's size wins here)

  final result shape:  [3, 4]

  neither shape is "the small one" -- each contributes its own real
  dimension at the position where the OTHER shape only has a 1.
```

File 009 below tests `broadcastShapes()` against five such cases, including both of the ones traced above, checking every result against a shape this book computed by hand -- not merely that the function runs, but that it computes the exact right answer.

```cpp
// Chapter 6: Graph Validation and Shape Inference
// 009_shape_inference_by_topological_walk.cpp
//
// Section 6.1 -- what "valid" means for a tensor graph (the broadcasting
// rule two operand shapes must satisfy) -- and Section 6.2 -- inferring
// every node's own output shape by walking the graph in the exact
// topological order Chapter 4's File 006 already built.
//
// A Node in this book's Graph has never stored a shape -- Chapter 4 and
// 5 both built and parsed graphs entirely in terms of ids and edges,
// with no notion of "how big" any value actually is. Shape inference is
// a separate PASS (Chapter 2's own vocabulary: something that reads an
// IR and computes a result, without ever touching source text again)
// that walks an already-built Graph and produces a SIDE TABLE --
// map<nodeId, Shape> -- rather than modifying Node itself. Keeping shape
// data external like this is deliberate: Chapter 8's pass manager will
// run many different analyses over the same Graph, and none of them
// need to change Node's own definition to do it.
//
// Reuses Chapter 4's own Value/Node/Graph AND its topologicalSort()
// (Kahn's algorithm) completely unchanged -- shape inference's only new
// idea is WHAT to compute at each node once the order to compute it in
// is already known.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 009_shape_inference_by_topological_walk.cpp -o 009_shape_inference_by_topological_walk
// Run:     ./009_shape_inference_by_topological_walk
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
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

// ==================== Topological sort (from Chapter 4's File 006, unchanged) ====================

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

// ============================== Shape (new) ==============================

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

static bool shapesEqual(const Shape& a, const Shape& b) { return a.dims == b.dims; }

// ======================= Section 6.1: the broadcasting rule =======================
//
// The same elementwise broadcasting rule NumPy, PyTorch, and TensorFlow
// all use: compare two shapes dimension by dimension, starting from the
// TRAILING (rightmost) dimension and working left; a dimension missing
// on the shorter shape (because it has fewer dimensions) is treated as
// size 1; two aligned dimensions are compatible when they are EQUAL, or
// when AT LEAST ONE of them is 1 -- in which case the result takes
// whichever size is not 1. If any aligned pair is neither equal nor has
// a 1, the two shapes cannot be broadcast together at all.
static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        // Index from the trailing end on each side independently.
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
                                      " are not broadcast-compatible (" + std::to_string(da) + " vs " +
                                      std::to_string(db) + " at trailing position " + std::to_string(i) + ")");
        }
        result[rank - 1 - i] = outDim;
    }
    return Shape{result};
}

// ======================= Section 6.2: shape inference over a graph =======================
//
// declaredShapes supplies the shape for every Input/Const node -- there
// is nothing to INFER for a leaf, its shape is given, not computed.
// Every other node's shape is computed from its own already-known
// inputs' shapes, in the exact order topologicalSort() already proved
// is valid: by the time this loop reaches any node, every one of its
// inputs has ALREADY been visited (that is precisely what a topological
// order guarantees), so `shapes.at(...)` below can never fail to find
// an input's shape it needs.
static std::map<int, Shape> inferShapes(const Graph& g, const std::map<int, Shape>& declaredShapes) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("inferShapes: graph is not acyclic");

    std::map<int, Shape> shapes;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        std::string context = "node %" + std::to_string(id) + " (" + n->debugName + ")";

        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            auto it = declaredShapes.find(id);
            if (it == declaredShapes.end()) {
                throw std::runtime_error(context + ": no declared shape was provided for this leaf node");
            }
            shapes[id] = it->second;
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            const Shape& lhs = shapes.at(n->inputs[0].nodeId);
            const Shape& rhs = shapes.at(n->inputs[1].nodeId);
            shapes[id] = broadcastShapes(lhs, rhs, context);
        } else { // ReLU -- shape-preserving: output shape equals input shape exactly.
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

int main() {
    printf("=== Section 6.1: the broadcasting rule, tested on its own ===\n\n");

    struct BroadcastCase { Shape a, b, expected; };
    std::vector<BroadcastCase> cases = {
        {{{3, 4}},    {{3, 4}},    {{3, 4}}},   // identical shapes
        {{{3, 4}},    {{4}},       {{3, 4}}},   // trailing match, leading dim padded with 1
        {{{3, 1}},    {{1, 4}},    {{3, 4}}},   // mutual broadcasting, both directions
        {{{5}},       {{1}},       {{5}}},      // scalar-like broadcasting
        {{{2, 3, 4}}, {{4}},       {{2, 3, 4}}} // higher rank + trailing match
    };

    bool allBroadcastCasesMatch = true;
    for (const auto& c : cases) {
        Shape result = broadcastShapes(c.a, c.b, "test case");
        bool matches = shapesEqual(result, c.expected);
        allBroadcastCasesMatch = allBroadcastCasesMatch && matches;
        printf("  %s broadcast %s -> %s  (expected %s, %s)\n", shapeStr(c.a).c_str(), shapeStr(c.b).c_str(),
               shapeStr(result).c_str(), shapeStr(c.expected).c_str(), matches ? "confirmed" : "MISMATCH");
    }
    printf("\nself-check: every broadcasting case matches its hand-computed expected shape (%s)\n\n",
           allBroadcastCasesMatch ? "confirmed" : "MISMATCH");

    printf("=== Section 6.2: inferring every node's shape over Chapter 4's diamond graph ===\n\n");

    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");

    // Only the two leaves need a DECLARED shape -- "a" is a full [3, 4]
    // tensor, "b" is a [4]-shaped tensor that broadcasts against it,
    // deliberately exercising broadcasting on the very first Add.
    std::map<int, Shape> declared = {
        {a.nodeId, Shape{{3, 4}}},
        {b.nodeId, Shape{{4}}},
    };

    std::map<int, Shape> inferred = inferShapes(g, declared);

    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s): shape %s\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str(),
               shapeStr(inferred.at(n->id)).c_str());
    }

    bool outShapeCorrect = shapesEqual(inferred.at(out.nodeId), Shape{{3, 4}});
    printf("\nself-check: 'out' -- reached through two broadcasts (t1) and one shape-preserving\n");
    printf("ReLU (t3) -- correctly infers to [3, 4] (%s)\n", outShapeCorrect ? "confirmed" : "MISMATCH");

    return (allBroadcastCasesMatch && outShapeCorrect) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 009_shape_inference_by_topological_walk.cpp -o 009_shape_inference_by_topological_walk
./009_shape_inference_by_topological_walk
```

**Output:**

```text
=== Section 6.1: the broadcasting rule, tested on its own ===

  [3, 4] broadcast [3, 4] -> [3, 4]  (expected [3, 4], confirmed)
  [3, 4] broadcast [4] -> [3, 4]  (expected [3, 4], confirmed)
  [3, 1] broadcast [1, 4] -> [3, 4]  (expected [3, 4], confirmed)
  [5] broadcast [1] -> [5]  (expected [5], confirmed)
  [2, 3, 4] broadcast [4] -> [2, 3, 4]  (expected [2, 3, 4], confirmed)

self-check: every broadcasting case matches its hand-computed expected shape (confirmed)

=== Section 6.2: inferring every node's shape over Chapter 4's diamond graph ===

  %0 = Input(a): shape [3, 4]
  %1 = Input(b): shape [4]
  %2 = Add(t1): shape [3, 4]
  %3 = Mul(t2): shape [3, 4]
  %4 = ReLU(t3): shape [3, 4]
  %5 = Add(out): shape [3, 4]

self-check: 'out' -- reached through two broadcasts (t1) and one shape-preserving
ReLU (t3) -- correctly infers to [3, 4] (confirmed)
```

!!! warning "[COMMON TRAP] Assuming one matching dimension is enough"
    `[3, 4]` and `[3, 5]` share a dimension (both have `3` in the leading position) -- it's tempting to think that overlap counts for something. It does not: broadcasting checks *every aligned position independently*, and at the trailing position here, `4` and `5` are neither equal nor is either one `1`, so the whole pair is incompatible regardless of what happens to line up elsewhere. File 010's own Section 6.3 tests exactly this case, deliberately, to make sure `broadcastShapes()` rejects it rather than being satisfied by a partial match.

## 6.2 Inferring Shapes by Walking the Graph in Topological Order

### Intuition

A relay race only works because each runner already has the baton in hand before they need to start running -- nobody is ever asked to run *before* the previous leg finishes. Shape inference has exactly this same dependency: a node's own output shape can only be computed once every one of its inputs' shapes is already known, which means visiting nodes in *some* order where a node's producers always come before it. Chapter 4 already built and proved exactly this ordering -- `topologicalSort()` -- for an entirely different reason (proving a graph has a valid execution order at all), and shape inference reuses that same function, unchanged, simply because the ordering guarantee it provides is exactly the one this new problem also needs.

```text
THE DIAMOND GRAPH, SHAPES FILLING IN AS THE TOPOLOGICAL WALK PROCEEDS:

  step 1: visit %0 (a)   -- a LEAF: shape is DECLARED, not computed: [3, 4]
  step 2: visit %1 (b)   -- a LEAF: shape is DECLARED, not computed: [4]
  step 3: visit %2 (t1)  -- Add(%0, %1): broadcast([3,4], [4]) = [3, 4]
                             (both %0 and %1 ALREADY have a shape --
                              they were visited in steps 1 and 2)
  step 4: visit %3 (t2)  -- Mul(%2, %0): broadcast([3,4], [3,4]) = [3, 4]
  step 5: visit %4 (t3)  -- ReLU(%2): shape-preserving = [3, 4]
  step 6: visit %5 (out) -- Add(%3, %4): broadcast([3,4], [3,4]) = [3, 4]

  at EVERY step above, every input this node needs was already computed
  in some earlier step -- never later, never "not yet" -- which is
  exactly what a topological order guarantees, for any graph it succeeds
  on at all.
```

It is worth being concrete about what would go wrong without that ordering guarantee. Imagine visiting nodes in raw id order were somehow *not* a valid topological order for some other graph -- say a node whose own producer happened to have a higher id (impossible through this book's own `addBinary`/`addUnary` API, by Chapter 4's own cycle-freedom argument, but instructive to picture anyway):

```text
WHAT WOULD BREAK WITHOUT A TOPOLOGICAL ORDER (a hypothetical, not something
this book's own API can actually construct):

  visit %5 first, whose input is %2  --  shapes.at(2) -- FAILS: %2 has
                                          no entry in `shapes` yet, because
                                          it has not been visited.

  this is exactly why inferShapes() calls topologicalSort() FIRST and
  walks its `order`, never `g.nodes()` in raw id order -- the two happen
  to coincide for THIS chapter's own diamond graph, but inferShapes()
  never relies on that coincidence.
```

### Background

File 009's own `inferShapes()` is short specifically because it delegates the hard part -- finding a valid order -- entirely to Chapter 4's `topologicalSort()`, and only has to decide *what to compute* once that order is in hand: a leaf (`Input`/`Const`) takes its shape from the caller-supplied `declaredShapes` map, since there is nothing about a leaf's own graph structure that could tell you how big it is; `Add`/`Mul` broadcast their two already-known input shapes together via Section 6.1's own rule; `ReLU` is shape-preserving, since an elementwise unary operation like this book's own `ReLU` (a per-element `max(0, x)`) never changes how many elements exist, only their individual values. The result -- `map<int, Shape>` -- is a brand-new structure this function builds and returns; the `Graph` it walked is never modified, exactly matching this chapter's opening point that shape data lives in a separate table, not inside `Node` itself.

!!! warning "[COMMON TRAP] Assuming shape inference could just walk `g.nodes()` in id order instead"
    For this chapter's own diamond graph, node ids happen to already be in a valid topological order (Chapter 4's own `addBinary`/`addUnary` API can only reference an already-existing node, so a node's own id is always assigned after every one of its producers' ids) -- so walking `g.nodes()` directly, in raw id order, would happen to give the same answer here. `inferShapes()` deliberately does not rely on that coincidence: it calls `topologicalSort()` and walks *its* `order` instead, because a future graph-construction API (a pass that rewrites or reorders nodes, for instance, which Part 2 starts building) is not guaranteed to keep id order and topological order in sync the way this chapter's simple, append-only construction API always does.

## 6.3 Catching a Real Shape Mismatch Before It Propagates

### Intuition

A quality-control checkpoint partway down an assembly line exists so a defective part gets pulled *there*, not three stations later when it's already been bolted into something else and the defect is much harder to trace back to its actual source. Shape inference plays exactly this role for a tensor graph: an incompatible pair of shapes is caught the moment `inferShapes()` reaches the node that combines them, with a specific, named error identifying that exact node and those exact shapes -- never silently producing a wrong or nonsensical shape that some much later pass, or worse, generated code, would have to somehow puzzle out was wrong in the first place.

```text
FOUR THINGS inferShapes() CAN ENCOUNTER, AND WHERE EACH IS CAUGHT:

  a valid graph, every shape declared and every Add/Mul compatible
    --> no exception -- shapes computed for every node, returned normally

  Add/Mul with shapes that share nothing broadcastable
    ([3,4] vs [5,6]: neither dimension pair is equal or 1)
    --> caught INSIDE broadcastShapes() itself, the moment the
        incompatible pair is compared

  Add/Mul with shapes that are close but not quite compatible
    ([3,4] vs [3,5]: the trailing pair, 4 vs 5, fails even though the
     OTHER pair, 3 vs 3, matches)
    --> caught the SAME way -- broadcastShapes() checks every aligned
        pair independently, so a partial match never short-circuits it

  a leaf (Input/Const) with no entry in declaredShapes at all
    --> caught in inferShapes() itself, BEFORE broadcastShapes() is
        ever reached for that leaf's own consumers
```

### Background

File 010 below reuses File 009's `Shape`, `broadcastShapes()`, `Graph`, `topologicalSort()`, and `inferShapes()` completely unchanged -- Section 6.3 adds no new inference logic at all, only a test harness proving the checks already built into that code actually fire, and fire only when they should. Three deliberately broken graphs are tested: two different flavors of genuinely incompatible shapes (one where *no* dimension pair lines up, one where only the trailing pair fails despite another pair matching, specifically to rule out any "partial credit" bug), and one graph where a leaf's declared shape was simply never provided. A fourth, valid control case -- File 009's own diamond graph, run through `inferShapes()` again here -- confirms none of these checks misfire on correct input, the same control-case discipline Chapter 5's own File 008 used for parse errors.

```cpp
// Chapter 6: Graph Validation and Shape Inference
// 010_shape_validation_errors.cpp
//
// Section 6.3 -- catching a real shape mismatch, and a missing declared
// leaf shape, before a graph is ever handed to a later pass or codegen
// stage that would trust its shapes blindly.
//
// Reuses File 009's own Shape/broadcastShapes()/Graph/topologicalSort()/
// inferShapes() completely unchanged. The only new code here is the test
// harness: two genuinely shape-incompatible graphs, one graph missing a
// declared leaf shape, and one valid control graph (File 009's own
// diamond) confirming none of these checks misfire on correct input --
// the same four-invalid-cases-plus-one-control-case discipline Chapter
// 5's own File 008 used for parse errors.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 010_shape_validation_errors.cpp -o 010_shape_validation_errors
// Run:     ./010_shape_validation_errors
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include <functional>

// ==================== Value / Node / Graph / TopoResult / Shape (from File 009) ====================

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
                                      " are not broadcast-compatible (" + std::to_string(da) + " vs " +
                                      std::to_string(db) + " at trailing position " + std::to_string(i) + ")");
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
        std::string context = "node %" + std::to_string(id) + " (" + n->debugName + ")";

        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            auto it = declaredShapes.find(id);
            if (it == declaredShapes.end()) {
                throw std::runtime_error(context + ": no declared shape was provided for this leaf node");
            }
            shapes[id] = it->second;
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            const Shape& lhs = shapes.at(n->inputs[0].nodeId);
            const Shape& rhs = shapes.at(n->inputs[1].nodeId);
            shapes[id] = broadcastShapes(lhs, rhs, context);
        } else {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

// ============================== Test harness (new) ==============================

struct TestCase {
    std::string name;
    std::function<void()> run;  // throws on failure, returns normally on success
    bool expectError;
};

int main() {
    printf("=== Section 6.3: rejecting shape-invalid graphs instead of trusting them blindly ===\n\n");

    std::vector<TestCase> tests;

    tests.push_back({
        "valid graph (control case: File 009's own diamond)",
        [] {
            Graph g;
            Value a  = g.addInput("a");
            Value b  = g.addInput("b");
            Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
            Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
            Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
            g.addBinary(OpKind::Add, t2, t3, "out");
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
            inferShapes(g, declared);
        },
        false
    });

    tests.push_back({
        "incompatible shapes, neither dimension matches or is 1",
        [] {
            Graph g;
            Value a = g.addInput("a");
            Value b = g.addInput("b");
            g.addBinary(OpKind::Add, a, b, "bad");
            // [3, 4] vs [5, 6] -- at the trailing position, 4 vs 6: not
            // equal, and neither is 1. Genuinely incompatible.
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{5, 6}}}};
            inferShapes(g, declared);
        },
        true
    });

    tests.push_back({
        "incompatible shapes, only the leading dimension differs (still rejected)",
        [] {
            Graph g;
            Value a = g.addInput("a");
            Value b = g.addInput("b");
            g.addBinary(OpKind::Mul, a, b, "bad");
            // [3, 4] vs [3, 5] -- trailing position 4 vs 5: not equal,
            // neither is 1, so this is STILL incompatible even though
            // the OTHER dimension (3 vs 3) matches perfectly -- every
            // aligned pair has to be individually compatible, matching
            // is not "good enough on average."
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{3, 5}}}};
            inferShapes(g, declared);
        },
        true
    });

    tests.push_back({
        "missing declared shape for a leaf node",
        [] {
            Graph g;
            Value a = g.addInput("a");
            Value b = g.addInput("b");  // deliberately never given a declared shape below
            g.addBinary(OpKind::Add, a, b, "bad");
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}};  // "b" missing on purpose
            inferShapes(g, declared);
        },
        true
    });

    bool allBehavedAsExpected = true;
    for (const auto& tc : tests) {
        printf("--- %s ---\n", tc.name.c_str());
        bool threw = false;
        std::string errorMessage;
        try {
            tc.run();
        } catch (const std::exception& e) {
            threw = true;
            errorMessage = e.what();
        }
        bool behavedAsExpected = (threw == tc.expectError);
        allBehavedAsExpected = allBehavedAsExpected && behavedAsExpected;

        if (threw) {
            printf("  threw: \"%s\"\n", errorMessage.c_str());
        } else {
            printf("  inferred successfully, no exception thrown\n");
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
g++ -std=c++17 -Wall -Wextra -O2 010_shape_validation_errors.cpp -o 010_shape_validation_errors
./010_shape_validation_errors
```

**Output:**

```text
=== Section 6.3: rejecting shape-invalid graphs instead of trusting them blindly ===

--- valid graph (control case: File 009's own diamond) ---
  inferred successfully, no exception thrown
  expected success, got success -- confirmed

--- incompatible shapes, neither dimension matches or is 1 ---
  threw: "node %2 (bad): shapes [3, 4] and [5, 6] are not broadcast-compatible (4 vs 6 at trailing position 0)"
  expected an exception, got an exception -- confirmed

--- incompatible shapes, only the leading dimension differs (still rejected) ---
  threw: "node %2 (bad): shapes [3, 4] and [3, 5] are not broadcast-compatible (4 vs 5 at trailing position 0)"
  expected an exception, got an exception -- confirmed

--- missing declared shape for a leaf node ---
  threw: "node %1 (b): no declared shape was provided for this leaf node"
  expected an exception, got an exception -- confirmed

self-check: every test case behaved exactly as expected (confirmed)
```

!!! warning "[COMMON TRAP] Assuming a missing declared shape and an incompatible pair of shapes are the same kind of bug"
    Both eventually surface as a thrown exception, which can make them feel like the same failure -- but they are caught at genuinely different places for a real reason. A missing declared shape is caught inside `inferShapes()` itself, the instant a leaf with no entry in `declaredShapes` is visited -- it is purely a bookkeeping problem, nothing about the graph's own structure is wrong. An incompatible shape pair is caught inside `broadcastShapes()`, and only becomes visible once two *already-known* shapes are actually compared -- it is a real statement about the computation itself being impossible to execute, not a missing piece of input. Conflating the two would make debugging harder in practice: "you forgot to declare a shape" and "the computation you described cannot work for any declared shapes" call for completely different fixes.

## Chapter Summary

Chapters 4 and 5 built and parsed graphs entirely in terms of edges -- which value depends on which -- with no notion of size anywhere in `Node` itself. This chapter added shape as a separate, computed side table, kept that way deliberately so future passes never need to touch `Node`'s own definition to add a new kind of analysis. Section 6.1 established the precise rule two shapes must satisfy to combine at all: NumPy's own stated rule (quoted directly, since this is a real, specific fact about how real frameworks work), comparing two shapes from their trailing dimension leftward, treating a missing dimension as `1`, and requiring every aligned pair to be either equal or have a `1` on one side. Section 6.2 built `inferShapes()`, which computes every node's own shape by walking Chapter 4's `topologicalSort()` order exactly once -- a leaf's shape comes from a caller-supplied declaration, `Add`/`Mul` broadcast their already-known inputs, `ReLU` preserves its input's shape -- correctness guaranteed by the same "every input is already computed by the time you need it" property that made topological order the right tool for Chapter 4's own cycle-freedom proof. Section 6.3 proved the resulting checks work both ways: two genuinely different flavors of incompatible shapes, and a missing declared leaf shape, are each caught with a specific, descriptive error naming the exact node and shapes involved, while a valid control graph confirms none of those checks misfire on correct input. Part 1 is now complete: Chapter 7 closes it out with a real printer for this same `Graph`, and Part 2 begins writing genuine optimization passes over graphs that Part 1's own three chapters -- representation, frontend, and now validation -- guarantee are well-formed before any pass ever touches them.

## Self-Check Questions

1. Why does shape live in a separate `map<int, Shape>` rather than as a field directly on `Node`?
2. NumPy's own documentation states two dimensions are compatible "when they are equal, or one of them is 1." Using that rule, is `[6]` compatible with `[3]`? Walk through why or why not.
3. In the `[3, 1]` vs. `[1, 4]` example, neither shape is "the one being broadcast" -- explain what that means concretely, using the two aligned positions.
4. Why does `inferShapes()` call `topologicalSort()` and walk its `order`, rather than simply iterating `g.nodes()` directly the way File 009's earlier diamond-printing code in Chapter 4 did?
5. `ReLU`'s shape rule is "shape-preserving." Why is that the correct rule for `ReLU` specifically, and would it still be correct for an operation that summed all of a tensor's elements down to a single number?
6. File 010 tests two different kinds of incompatible-shape graphs (`[3,4]` vs `[5,6]`, and `[3,4]` vs `[3,5]`) rather than just one. What specific bug would testing only the first case risk leaving undetected?
7. Per Section 6.3's own [COMMON TRAP], why are a missing declared shape and an incompatible shape pair caught in two different places in the code, rather than both being checked by `inferShapes()` itself up front?
8. If `broadcastShapes()` had a bug that made it return the *smaller* dimension instead of the larger one whenever neither side was `1` (instead of correctly throwing), would File 009's own five-case test table have caught that bug? Why or why not?

## Where We Go Next

Part 1 has now built every piece Part 2 needs to assume is already true: a real graph data structure (Chapter 4), a way to build one from text (Chapter 5), and a way to check whether a graph's shapes actually make sense (this chapter). Chapter 7 closes out Part 1 with a real, proper printer for `Graph` -- something more complete than the ad hoc `printGraph()` helper every chapter since Chapter 4 has rewritten a slightly different version of -- before Part 2 starts writing the first genuine optimization passes over these now-validated graphs.

## Worked Solutions

1. Because shape is not a structural fact about the graph the way an edge is -- it has to be *computed*, and different future passes may want different derived facts about the same graph (shape today, something like a cost estimate or a memory-liveness range later) without each one needing its own field bolted onto `Node`. Keeping shape (and any future analysis result) as an external side table, produced by a pass that reads the graph and returns a fresh structure, means `Node`'s own definition never has to change to support a new kind of analysis.
2. No. Aligning `[6]` against `[3]` at their one shared trailing position gives `da=6`, `db=3` -- they are not equal, and neither one is `1`, so the compatibility test ("equal, or one of them is 1") fails on both of its conditions. `broadcastShapes()` would throw for this pair, the same way it does for `[3,4]` vs `[5,6]`.
3. At the trailing position, `A=1` (from `[3,1]`) and `B=4` (from `[1,4]`) -- here `A` is the one that is `1`, so the result takes `B`'s size, `4`. At the next position left, `A=3` and `B=1` -- here `B` is the one that is `1`, so the result takes `A`'s size, `3`. Each shape "wins" (contributes its own real size to the result) at exactly the position where the other shape only has a `1` -- neither shape is the broadcast one throughout; each is, at a different position.
4. `topologicalSort()`'s `order` is *guaranteed*, by Chapter 4's own construction, to visit every node after all of its own producers -- that guarantee is exactly what lets `inferShapes()` assume `shapes.at(...)` will always find an already-computed shape for any input it looks up. Iterating `g.nodes()` directly happens to give the same order for this book's own simple, append-only graph-construction API (Chapter 4's own `addBinary`/`addUnary` can only reference an already-existing node, so ids are always assigned in a valid order already) -- but `inferShapes()` does not rely on that coincidence, specifically so it keeps working correctly if a future pass ever builds or reorders a `Graph` in a way where node id order and topological order are no longer the same thing.
5. `ReLU` computes `max(0, x)` independently for every element of its input -- it changes individual *values*, never how many elements exist or how they're arranged, so its output shape is identical to its input shape by definition. A sum-to-a-single-number reduction is fundamentally different: it takes every element of a possibly large input and produces exactly one number, so "shape-preserving" would be flatly wrong for it -- that operation would need its own shape rule (output shape `[1]`, or `[]`, regardless of the input's shape), which is exactly why shape inference has to dispatch on each specific operation's own real behavior rather than applying one universal rule to everything.
6. Testing only `[3,4]` vs `[5,6]` (where *neither* dimension pair matches at all) would risk leaving undetected a bug where `broadcastShapes()` incorrectly treated "at least one dimension pair matches somewhere" as sufficient for overall compatibility -- a bug that would wrongly accept `[3,4]` vs `[3,5]` (where the leading pair, `3` vs `3`, matches, even though the trailing pair, `4` vs `5`, does not). Testing both cases separately confirms the function checks every aligned position independently, with no shortcut where a match at one position can compensate for a mismatch at another.
7. Because they are genuinely different problems needing different fixes: a missing declared shape means the graph's own structure and computation are perfectly fine, but the caller simply never supplied a piece of required input (an easy, mechanical fix -- provide the shape); an incompatible shape pair means the computation the graph describes cannot execute for *any* pair of declared shapes with those particular sizes, a real statement about the tensor program itself being wrong. Checking both the same way, in the same place, would blur that distinction in the error message a user actually sees.
8. No, it would not. Walk all five of File 009's own test cases through the buggy `else` branch's trigger condition -- "neither side is `1` and the two dimensions are unequal" -- and check whether any aligned position in any of the five ever reaches it: `[3,4]`/`[3,4]` hits only the "equal" branch at both positions; `[3,4]`/`[4]` hits "equal" at the trailing position and "`db` is 1" at the padded position; `[3,1]`/`[1,4]` hits "`da` is 1" then "`db` is 1"; `[5]`/`[1]` hits "`db` is 1"; `[2,3,4]`/`[4]` hits "equal" once and "`db` is 1" (padding) twice. Not one of the five ever reaches the buggy `else` branch at all, because all five were deliberately chosen to be *compatible* pairs -- Section 6.1's own point was testing broadcasting that succeeds. A bug that only misbehaves on genuinely incompatible input is invisible to a test table built entirely from compatible input; catching it would need a new case in that same table -- two shapes with an aligned, unequal, neither-is-1 pair -- which File 010's own incompatible-shape tests happen to construct, but check only for "throws," not for "returns the wrong dimension instead of throwing," so even File 010 would need a small addition (asserting an exception is thrown, which it already does, AND that no dimension value is ever silently returned) to fully close this specific gap.

---

**Sources cited in this chapter:**

- NumPy, ["Broadcasting"](https://numpy.org/doc/stable/user/basics.broadcasting.html) -- the exact broadcasting rule this chapter's own `broadcastShapes()` implements, quoted directly: "When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimension and works its way left," and "Two dimensions are compatible when 1. they are equal, or 2. one of them is 1." Used in Section 6.1. `broadcastShapes()`'s own implementation is original to this book, not copied from NumPy's own source -- only the rule it enforces is the same rule, which is also the one PyTorch and TensorFlow both use.
