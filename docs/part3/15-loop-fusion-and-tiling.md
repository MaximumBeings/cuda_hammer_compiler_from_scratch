# 15. Loop Fusion and Tiling

**What you will understand:** `LoopNest`, the first explicit representation of actual ITERATION this book's IR has ever needed -- built entirely from information Chapters 6 and 14 already computed (`Shape`, `reduceElementCount`), not new data the graph has to carry; `tileLoop()`, which splits one loop into an outer/inner pair while correctly handling the boundary tile that doesn't divide evenly; and `loopNestsCompatibleForFusion()`, a second, independent axis of fusion -- separate from Chapters 13-14's own operator-fusion question -- that asks whether two nodes' own loops could still merge at codegen time even when their FusedStep programs stayed apart.

**What you need to know first:** Chapter 6's `Shape`/`inferShapes()`/`numElements()`, Chapter 13's entire fusion IR and `elementwiseFusionPass()`, and Chapter 14's `OpKind::Sum`/`FusedReduction`/`reductionFusionPass()` -- this chapter builds a new VIEW on top of all of it rather than changing any of it.

---

Every fused node Chapters 13 and 14 ever built answers one question precisely: WHAT does this node compute, expressed as a small ordered program of `FusedStep`s. Neither chapter ever had to answer a second, equally real question: HOW MANY TIMES does that program actually run, and in what order? `evaluate()`'s own representative-scalar convention -- one scalar standing in for a whole tensor, honest since Chapter 9 about what it does and doesn't model -- was exactly why that second question never had to be answered. It matters now for a concrete reason: Part 4 of this book generates real loops, and a real loop needs a real iteration count before it can be written down at all. This chapter builds the piece that answers it -- `LoopNest`, computed directly from Shape information the IR already carries -- and two things codegen and autotuning will both need done to that iteration space: splitting one loop into a tile-sized pair (tiling), and recognizing when two SEPARATE nodes' own loops could still be merged into one shared loop nest even though operator fusion never merged their programs (loop fusion, a genuinely different, independent question). Real compilers keep these apart on purpose -- Chapter 3 already cited TVM's own real schedule primitives, where `split`, `tile`, `reorder`, and `fuse` are each their own separate operation, layered on top of whatever earlier pass decided which computations belong together at all.

```text
WHAT LOOP-LEVEL REASONING ADDS ON TOP OF OPERATOR FUSION:

  OPERATOR FUSION (Chapters 13-14) answers ONE question:
    which VALUES share a single node's own internal FusedStep program?

  THIS CHAPTER answers TWO MORE, independent questions:

    (1) how many times does a given node's own program actually RUN?
          -> Section 15.1: LoopNest, built from Shape + reduceElementCount

    (2) how much of a tensor's own data does ONE run touch at a time?
          -> Section 15.2: tiling, splitting one loop into an outer/inner pair

    (3) even when two nodes stay SEPARATE, could their own loops still
        merge into one shared loop nest once real code gets generated?
          -> Section 15.3: loopNestsCompatibleForFusion()

  None of this changes WHAT a fused node computes -- Chapters 13-14's
  own FusedStep programs are untouched by every idea in this chapter.
  It gives CUDA Hammer a real answer to "how does this become an
  actual loop" -- the question Part 4 (Lowering to Loops) needs
  answered before it can emit one line of real generated code.
```

## 15.1 From Shapes to Loop Nests: Giving Every Computed Node Real Iteration Bounds

### Intuition

Picture handing someone Chapter 13's own recipe card for a fused kernel -- a short list of steps, "add these two, then relu the result" -- without ever telling them how many TIMES to run the recipe, or on what. The recipe alone is not a real set of instructions; it needs a loop wrapped around it, and that loop needs real bounds. For a plain elementwise node, those bounds are hiding in plain sight: the node's own OUTPUT shape already says exactly how many elements it produces, and it produces exactly one output element per run of the recipe, so the shape IS the iteration space. A reduction breaks that equivalence the same way it broke everything else in Chapter 14: its own output shape is a scalar -- zero loops' worth of information -- even though its recipe has to run once per element of whatever it's summing. The loop nest a reduction needs has one loop its own output shape can never supply, and that loop's extent has to come from somewhere else: exactly the `reduceElementCount` Chapter 14's own pass already computed and stored, for a completely different reason, inside the `Sum` step itself.

### Background

A `LoopNest` here is nothing more than an ordered list of `Loop`s -- each one a name (for printing and, later, for real generated variable names) and an extent (how many times it runs) -- read outermost-first, the same convention every real loop-nest compiler uses, including the ones Chapter 3 already surveyed. `buildLoopNest()` reads no new data at all: for an `Add`/`Mul`/`ReLU`/`FusedElementwise` node, it takes the node's own already-inferred output `Shape` and emits one `Loop` per dimension, in order -- the exact per-element correspondence Chapter 13's own `FusedStep` model already depends on, now given an explicit iteration-space home. For a `FusedReduction` node, that same per-dimension walk over the (now-scalar) output shape contributes zero loops, so `buildLoopNest()` appends exactly one more: a `Loop` named `"reduce"`, whose extent is read straight out of the already-fused `Sum` step's own `reduceElementCount` -- the count Chapter 14's pass baked in at fusion time, reused here rather than recomputed. The function even handles a plain, not-yet-fused `OpKind::Sum` node the same way, falling back to the `elementCounts` table Chapter 14's own `evaluate()` already needed, which lets this section prove something concrete: the total iteration count a reduction's own loop nest reports is IDENTICAL before and after fusion runs. Fusion changes how a value gets computed -- Chapters 13 and 14 already measured that as a real change in bytes moved -- but it never changes how many total iterations the underlying computation actually needs, and this section proves that by direct comparison rather than simply asserting it, the same discipline every earlier chapter's own "does this claim actually hold" moment has used.

```text
A NODE'S OWN LOOP NEST, BUILT FROM INFORMATION THE IR ALREADY HAS:

  FusedElementwise node, output shape [3, 4]:      FusedReduction node, output shape [] (scalar):

    for dim0 in 0..2:                                for reduce in 0..3:
      for dim1 in 0..3:                                  accumulate into the ONE output
        run the node's own FusedStep program              (the node's own LAST FusedStep
        once per (dim0, dim1) pair                        is always this Sum)

    loop nest = [dim0:3, dim1:4]                      loop nest = [reduce:4]
    total iterations = 12                              total iterations = 4
    (matches Shape's own numElements() exactly)        (matches the Sum step's own
                                                         already-baked-in reduceElementCount)
```

```cpp
// Chapter 15: Loop Fusion and Tiling
// 033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds.cpp
//
// Section 15.1 -- everything Chapters 13 and 14 built lives at the level of
// a fused node's own INTERNAL PROGRAM: a FusedStep sequence, executed once
// per element by evaluate()'s own representative-scalar convention. That
// convention has always been honest about what it is NOT modeling: how many
// TIMES that program actually runs, and in what order, once CUDA Hammer
// starts generating real loops (Part 4). This section builds the first
// piece of that missing picture -- an explicit LoopNest, computed directly
// from information the IR already has (Chapter 6's Shape, Chapter 14's own
// reduceElementCount) rather than anything new the graph has to carry.
//
// The central new idea: a FusedElementwise node's own loop nest is just its
// OUTPUT shape's dimensions, one loop per dimension -- the node computes one
// output element per iteration, so the output shape IS the iteration space.
// A FusedReduction node breaks that equivalence the same way Chapter 14's
// own OpKind::Sum broke evaluate()'s per-element correspondence: its output
// shape is smaller than its own iteration space (a scalar output, Shape{},
// for however many elements it summed), so its loop nest needs ONE MORE
// loop than its output shape alone would suggest -- the reduction axis,
// whose extent is exactly the reduceElementCount Chapter 14's own pass
// already baked into the Sum step.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds.cpp -o 033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds
// Run:     ./033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind (from Chapters 13-14, unchanged) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

// ==================== FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand {
    OperandKind kind;
    int index;
};

struct FusedStep {
    OpKind op;
    std::vector<FusedOperand> operands;
    long long reduceElementCount = 1;
};

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;
    float constValue = 0.0f;
    std::vector<FusedStep> fusedSteps;
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
    Value addFusedElementwise(std::vector<Value> externalInputs, std::vector<FusedStep> steps,
                               const std::string& name) {
        Value out = addNode(OpKind::FusedElementwise, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
    }
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps,
                             const std::string& name) {
        Value out = addNode(OpKind::FusedReduction, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
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

// ==================== Shape / broadcastShapes / inferShapes (from Chapters 6 and 14, unchanged) ====================

struct Shape { std::vector<int> dims; };

static std::string shapeStr(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) { if (i) out += ", "; out += std::to_string(s.dims[i]); }
    out += "]";
    return out;
}
static long long numElements(const Shape& s) {
    long long n = 1;
    for (int d : s.dims) n *= d;
    return n;
}
static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        int outDim;
        if (da == db) outDim = da;
        else if (da == 1) outDim = db;
        else if (db == 1) outDim = da;
        else throw std::runtime_error(context + ": shapes " + shapeStr(a) + " and " + shapeStr(b) + " incompatible");
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
            shapes[id] = declaredShapes.at(id);
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        } else if (n->op == OpKind::ReLU) {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        } else {  // Sum -- inferShapes() is only ever called on a graph BEFORE
                  // fusion runs (every chapter through 14 has called it that
                  // way), so FusedElementwise/FusedReduction never reach here.
            shapes[id] = Shape{};
        }
    }
    return shapes;
}

// ==================== resolveIntoGroup / elementwiseFusionPass / reductionFusionPass (from Chapter 14, unchanged) ====================

struct FusionResult {
    Graph graph;
    std::map<int, int> representativeOldId;
};

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized,
                                      std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId,
                                      std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) {
        return FusedOperand{OperandKind::PriorStep, stepIt->second};
    }
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
                           (consumers.at(oldId) != 1);
    if (mustBeExternal) {
        auto extIt = externalIndexByOldId.find(oldId);
        if (extIt != externalIndexByOldId.end()) {
            return FusedOperand{OperandKind::ExternalInput, extIt->second};
        }
        int idx = static_cast<int>(externalInputs.size());
        externalInputs.push_back(materialized.at(oldId));
        externalIndexByOldId[oldId] = idx;
        return FusedOperand{OperandKind::ExternalInput, idx};
    }
    std::vector<FusedOperand> operands;
    for (const Value& in : p->inputs) {
        operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                             externalIndexByOldId, steps, stepIndexByOldId));
    }
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}

static FusionResult elementwiseFusionPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("elementwiseFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    FusionResult result;
    Graph& out = result.graph;
    std::map<int, Value> materialized;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        if (n->op == OpKind::Input) {
            Value v = out.addInput(n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Const) {
            Value v = out.addConst(n->constValue, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
        }
        steps.push_back(FusedStep{n->op, operands});
        Value v;
        if (steps.size() == 1) {
            if (n->op == OpKind::ReLU) v = out.addUnary(OpKind::ReLU, externalInputs[0], n->debugName);
            else v = out.addBinary(n->op, externalInputs[0], externalInputs[1], n->debugName);
        } else {
            v = out.addFusedElementwise(externalInputs, steps, n->debugName);
        }
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

static FusionResult reductionFusionPass(const Graph& g, const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("reductionFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    FusionResult result;
    Graph& out = result.graph;
    std::map<int, Value> materialized;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        if (n->op == OpKind::Input) {
            Value v = out.addInput(n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Const) {
            Value v = out.addConst(n->constValue, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Sum) {
            std::vector<Value> externalInputs;
            std::map<int, int> externalIndexByOldId;
            std::vector<FusedStep> steps;
            std::map<int, int> stepIndexByOldId;
            FusedOperand summedOperand = resolveIntoGroup(n->inputs[0].nodeId, g, consumers, materialized,
                                                            externalInputs, externalIndexByOldId,
                                                            steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
        }
        steps.push_back(FusedStep{n->op, operands});
        Value v;
        if (steps.size() == 1) {
            if (n->op == OpKind::ReLU) v = out.addUnary(OpKind::ReLU, externalInputs[0], n->debugName);
            else v = out.addBinary(n->op, externalInputs[0], externalInputs[1], n->debugName);
        } else {
            v = out.addFusedElementwise(externalInputs, steps, n->debugName);
        }
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

// ==================== Section 15.1: Loop / LoopNest / buildLoopNest() ====================
//
// A Loop is nothing more than a name (for printing) and an extent (how many
// times it runs). A LoopNest is an ORDERED list of Loops, read outermost
// first -- the same convention every real loop-nest-based compiler uses,
// including the ones Chapter 3 already surveyed (TVM's own schedule
// primitives operate on exactly this kind of explicit, ordered loop list).
struct Loop {
    std::string dimName;
    long long extent;
};
struct LoopNest {
    std::vector<Loop> loops;
    long long totalIterations() const {
        long long total = 1;
        for (const Loop& l : loops) total *= l.extent;
        return total;  // an EMPTY loop list is the empty product: 1 iteration,
                        // the same "scalar means one" convention Chapter 6's
                        // own numElements() already established for shapes.
    }
};
static std::string loopNestStr(const LoopNest& nest) {
    std::string out = "[";
    for (size_t i = 0; i < nest.loops.size(); ++i) {
        if (i) out += ", ";
        out += nest.loops[i].dimName + ":" + std::to_string(nest.loops[i].extent);
    }
    out += "]";
    return out;
}

// buildLoopNest() reads ONLY information the IR already has: Chapter 6's own
// inferred output Shape, and -- for a reduction -- the reduceElementCount
// Chapter 14's own pass already baked into the Sum step (or, for a Sum node
// that hasn't been fused yet, elementCounts, the same table Chapter 14's own
// evaluate() already needed). Nothing here is new DATA -- it is a new VIEW
// of data every earlier chapter already computed.
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    if (n->op == OpKind::Input || n->op == OpKind::Const) {
        throw std::runtime_error("buildLoopNest: " + n->debugName +
                                  " is materialized from memory or a literal, not computed by a loop");
    }
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) {
        nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    }
    if (n->op == OpKind::Sum) {
        // Not yet fused -- the reduction axis's own extent has to come from
        // shape information looked up externally, the same table Chapter
        // 14's own reductionFusionPass() and evaluate() both already needed.
        nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    } else if (n->op == OpKind::FusedReduction) {
        // Already fused -- the reduction axis's own extent is already
        // sitting inside the Sum step, baked in by the pass at fusion time.
        // Chapter 14's own convention (the Sum step is always LAST) is what
        // makes fusedSteps.back() reliable here.
        long long reduceExtent = n->fusedSteps.back().reduceElementCount;
        nest.loops.push_back(Loop{"reduce", reduceExtent});
    }
    // Add, Mul, ReLU, FusedElementwise: outShape's own dims ARE the loop
    // nest -- nothing more to add. One output element per iteration, same
    // per-element correspondence Chapters 13-14 already relied on.
    return nest;
}

int main() {
    printf("=== Section 15.1: buildLoopNest() -- from Shape to real iteration bounds ===\n\n");

    // ---- Test 1: Chapter 13's own diamond graph (a=[3,4], b=[4]) ----
    // Hand-derivation before running anything: a=Input, b=Input,
    // t1=Add(a,b) has TWO consumers (t2, t3) so it stays a plain node;
    // t2=Mul(t1,a), t3=ReLU(t1), out=Add(t2,t3) are all single-consumer and
    // fuse into one FusedElementwise. Every non-leaf node's own shape is
    // [3,4] (Chapter 6's own broadcast result), so BOTH t1 (plain Add) and
    // the fused node share the exact same loop nest: [dim0:3, dim1:4],
    // 12 total iterations -- matching Chapter 12's own "12 elements/node"
    // figure for this graph exactly.
    Graph diamond;
    Value a  = diamond.addInput("a");
    Value b  = diamond.addInput("b");
    Value t1 = diamond.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = diamond.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = diamond.addUnary(OpKind::ReLU, t1, "t3");
    Value dout = diamond.addBinary(OpKind::Add, t2, t3, "out");
    (void)dout;

    std::map<int, Shape> declaredD = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> shapesD = inferShapes(diamond, declaredD);

    FusionResult diamondFused = elementwiseFusionPass(diamond);
    printf("Diamond graph after elementwiseFusionPass() (%zu nodes):\n", diamondFused.graph.size());

    const Node* plainT1 = nullptr;
    const Node* fusedNode = nullptr;
    for (const auto& n : diamondFused.graph.nodes()) {
        if (n->debugName == "t1") plainT1 = n.get();
        if (n->op == OpKind::FusedElementwise) fusedNode = n.get();
    }
    bool structureOk = (plainT1 != nullptr && fusedNode != nullptr && diamondFused.graph.size() == 4);
    printf("self-check: t1 stayed a plain node and one FusedElementwise node was built, 6 -> 4 (%s)\n",
           structureOk ? "confirmed" : "MISMATCH");

    // shapesD was computed on the ORIGINAL graph; both t1 and the fused
    // node's own output are the SAME computed value as before fusion
    // (fusion only changes HOW a value is computed, never its own shape),
    // so shapesD.at(originalId) is the right table to read from here.
    Shape t1Shape = shapesD.at(t1.nodeId);
    Shape outShape = shapesD.at(dout.nodeId);

    LoopNest t1Nest = buildLoopNest(plainT1, {{plainT1->id, t1Shape}}, {});
    LoopNest fusedNest = buildLoopNest(fusedNode, {{fusedNode->id, outShape}}, {});

    printf("\nt1's own loop nest (plain Add, shape %s): %s, %lld total iterations\n",
           shapeStr(t1Shape).c_str(), loopNestStr(t1Nest).c_str(), t1Nest.totalIterations());
    printf("fused node's own loop nest (shape %s):    %s, %lld total iterations\n",
           shapeStr(outShape).c_str(), loopNestStr(fusedNest).c_str(), fusedNest.totalIterations());

    bool t1Matches = (t1Nest.totalIterations() == numElements(t1Shape)) && (t1Nest.loops.size() == 2) &&
                      (t1Nest.loops[0].extent == 3) && (t1Nest.loops[1].extent == 4);
    bool fusedMatches = (fusedNest.totalIterations() == numElements(outShape)) && (fusedNest.loops.size() == 2) &&
                         (fusedNest.loops[0].extent == 3) && (fusedNest.loops[1].extent == 4);
    printf("self-check: both loop nests are [dim0:3, dim1:4], 12 iterations, agreeing with Chapter 6's\n");
    printf("own numElements() on the same shape (%s)\n", (t1Matches && fusedMatches) ? "confirmed" : "MISMATCH");

    // ---- Test 2: Chapter 14's own reduction graph (x=[4]) ----
    // Hand-derivation: x=Input, t1=ReLU(x), s=Sum(t1). t1 has one consumer
    // (s) and inlines into s's own FusedReduction body: 2 steps (ReLU, Sum),
    // the Sum step carrying reduceElementCount=4 (baked in from x's own 4
    // elements). s's own OUTPUT shape is Shape{} -- zero dims, zero loops
    // from that alone -- so its loop nest needs the one EXTRA loop this
    // section's whole argument is about: a single "reduce" loop of extent
    // 4, read straight out of the already-fused step, not recomputed.
    printf("\n=== Test 2: a FusedReduction node's own loop nest needs ONE loop shape alone can't supply ===\n\n");

    Graph rg;
    Value x  = rg.addInput("x");
    Value rt1 = rg.addUnary(OpKind::ReLU, x, "t1");
    Value s  = rg.addUnary(OpKind::Sum, rt1, "s");
    (void)s;

    std::map<int, Shape> declaredR = {{x.nodeId, Shape{{4}}}};
    std::map<int, Shape> shapesR = inferShapes(rg, declaredR);
    std::map<int, long long> elementCountsR;
    for (const auto& kv : shapesR) elementCountsR[kv.first] = numElements(kv.second);

    FusionResult rFused = reductionFusionPass(rg, elementCountsR);
    const Node* fusedReduction = nullptr;
    for (const auto& n : rFused.graph.nodes()) {
        if (n->op == OpKind::FusedReduction) fusedReduction = n.get();
    }
    bool reductionStructureOk = (fusedReduction != nullptr && rFused.graph.size() == 2);
    printf("self-check: t1 inlined into s's own FusedReduction body, 3 -> 2 nodes (%s)\n",
           reductionStructureOk ? "confirmed" : "MISMATCH");

    Shape sOutShape = shapesR.at(s.nodeId);
    LoopNest reductionNest = buildLoopNest(fusedReduction, {{fusedReduction->id, sOutShape}}, {});
    printf("\ns's own output shape: %s (zero dims -- a scalar, zero loops from shape alone)\n",
           shapeStr(sOutShape).c_str());
    printf("s's own loop nest:    %s, %lld total iterations\n",
           loopNestStr(reductionNest).c_str(), reductionNest.totalIterations());

    bool reduceLoopOk = (reductionNest.loops.size() == 1) && (reductionNest.loops[0].dimName == "reduce") &&
                         (reductionNest.loops[0].extent == 4) &&
                         (reductionNest.loops[0].extent == fusedReduction->fusedSteps.back().reduceElementCount);
    printf("self-check: the loop nest's one loop is a \"reduce\" loop of extent 4, agreeing exactly\n");
    printf("with the Sum step's own already-baked-in reduceElementCount (%s)\n", reduceLoopOk ? "confirmed" : "MISMATCH");

    // ---- Test 3: the loop nest is INVARIANT across the fusion boundary ----
    // Fusion changes HOW s is computed (one fused kernel instead of two
    // separate ones) -- Chapters 13 and 14 already measured that as a real
    // change in bytes moved. It does NOT change how many total iterations
    // s's own reduction needs: build the loop nest for the ORIGINAL,
    // unfused Sum node (still a plain OpKind::Sum at this point) and confirm
    // it is the exact same total iteration count as the fused version above.
    printf("\n=== Test 3: the loop nest's own total iteration count survives fusion unchanged ===\n\n");

    const Node* plainSum = rg.node(s.nodeId);
    LoopNest unfusedReductionNest = buildLoopNest(plainSum, shapesR, elementCountsR);
    printf("unfused Sum node's own loop nest: %s, %lld total iterations\n",
           loopNestStr(unfusedReductionNest).c_str(), unfusedReductionNest.totalIterations());
    printf("fused   node's own loop nest:     %s, %lld total iterations\n",
           loopNestStr(reductionNest).c_str(), reductionNest.totalIterations());

    bool invariant = (unfusedReductionNest.totalIterations() == reductionNest.totalIterations()) &&
                      (unfusedReductionNest.loops.size() == reductionNest.loops.size()) &&
                      (unfusedReductionNest.loops[0].extent == reductionNest.loops[0].extent);
    printf("self-check: fusion changed HOW s is computed but not how many total iterations its own\n");
    printf("reduction needs -- the loop nest itself is invariant across the fusion boundary (%s)\n",
           invariant ? "confirmed" : "MISMATCH");

    bool allOk = structureOk && t1Matches && fusedMatches && reductionStructureOk && reduceLoopOk && invariant;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds.cpp -o 033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds
./033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds
```

**Output:**

```text
=== Section 15.1: buildLoopNest() -- from Shape to real iteration bounds ===

Diamond graph after elementwiseFusionPass() (4 nodes):
self-check: t1 stayed a plain node and one FusedElementwise node was built, 6 -> 4 (confirmed)

t1's own loop nest (plain Add, shape [3, 4]): [dim0:3, dim1:4], 12 total iterations
fused node's own loop nest (shape [3, 4]):    [dim0:3, dim1:4], 12 total iterations
self-check: both loop nests are [dim0:3, dim1:4], 12 iterations, agreeing with Chapter 6's
own numElements() on the same shape (confirmed)

=== Test 2: a FusedReduction node's own loop nest needs ONE loop shape alone can't supply ===

self-check: t1 inlined into s's own FusedReduction body, 3 -> 2 nodes (confirmed)

s's own output shape: [] (zero dims -- a scalar, zero loops from shape alone)
s's own loop nest:    [reduce:4], 4 total iterations
self-check: the loop nest's one loop is a "reduce" loop of extent 4, agreeing exactly
with the Sum step's own already-baked-in reduceElementCount (confirmed)

=== Test 3: the loop nest's own total iteration count survives fusion unchanged ===

unfused Sum node's own loop nest: [reduce:4], 4 total iterations
fused   node's own loop nest:     [reduce:4], 4 total iterations
self-check: fusion changed HOW s is computed but not how many total iterations its own
reduction needs -- the loop nest itself is invariant across the fusion boundary (confirmed)
```

!!! note "Why buildLoopNest() never needs a new field on Node"
    Every number `buildLoopNest()` reads already existed somewhere in the IR before this section: the output `Shape` Chapter 6's `inferShapes()` computes, and the `reduceElementCount` Chapter 14's own `reductionFusionPass()` already bakes into a `Sum` step. That is a deliberate design choice, not a coincidence -- a `LoopNest` is a VIEW of the graph's own existing shape information, recomputed on demand, rather than a new piece of state the graph has to keep in sync as passes run. The same reasoning Chapter 8 used to justify `Graph` having no in-place mutation API (a stale derived field is a bug waiting to happen) applies here too.

## 15.2 Tiling a Loop: Splitting One Loop Into Two, Correctly at the Boundary

### Intuition

A `LoopNest` says how many total iterations a computation needs; it says nothing about how much of a tensor's own data one PASS through the hardware should touch before moving on -- the real, practical question behind cache blocking and shared-memory staging, and one of the exact knobs Part 5's own autotuner will eventually search over rather than hand-pick. Tiling is the mechanical operation underneath that knob: take one loop of extent N, and run it instead as an OUTER loop over "which tile" and an INNER loop over "which element within that tile." When the tile size happens to divide N evenly, this is nothing more than bookkeeping. It almost never divides evenly in practice, and that is where a careless implementation goes wrong: the LAST tile is shorter than every tile before it, and a tiling scheme that doesn't account for that either silently drops real elements or reads past the end of real memory -- not a diagram-level nicety, but the exact kind of off-by-one a real compiler cannot afford to ship.

### Background

`tileLoop()` takes a `Loop` and a chosen tile size and returns a `TiledLoop` carrying four numbers, not two: the original extent, the tile size, and the resulting OUTER extent (`ceil(originalExtent / tileSize)`, computed as integer ceiling division so a partial final tile still gets counted). Deliberately absent from that struct is any single "inner extent" field, because there isn't one single correct value -- every tile except possibly the last covers exactly `tileSize` elements, and the last one covers whatever remains. `actualInnerExtent()` is the one function this whole section exists to get right: given a `TiledLoop` and an outer index, it returns `min(tileSize, originalExtent - outerIndex * tileSize)` -- `tileSize` for every full tile, and the true, smaller remainder for a partial one, with no separate "is this the last tile" branch needed anywhere that calls it. This section proves the tiled and untiled forms visit the exact same set of indices the same way every correctness claim in this book gets proved: by literally enumerating both index sets and comparing them, across three real cases -- an evenly-dividing tile size, a tile size that leaves a genuine one-element boundary tile, and a tile size larger than the whole loop -- before applying the same operation to one loop inside a real, two-dimensional `LoopNest` from Section 15.1, showing tiling turns a k-loop nest into a (k+1)-loop nest while visiting the exact same set of index PAIRS.

```text
TILING A LOOP OF EXTENT 13 BY A TILE SIZE OF 4 (the boundary case):

  UNTILED:  one loop, 13 iterations
    i = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

  TILED (outer, inner):  outerExtent = ceil(13 / 4) = 4

    outer=0: inner 0..3   -> i = 0, 1, 2, 3       (full tile,    4 elements)
    outer=1: inner 0..3   -> i = 4, 5, 6, 7       (full tile,    4 elements)
    outer=2: inner 0..3   -> i = 8, 9, 10, 11     (full tile,    4 elements)
    outer=3: inner 0..0   -> i = 12               (PARTIAL tile, 1 element)

  same 13 indices visited either way -- actualInnerExtent() is exactly
  what keeps that last, partial tile from reading past index 12
```

```cpp
// Chapter 15: Loop Fusion and Tiling
// 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp
//
// Section 15.2 -- a LoopNest (Section 15.1) tells CUDA Hammer how many
// iterations a fused node's own generated loop needs. It says nothing yet
// about how much of a tensor's own data one PASS through that loop should
// touch at a time -- the real, practical question behind cache blocking,
// shared-memory staging, and thread-block sizing, all of which Part 4's own
// codegen chapters and Part 5's own autotuner will eventually have to
// answer for real. This section builds the piece those later chapters will
// need: TILING, splitting one loop of extent N into an OUTER loop (which
// tile) and an INNER loop (which element within that tile), given a chosen
// tile size.
//
// The one genuine subtlety, and the one this section spends most of its
// own code proving rather than asserting: tiling is only a clean, even
// split when the tile size happens to divide the original extent evenly.
// It usually doesn't. The LAST tile is then partial -- shorter than every
// tile before it -- and a tiling scheme that doesn't account for that
// either visits too FEW indices (silently dropping real tensor elements)
// or too MANY (reading or writing past the end of real memory). Getting
// this right is not a diagram-level nicety; it is the exact kind of
// off-by-one a real compiler cannot afford to get wrong.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp -o 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary
// Run:     ./034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>

// ==================== Loop / LoopNest (from Section 15.1, unchanged) ====================

struct Loop {
    std::string dimName;
    long long extent;
};
struct LoopNest {
    std::vector<Loop> loops;
    long long totalIterations() const {
        long long total = 1;
        for (const Loop& l : loops) total *= l.extent;
        return total;
    }
};
static std::string loopNestStr(const LoopNest& nest) {
    std::string out = "[";
    for (size_t i = 0; i < nest.loops.size(); ++i) {
        if (i) out += ", ";
        out += nest.loops[i].dimName + ":" + std::to_string(nest.loops[i].extent);
    }
    out += "]";
    return out;
}

// ==================== Section 15.2: TiledLoop / tileLoop() ====================
//
// A TiledLoop keeps FOUR numbers, not two, precisely because "the inner
// loop's own extent" is not one fixed number once the boundary tile is
// accounted for -- it is tileSize for every tile except (possibly) the
// last one. Storing originalExtent alongside tileSize lets
// actualInnerExtent() compute the true extent for ANY outer index,
// including that last, possibly-shorter tile, without ever needing a
// special-cased "if this is the last tile" branch at the call site.
struct TiledLoop {
    std::string dimName;
    long long originalExtent;
    long long tileSize;
    long long outerExtent;
};

static TiledLoop tileLoop(const Loop& loop, long long tileSize) {
    if (tileSize <= 0) throw std::runtime_error("tileLoop: tile size must be positive");
    // Ceiling division -- the number of tiles needed to cover originalExtent
    // elements at tileSize elements per tile, rounding UP: a partial final
    // tile still needs a full extra outer iteration to reach it at all.
    long long outerExtent = (loop.extent + tileSize - 1) / tileSize;
    return TiledLoop{loop.dimName, loop.extent, tileSize, outerExtent};
}

// The one function this whole section exists to get right: how many
// elements does the INNER loop actually cover for a given outer index?
// Every tile except the last covers exactly tileSize elements. The last
// tile covers whatever remains -- originalExtent minus everything the
// tiles before it already covered -- which is tileSize only when
// tileSize evenly divides originalExtent, and strictly less otherwise.
static long long actualInnerExtent(const TiledLoop& tl, long long outerIndex) {
    long long alreadyCovered = outerIndex * tl.tileSize;
    long long remaining = tl.originalExtent - alreadyCovered;
    return std::min(tl.tileSize, remaining);
}

// ==================== Correctness proof: enumerate, don't assert ====================
//
// The book's own standing discipline (every chapter since Part 2) is to
// PROVE an equivalence by actually executing both sides and comparing real
// results, not by inspecting the formulas and declaring them equivalent.
// Here, that means: the SET of indices a tiled, nested pair of loops
// actually visits must be exactly the set the single untiled loop visits --
// no index missing, no index visited twice, nothing visited past the end.
static std::set<long long> enumerateUntiled(const Loop& loop) {
    std::set<long long> visited;
    for (long long i = 0; i < loop.extent; ++i) visited.insert(i);
    return visited;
}
static std::set<long long> enumerateTiled(const TiledLoop& tl) {
    std::set<long long> visited;
    for (long long outer = 0; outer < tl.outerExtent; ++outer) {
        long long innerExtent = actualInnerExtent(tl, outer);
        for (long long inner = 0; inner < innerExtent; ++inner) {
            visited.insert(outer * tl.tileSize + inner);
        }
    }
    return visited;
}

int main() {
    printf("=== Section 15.2: tileLoop() -- splitting one loop into two, correctly at the boundary ===\n\n");

    // ---- Case 1: tile size divides the extent evenly ----
    // Hand-derivation: extent=12, tileSize=4 -> outerExtent = ceil(12/4) = 3,
    // three FULL tiles of 4 elements each (4, 4, 4), no boundary tile at all.
    {
        Loop loop{"i", 12};
        TiledLoop tiled = tileLoop(loop, 4);
        printf("Case 1 (even split): extent=%lld, tileSize=%lld -> outerExtent=%lld\n",
               loop.extent, tiled.tileSize, tiled.outerExtent);
        bool outerOk = (tiled.outerExtent == 3);
        long long e0 = actualInnerExtent(tiled, 0), e1 = actualInnerExtent(tiled, 1), e2 = actualInnerExtent(tiled, 2);
        printf("  tile sizes actually visited: %lld, %lld, %lld\n", e0, e1, e2);
        bool tileSizesOk = (e0 == 4 && e1 == 4 && e2 == 4);
        std::set<long long> untiled = enumerateUntiled(loop);
        std::set<long long> tiledSet = enumerateTiled(tiled);
        bool setsEqual = (untiled == tiledSet);
        printf("  self-check: outerExtent=3, every tile full (4,4,4), tiled/untiled index sets equal (%s)\n",
               (outerOk && tileSizesOk && setsEqual) ? "confirmed" : "MISMATCH");
        if (!(outerOk && tileSizesOk && setsEqual)) return 1;
    }

    // ---- Case 2: tile size does NOT divide the extent evenly ----
    // Hand-derivation: extent=13, tileSize=4 -> outerExtent = ceil(13/4) = 4.
    // Three full tiles (4, 4, 4) cover indices 0..11 -- 12 elements -- and
    // the FOURTH tile covers only what's left: 13 - 12 = 1 element. A
    // tiling scheme that used a fixed inner extent of 4 for every outer
    // index would read index 12, 13, 14 -- three indices past the real
    // data. actualInnerExtent() is exactly the function that prevents that.
    {
        Loop loop{"i", 13};
        TiledLoop tiled = tileLoop(loop, 4);
        printf("\nCase 2 (uneven split): extent=%lld, tileSize=%lld -> outerExtent=%lld\n",
               loop.extent, tiled.tileSize, tiled.outerExtent);
        bool outerOk = (tiled.outerExtent == 4);
        long long e0 = actualInnerExtent(tiled, 0), e1 = actualInnerExtent(tiled, 1);
        long long e2 = actualInnerExtent(tiled, 2), e3 = actualInnerExtent(tiled, 3);
        printf("  tile sizes actually visited: %lld, %lld, %lld, %lld  (last tile is PARTIAL)\n", e0, e1, e2, e3);
        bool tileSizesOk = (e0 == 4 && e1 == 4 && e2 == 4 && e3 == 1);
        std::set<long long> untiled = enumerateUntiled(loop);
        std::set<long long> tiledSet = enumerateTiled(tiled);
        bool setsEqual = (untiled == tiledSet);
        printf("  self-check: outerExtent=4, last tile is 1 element (not 4), tiled/untiled index sets\n");
        printf("  equal -- no index missing, none visited past the end (%s)\n",
               (outerOk && tileSizesOk && setsEqual) ? "confirmed" : "MISMATCH");
        if (!(outerOk && tileSizesOk && setsEqual)) return 1;
    }

    // ---- Case 3: tile size LARGER than the whole extent ----
    // Hand-derivation: extent=4, tileSize=8 -> outerExtent = ceil(4/8) = 1,
    // a single tile whose own actual extent is min(8, 4) = 4, not 8: the
    // whole loop fits inside one (partially-empty) tile.
    {
        Loop loop{"i", 4};
        TiledLoop tiled = tileLoop(loop, 8);
        printf("\nCase 3 (tile larger than extent): extent=%lld, tileSize=%lld -> outerExtent=%lld\n",
               loop.extent, tiled.tileSize, tiled.outerExtent);
        bool outerOk = (tiled.outerExtent == 1);
        long long e0 = actualInnerExtent(tiled, 0);
        printf("  tile size actually visited: %lld (not %lld)\n", e0, tiled.tileSize);
        bool tileSizeOk = (e0 == 4);
        std::set<long long> untiled = enumerateUntiled(loop);
        std::set<long long> tiledSet = enumerateTiled(tiled);
        bool setsEqual = (untiled == tiledSet);
        printf("  self-check: one tile, its own actual extent is 4 (the whole loop), sets equal (%s)\n",
               (outerOk && tileSizeOk && setsEqual) ? "confirmed" : "MISMATCH");
        if (!(outerOk && tileSizeOk && setsEqual)) return 1;
    }

    // ---- Case 4: applying tiling to a REAL, multi-dimensional LoopNest ----
    // Section 15.1's own diamond-graph loop nest was [dim0:3, dim1:4],
    // 12 total iterations. Tile dim1 (extent 4) by tileSize=2 -- an EVENLY
    // dividing choice on purpose, so this case can focus on what tiling
    // does to a loop NEST's own shape rather than repeating the boundary-
    // tile arithmetic Cases 1-3 already covered. Tiling turns a 2-loop nest
    // into a 3-loop nest (dim0, dim1_outer, dim1_inner) while visiting the
    // exact same set of (dim0, dim1) POINTS -- proven the same way, by
    // enumerating both index sets directly rather than trusting the
    // arithmetic.
    printf("\n=== Case 4: tiling one loop inside a real 2-D LoopNest ===\n\n");
    LoopNest original{{Loop{"dim0", 3}, Loop{"dim1", 4}}};
    Loop dim1 = original.loops[1];
    TiledLoop tiledDim1 = tileLoop(dim1, 2);  // 4 / 2 = 2, evenly -- outerExtent=2, tileSize=2

    LoopNest tiledNest;
    tiledNest.loops.push_back(original.loops[0]);                                    // dim0:3, untouched
    tiledNest.loops.push_back(Loop{"dim1_outer", tiledDim1.outerExtent});             // dim1_outer:2
    tiledNest.loops.push_back(Loop{"dim1_inner", tiledDim1.tileSize});                // dim1_inner:2

    printf("original loop nest: %s, %lld total iterations\n", loopNestStr(original).c_str(), original.totalIterations());
    printf("tiled loop nest:     %s, %lld total iterations\n", loopNestStr(tiledNest).c_str(), tiledNest.totalIterations());

    bool sameTotal = (original.totalIterations() == tiledNest.totalIterations());
    bool oneMoreLoop = (tiledNest.loops.size() == original.loops.size() + 1);
    printf("self-check: tiling turned a 2-loop nest into a 3-loop nest, same 12 total iterations (%s)\n",
           (sameTotal && oneMoreLoop) ? "confirmed" : "MISMATCH");

    std::set<std::pair<long long, long long>> untiledPoints;
    for (long long i = 0; i < original.loops[0].extent; ++i)
        for (long long j = 0; j < original.loops[1].extent; ++j)
            untiledPoints.insert({i, j});

    std::set<std::pair<long long, long long>> tiledPoints;
    for (long long i = 0; i < tiledNest.loops[0].extent; ++i)
        for (long long jo = 0; jo < tiledNest.loops[1].extent; ++jo)
            for (long long ji = 0; ji < tiledNest.loops[2].extent; ++ji)
                tiledPoints.insert({i, jo * tiledDim1.tileSize + ji});

    bool pointsEqual = (untiledPoints == tiledPoints);
    printf("self-check: the (dim0, dim1) index pairs visited by the tiled 3-loop form exactly match\n");
    printf("the pairs visited by the original 2-loop form -- 12 points, none missing, none repeated (%s)\n",
           pointsEqual ? "confirmed" : "MISMATCH");

    bool allOk = sameTotal && oneMoreLoop && pointsEqual;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp -o 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary
./034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary
```

**Output:**

```text
=== Section 15.2: tileLoop() -- splitting one loop into two, correctly at the boundary ===

Case 1 (even split): extent=12, tileSize=4 -> outerExtent=3
  tile sizes actually visited: 4, 4, 4
  self-check: outerExtent=3, every tile full (4,4,4), tiled/untiled index sets equal (confirmed)

Case 2 (uneven split): extent=13, tileSize=4 -> outerExtent=4
  tile sizes actually visited: 4, 4, 4, 1  (last tile is PARTIAL)
  self-check: outerExtent=4, last tile is 1 element (not 4), tiled/untiled index sets
  equal -- no index missing, none visited past the end (confirmed)

Case 3 (tile larger than extent): extent=4, tileSize=8 -> outerExtent=1
  tile size actually visited: 4 (not 8)
  self-check: one tile, its own actual extent is 4 (the whole loop), sets equal (confirmed)

=== Case 4: tiling one loop inside a real 2-D LoopNest ===

original loop nest: [dim0:3, dim1:4], 12 total iterations
tiled loop nest:     [dim0:3, dim1_outer:2, dim1_inner:2], 12 total iterations
self-check: tiling turned a 2-loop nest into a 3-loop nest, same 12 total iterations (confirmed)
self-check: the (dim0, dim1) index pairs visited by the tiled 3-loop form exactly match
the pairs visited by the original 2-loop form -- 12 points, none missing, none repeated (confirmed)
```

!!! warning "[COMMON TRAP] A fixed inner extent silently reads past the end of real data"
    It is tempting to give `TiledLoop` a single `innerExtent` field, set once to the chosen tile size, and loop `for (inner = 0; inner < innerExtent; ++inner)` unconditionally for every outer index. That code compiles, runs, and produces exactly the right answer -- right up until `originalExtent` isn't a multiple of `tileSize`, at which point the LAST tile silently reads (or writes) indices past the real data: 13 elements tiled by 4 with a fixed inner extent visits index 12, 13, and 14, and only the first of those is real. Nothing about that bug announces itself in a toy example with small, clean numbers; it shows up as a crash, or worse, quietly wrong output, only once real tensor sizes stop being convenient multiples of the tile size chosen. `actualInnerExtent()`'s whole reason to exist is closing that gap structurally, not by remembering to check it at every call site.

## 15.3 Loop Fusion Across Nodes: A Second, Independent Axis From Operator Fusion

### Intuition

Chapters 13 and 14 already answered a hard question correctly: which VALUES are safe to compute inside the same node's own internal program. That question has a real cost when the answer is no -- Chapter 14's own Graph B fused nothing at all, for two independent reasons at once, and the chapter framed that honestly as the correct output for a graph whose structure offered no safe fusion. But "operator fusion says no" and "these two computations' own LOOPS are incompatible" are not the same claim. A node that stays structurally separate because it's SHARED, or because it's a REDUCTION, can still have a loop that happens to run exactly as many times, over exactly the same range, as some other separate node's own loop -- and if so, a later codegen stage could still choose to interleave them into one shared loop, without operator fusion ever needing to agree. Real compilers build exactly this separation on purpose: Chapter 3 already cited TVM's own `fuse` schedule primitive as its own distinct operation, layered on top of whatever earlier decision grouped computations into stages at all.

### Background

`loopNestsCompatibleForFusion()` asks a narrow, exact, deliberately conservative question: do two `LoopNest`s have the same number of loops, and does every loop, IN ORDER, have the same extent? Nothing more sophisticated than that -- the same conservative instinct Chapter 10's own `cseKey()` already established for this book (treating `add(a,b)` and `add(b,a)` as different rather than risking a false merge), applied here to loop shapes instead of expression keys. Run on Chapter 14's own Graph B, it produces a genuinely interesting result: `t1` (a plain `ReLU` node, kept separate because it's shared between `s` and `y2`) has loop nest `[dim0:4]`, and `s` (a `FusedReduction` node, kept separate because reductions are never inlined) has loop nest `[reduce:4]` -- same loop count, same extent, compatible. Operator fusion refused to merge these two nodes' own FusedStep programs, for two entirely separate and entirely correct reasons. Loop-nest compatibility doesn't ask either of those questions again; it only asks whether the two nodes' own iteration spaces could still run as one shared loop at codegen time, and here the honest answer is yes -- a codegen stage could compute `t1[i]`, fold it directly into `s`'s own running sum for that same `i`, and still leave it available for `y2`'s own separate later use, all inside one physical loop, with neither Chapter 13's nor Chapter 14's own fusion rule ever needing to say yes. The check is also honestly limited on purpose: it compares extents IN ORDER, so a genuinely equivalent but permuted loop nest -- `[3, 4]` against `[4, 3]` -- is reported incompatible, a real, stated limitation left open rather than silently patched, exactly the kind of honesty Chapter 10's own commutativity discussion already modeled for this book.

```text
GRAPH B (Chapter 14's own example) -- OPERATOR FUSION already said no, twice:

  x            w
  |            |
  ReLU (t1) ---+------ 2 consumers: s AND y2 -- SHARED, stays plain (Ch13's rule)
  |            |
  Sum (s)      |
  |            |    -- 1 consumer: y -- REDUCTION, stays external (Ch14's rule)
  Add (y)   Mul (y2)

  t1's own loop nest:  [dim0:4]      -- one loop, extent 4
  s's own loop nest:   [reduce:4]    -- one loop, extent 4

  loopNestsCompatibleForFusion(t1, s) -> YES: same loop count, same extent

  operator fusion (Ch13-14) and loop-nest compatibility (this section)
  are two DIFFERENT questions about the SAME pair of nodes -- and here
  they disagree, on purpose
```

```cpp
// Chapter 15: Loop Fusion and Tiling
// 035_loop_fusion_across_nodes_a_second_independent_axis_from_operator_fusion.cpp
//
// Section 15.3 -- Chapters 13 and 14 answered one question: which VALUES
// can share a single node's own internal FusedStep program. Sections 15.1
// and 15.2 gave every computed node an explicit LoopNest, independent of
// whether that node happens to be fused or plain. This section asks a
// DIFFERENT question, at a different level: even when two nodes remain
// SEPARATE in the graph -- because operator fusion said no -- could their
// own loops still be merged into one shared loop nest once real code gets
// generated? Real compilers keep these two questions apart on purpose:
// Chapter 3 already cited TVM's own schedule primitives, where `fuse` is a
// SEPARATE primitive from whatever decided two computations belong in the
// same stage at all. This section builds CUDA Hammer's own version of that
// separation: loopNestsCompatibleForFusion(), a conservative, EXACT check
// answering only "could these two loops run as one," decoupled entirely
// from Chapters 13-14's own "should these two VALUES share a program."
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 035_loop_fusion_across_nodes_a_second_independent_axis_from_operator_fusion.cpp -o 035_loop_fusion_across_nodes_a_second_independent_axis_from_operator_fusion
// Run:     ./035_loop_fusion_across_nodes_a_second_independent_axis_from_operator_fusion
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand {
    OperandKind kind;
    int index;
};

struct FusedStep {
    OpKind op;
    std::vector<FusedOperand> operands;
    long long reduceElementCount = 1;
};

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;
    float constValue = 0.0f;
    std::vector<FusedStep> fusedSteps;
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
    Value addFusedElementwise(std::vector<Value> externalInputs, std::vector<FusedStep> steps,
                               const std::string& name) {
        Value out = addNode(OpKind::FusedElementwise, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
    }
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps,
                             const std::string& name) {
        Value out = addNode(OpKind::FusedReduction, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
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

// ==================== Shape / broadcastShapes / inferShapes (from Chapters 6 and 14, unchanged) ====================

struct Shape { std::vector<int> dims; };

static std::string shapeStr(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) { if (i) out += ", "; out += std::to_string(s.dims[i]); }
    out += "]";
    return out;
}
static long long numElements(const Shape& s) {
    long long n = 1;
    for (int d : s.dims) n *= d;
    return n;
}
static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        int outDim;
        if (da == db) outDim = da;
        else if (da == 1) outDim = db;
        else if (db == 1) outDim = da;
        else throw std::runtime_error(context + ": shapes " + shapeStr(a) + " and " + shapeStr(b) + " incompatible");
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
            shapes[id] = declaredShapes.at(id);
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        } else if (n->op == OpKind::ReLU) {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        } else {  // Sum -- inferShapes() only ever runs before fusion
            shapes[id] = Shape{};
        }
    }
    return shapes;
}

// ==================== resolveIntoGroup / elementwiseFusionPass / reductionFusionPass (from Chapter 14, unchanged) ====================

struct FusionResult {
    Graph graph;
    std::map<int, int> representativeOldId;
};

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized,
                                      std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId,
                                      std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) {
        return FusedOperand{OperandKind::PriorStep, stepIt->second};
    }
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
                           (consumers.at(oldId) != 1);
    if (mustBeExternal) {
        auto extIt = externalIndexByOldId.find(oldId);
        if (extIt != externalIndexByOldId.end()) {
            return FusedOperand{OperandKind::ExternalInput, extIt->second};
        }
        int idx = static_cast<int>(externalInputs.size());
        externalInputs.push_back(materialized.at(oldId));
        externalIndexByOldId[oldId] = idx;
        return FusedOperand{OperandKind::ExternalInput, idx};
    }
    std::vector<FusedOperand> operands;
    for (const Value& in : p->inputs) {
        operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                             externalIndexByOldId, steps, stepIndexByOldId));
    }
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}

static FusionResult elementwiseFusionPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("elementwiseFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    FusionResult result;
    Graph& out = result.graph;
    std::map<int, Value> materialized;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        if (n->op == OpKind::Input) {
            Value v = out.addInput(n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Const) {
            Value v = out.addConst(n->constValue, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
        }
        steps.push_back(FusedStep{n->op, operands});
        Value v;
        if (steps.size() == 1) {
            if (n->op == OpKind::ReLU) v = out.addUnary(OpKind::ReLU, externalInputs[0], n->debugName);
            else v = out.addBinary(n->op, externalInputs[0], externalInputs[1], n->debugName);
        } else {
            v = out.addFusedElementwise(externalInputs, steps, n->debugName);
        }
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

static FusionResult reductionFusionPass(const Graph& g, const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("reductionFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    FusionResult result;
    Graph& out = result.graph;
    std::map<int, Value> materialized;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        if (n->op == OpKind::Input) {
            Value v = out.addInput(n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Const) {
            Value v = out.addConst(n->constValue, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Sum) {
            std::vector<Value> externalInputs;
            std::map<int, int> externalIndexByOldId;
            std::vector<FusedStep> steps;
            std::map<int, int> stepIndexByOldId;
            FusedOperand summedOperand = resolveIntoGroup(n->inputs[0].nodeId, g, consumers, materialized,
                                                            externalInputs, externalIndexByOldId,
                                                            steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
        }
        steps.push_back(FusedStep{n->op, operands});
        Value v;
        if (steps.size() == 1) {
            if (n->op == OpKind::ReLU) v = out.addUnary(OpKind::ReLU, externalInputs[0], n->debugName);
            else v = out.addBinary(n->op, externalInputs[0], externalInputs[1], n->debugName);
        } else {
            v = out.addFusedElementwise(externalInputs, steps, n->debugName);
        }
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

// ==================== Loop / LoopNest / buildLoopNest (from Section 15.1, unchanged) ====================

struct Loop {
    std::string dimName;
    long long extent;
};
struct LoopNest {
    std::vector<Loop> loops;
    long long totalIterations() const {
        long long total = 1;
        for (const Loop& l : loops) total *= l.extent;
        return total;
    }
};
static std::string loopNestStr(const LoopNest& nest) {
    std::string out = "[";
    for (size_t i = 0; i < nest.loops.size(); ++i) {
        if (i) out += ", ";
        out += nest.loops[i].dimName + ":" + std::to_string(nest.loops[i].extent);
    }
    out += "]";
    return out;
}
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    if (n->op == OpKind::Input || n->op == OpKind::Const) {
        throw std::runtime_error("buildLoopNest: " + n->debugName +
                                  " is materialized from memory or a literal, not computed by a loop");
    }
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) {
        nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    }
    if (n->op == OpKind::Sum) {
        nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    } else if (n->op == OpKind::FusedReduction) {
        long long reduceExtent = n->fusedSteps.back().reduceElementCount;
        nest.loops.push_back(Loop{"reduce", reduceExtent});
    }
    return nest;
}

// ==================== Section 15.3: loopNestsCompatibleForFusion() ====================
//
// Deliberately conservative and EXACT, the same honesty this book's other
// "compatible/matches" checks have always used (Chapter 10's own cseKey(),
// which does not attempt to recognize add(a,b) and add(b,a) as the same
// expression, is the closest precedent): two loop nests are compatible for
// fusion here ONLY when they have the same number of loops AND every loop,
// IN ORDER, has the same extent. A permuted-but-equal nest (e.g. [3,4] vs.
// [4,3]) is NOT recognized as fusable by this check -- a real, stated
// limitation, left open rather than silently patched. A smarter version
// could sort or canonicalize loop order first; this one does not.
static bool loopNestsCompatibleForFusion(const LoopNest& a, const LoopNest& b) {
    if (a.loops.size() != b.loops.size()) return false;
    for (size_t i = 0; i < a.loops.size(); ++i) {
        if (a.loops[i].extent != b.loops[i].extent) return false;
    }
    return true;
}

int main() {
    printf("=== Section 15.3: loopNestsCompatibleForFusion() -- a second, independent axis ===\n\n");

    // ---- Test A: Chapter 14's own Graph B, rebuilt verbatim ----
    // x=Input([4]), w=Input(scalar), t1=ReLU(x), s=Sum(t1), y=Add(s,w),
    // y2=Mul(t1,w). t1 has TWO consumers (s and y2) -- stays plain via
    // Chapter 13's own sharing rule. s has ONE consumer (y) but is a
    // reduction -- stays its own (trivial, 1-step) FusedReduction via
    // Chapter 14's own new rule. Chapter 14 already showed reductionFusionPass()
    // fuses NOTHING here: 6 nodes in, 6 nodes out. The question THIS section
    // asks is different: even though t1 and s remain separate NODES, are
    // their own LOOPS still compatible enough to merge at codegen time?
    printf("--- Test A: Chapter 14's Graph B -- operator fusion says no, does loop fusion agree? ---\n\n");

    Graph gB;
    Value x  = gB.addInput("x");
    Value w  = gB.addInput("w");
    Value t1 = gB.addUnary(OpKind::ReLU, x, "t1");
    Value s  = gB.addUnary(OpKind::Sum, t1, "s");
    Value y  = gB.addBinary(OpKind::Add, s, w, "y");
    Value y2 = gB.addBinary(OpKind::Mul, t1, w, "y2");
    (void)y; (void)y2;

    std::map<int, Shape> declaredB = {{x.nodeId, Shape{{4}}}, {w.nodeId, Shape{}}};
    std::map<int, Shape> shapesB = inferShapes(gB, declaredB);
    std::map<int, long long> elementCountsB;
    for (const auto& kv : shapesB) elementCountsB[kv.first] = numElements(kv.second);

    FusionResult resultB = reductionFusionPass(gB, elementCountsB);
    bool nothingFused = (resultB.graph.size() == 6 && gB.size() == 6);
    printf("self-check: reductionFusionPass() on Graph B fuses nothing, 6 -> 6 nodes, exactly as\n");
    printf("Chapter 14 Section 14.2 found (%s)\n", nothingFused ? "confirmed" : "MISMATCH");

    const Node* plainT1 = nullptr;
    const Node* fusedS = nullptr;
    for (const auto& n : resultB.graph.nodes()) {
        if (n->debugName == "t1") plainT1 = n.get();
        if (n->debugName == "s") fusedS = n.get();
    }
    LoopNest t1Nest = buildLoopNest(plainT1, {{plainT1->id, shapesB.at(t1.nodeId)}}, {});
    LoopNest sNest = buildLoopNest(fusedS, {{fusedS->id, shapesB.at(s.nodeId)}}, {});

    printf("\nt1's own loop nest (plain ReLU node): %s\n", loopNestStr(t1Nest).c_str());
    printf("s's own loop nest (FusedReduction node): %s\n", loopNestStr(sNest).c_str());

    bool compatibleAB = loopNestsCompatibleForFusion(t1Nest, sNest);
    printf("\nself-check: t1's [dim0:4] and s's [reduce:4] have the same loop count (1) and the\n");
    printf("same extent (4) -- loopNestsCompatibleForFusion() says YES (%s), even though\n",
           compatibleAB ? "confirmed" : "MISMATCH");
    printf("operator fusion (Chapters 13-14) already said no to merging their own step programs.\n");
    printf("This is the whole point of this section: a codegen stage could still choose to run\n");
    printf("t1's own loop and s's own reduction loop as ONE shared loop (computing t1[i], folding\n");
    printf("it into s's running sum, AND leaving it available for y2's own separate later use)\n");
    printf("without operator fusion ever having to say yes -- the same separation TVM's own real\n");
    printf("`fuse` schedule primitive (Chapter 3) keeps from its own op-fusion decisions.\n");

    // ---- Test B: two genuinely incompatible loop nests (different loop counts) ----
    printf("\n--- Test B: incompatible loop COUNTS (a 2-D nest vs. a 1-D nest) ---\n\n");

    Graph diamond;
    Value da  = diamond.addInput("a");
    Value db  = diamond.addInput("b");
    Value dt1 = diamond.addBinary(OpKind::Add, da, db, "t1");
    Value dt2 = diamond.addBinary(OpKind::Mul, dt1, da, "t2");
    Value dt3 = diamond.addUnary(OpKind::ReLU, dt1, "t3");
    Value dout = diamond.addBinary(OpKind::Add, dt2, dt3, "out");

    std::map<int, Shape> declaredD = {{da.nodeId, Shape{{3, 4}}}, {db.nodeId, Shape{{4}}}};
    std::map<int, Shape> shapesD = inferShapes(diamond, declaredD);
    FusionResult diamondFused = elementwiseFusionPass(diamond);
    const Node* diamondFusedNode = nullptr;
    for (const auto& n : diamondFused.graph.nodes()) {
        if (n->op == OpKind::FusedElementwise) diamondFusedNode = n.get();
    }
    LoopNest diamondNest = buildLoopNest(diamondFusedNode, {{diamondFusedNode->id, shapesD.at(dout.nodeId)}}, {});

    printf("diamond's fused node loop nest (shape [3,4]): %s\n", loopNestStr(diamondNest).c_str());
    printf("s's own loop nest (from Test A, shape [] + reduce:4): %s\n", loopNestStr(sNest).c_str());

    bool compatibleDiamondS = loopNestsCompatibleForFusion(diamondNest, sNest);
    printf("\nself-check: 2 loops vs. 1 loop -- loopNestsCompatibleForFusion() correctly says NO (%s)\n",
           !compatibleDiamondS ? "confirmed" : "MISMATCH");

    // ---- Test C: same loop COUNT, different extent ----
    printf("\n--- Test C: same loop count, different extent (a real Input of shape [5]) ---\n\n");

    Graph gC;
    Value z  = gC.addInput("z");
    Value ct1 = gC.addUnary(OpKind::ReLU, z, "t1");
    (void)ct1;
    std::map<int, Shape> declaredC = {{z.nodeId, Shape{{5}}}};
    std::map<int, Shape> shapesC = inferShapes(gC, declaredC);
    const Node* plainCt1 = gC.node(ct1.nodeId);
    LoopNest ct1Nest = buildLoopNest(plainCt1, {{plainCt1->id, shapesC.at(ct1.nodeId)}}, {});

    printf("t1's own loop nest (this section's Test A, shape [4]): %s\n", loopNestStr(t1Nest).c_str());
    printf("a NEW node's own loop nest (shape [5]):                %s\n", loopNestStr(ct1Nest).c_str());

    bool compatibleDifferentExtent = loopNestsCompatibleForFusion(t1Nest, ct1Nest);
    printf("\nself-check: same loop count (1), different extent (4 vs. 5) -- correctly says NO (%s)\n",
           !compatibleDifferentExtent ? "confirmed" : "MISMATCH");

    // ---- Stated limitation: a permuted-but-equal loop nest is NOT recognized ----
    printf("\n--- A stated limitation: order matters, even when the SET of extents is identical ---\n\n");

    LoopNest nestP{{Loop{"dim0", 3}, Loop{"dim1", 4}}};
    LoopNest nestQ{{Loop{"dim0", 4}, Loop{"dim1", 3}}};
    bool compatiblePermuted = loopNestsCompatibleForFusion(nestP, nestQ);
    printf("nest P: %s\nnest Q: %s\n", loopNestStr(nestP).c_str(), loopNestStr(nestQ).c_str());
    printf("\nself-check: same two extents (3 and 4), different ORDER -- loopNestsCompatibleForFusion()\n");
    printf("says NO (%s), the same conservative choice Chapter 10's own cseKey() made for\n",
           !compatiblePermuted ? "confirmed" : "MISMATCH");
    printf("add(a,b) vs. add(b,a): a real, honestly-stated limitation, not silently patched here.\n");

    bool allOk = nothingFused && compatibleAB && !compatibleDiamondS && !compatibleDifferentExtent && !compatiblePermuted;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 035_loop_fusion_across_nodes_a_second_independent_axis_from_operator_fusion.cpp -o 035_loop_fusion_across_nodes_a_second_independent_axis_from_operator_fusion
./035_loop_fusion_across_nodes_a_second_independent_axis_from_operator_fusion
```

**Output:**

```text
=== Section 15.3: loopNestsCompatibleForFusion() -- a second, independent axis ===

--- Test A: Chapter 14's Graph B -- operator fusion says no, does loop fusion agree? ---

self-check: reductionFusionPass() on Graph B fuses nothing, 6 -> 6 nodes, exactly as
Chapter 14 Section 14.2 found (confirmed)

t1's own loop nest (plain ReLU node): [dim0:4]
s's own loop nest (FusedReduction node): [reduce:4]

self-check: t1's [dim0:4] and s's [reduce:4] have the same loop count (1) and the
same extent (4) -- loopNestsCompatibleForFusion() says YES (confirmed), even though
operator fusion (Chapters 13-14) already said no to merging their own step programs.
This is the whole point of this section: a codegen stage could still choose to run
t1's own loop and s's own reduction loop as ONE shared loop (computing t1[i], folding
it into s's running sum, AND leaving it available for y2's own separate later use)
without operator fusion ever having to say yes -- the same separation TVM's own real
`fuse` schedule primitive (Chapter 3) keeps from its own op-fusion decisions.

--- Test B: incompatible loop COUNTS (a 2-D nest vs. a 1-D nest) ---

diamond's fused node loop nest (shape [3,4]): [dim0:3, dim1:4]
s's own loop nest (from Test A, shape [] + reduce:4): [reduce:4]

self-check: 2 loops vs. 1 loop -- loopNestsCompatibleForFusion() correctly says NO (confirmed)

--- Test C: same loop count, different extent (a real Input of shape [5]) ---

t1's own loop nest (this section's Test A, shape [4]): [dim0:4]
a NEW node's own loop nest (shape [5]):                [dim0:5]

self-check: same loop count (1), different extent (4 vs. 5) -- correctly says NO (confirmed)

--- A stated limitation: order matters, even when the SET of extents is identical ---

nest P: [dim0:3, dim1:4]
nest Q: [dim0:4, dim1:3]

self-check: same two extents (3 and 4), different ORDER -- loopNestsCompatibleForFusion()
says NO (confirmed), the same conservative choice Chapter 10's own cseKey() made for
add(a,b) vs. add(b,a): a real, honestly-stated limitation, not silently patched here.
```

!!! note "Two axes, three answers -- and where CUDA Hammer stands now"
    This chapter's own three sections add up to a genuinely new capability, not three unrelated tricks: CUDA Hammer can now look at any computed node and say how many iterations it needs (15.1), split any one of those iterations into a tile-sized pair (15.2), and ask whether two SEPARATE nodes' own loops could still run as one (15.3) -- all without touching a single line of Chapters 13-14's own fusion machinery. Nothing in this chapter changes WHAT any node computes; every idea here is a new lens on top of an unchanged IR. Operator fusion (Chapters 13-14) decides what a node's own internal program looks like. Loop nests (this chapter) give every computed node real iteration bounds. Loop-nest compatibility (Section 15.3) is a separate, later question about whether two different nodes' own loops could still be merged at codegen time. Part 4 is where all of this finally turns into real generated code; Part 5 is where choices like tile size get searched by a real autotuner rather than hand-picked the way this chapter's own examples picked them.

## Chapter Summary

This chapter gave CUDA Hammer its first explicit representation of actual iteration, built entirely from information already sitting in the IR rather than any new field on `Node`. Section 15.1 introduced `LoopNest`, computed by `buildLoopNest()` from a node's own inferred output `Shape` (one loop per dimension, the same per-element correspondence Chapter 13's own `FusedStep` model already relies on) plus, for a `FusedReduction` node, one additional loop whose extent comes straight from Chapter 14's own already-baked-in `reduceElementCount` -- and proved that total iteration count survives the fusion boundary unchanged, even though HOW a value gets computed does not. Section 15.2 built `tileLoop()` and `actualInnerExtent()`, splitting one loop into an outer/inner pair while correctly handling the boundary tile that doesn't divide evenly -- proved, across three cases and one real two-dimensional `LoopNest`, by literally enumerating the indices both forms visit and comparing the sets directly rather than trusting the arithmetic. Section 15.3 closed the chapter with `loopNestsCompatibleForFusion()`, a conservative, exact check answering a genuinely different question from Chapters 13-14's own operator-fusion rules -- proved on Chapter 14's own Graph B, where two nodes kept separate for two entirely different reasons turn out to have perfectly compatible loop nests anyway, the real, practical gap between "these values can't share a program" and "these loops can't share a nest" that real schedule-based compilers like TVM keep as two separate decisions on purpose.

## Self-Check Questions

1. `buildLoopNest()` never adds a new field to `Node` or `FusedStep`. What TWO pieces of already-existing IR information does it read, and for which node kinds does it read each one?
2. Why does a `FusedElementwise` node's own loop nest come entirely from its output `Shape`, while a `FusedReduction` node's own loop nest needs one loop its output `Shape` can never supply?
3. Section 15.1's Test 3 builds a loop nest for a plain, not-yet-fused `OpKind::Sum` node and compares it against the fused version's own loop nest. What does that comparison prove, and why does it matter that the two totals come out identical?
4. `TiledLoop` stores `originalExtent` and `tileSize` separately rather than a single precomputed `innerExtent`. What real bug does storing a single fixed `innerExtent` risk, and under what specific condition does that bug actually manifest?
5. Section 15.2's Case 4 tiles one loop inside a real two-dimensional `LoopNest` rather than a bare `Loop`. What changes about the loop nest's own STRUCTURE (not just its numbers), and what stays exactly the same?
6. `loopNestsCompatibleForFusion()` answers a genuinely different question from Chapters 13-14's own fusion rules. Using Section 15.3's own Graph B example, explain concretely what each kind of check says about the SAME pair of nodes, and why they can disagree without either one being wrong.
7. `loopNestsCompatibleForFusion()` compares loop extents strictly IN ORDER. What real, concrete pair of loop nests does this section show being reported incompatible despite having the exact same set of extents, and why wasn't that gap fixed here?
8. This chapter's own closing note distinguishes three separate ideas: operator fusion, loop nests, and loop-nest compatibility. What does each one decide, and which earlier chapters' own infrastructure does each one build on without modifying it?

## Where We Go Next

This chapter gave every computed node real iteration bounds and a way to reason about splitting or merging loops -- but it never generated a single line of actual executable code, and it never had to decide, for a REAL graph with more than two candidate nodes, exactly which fusion opportunities to take. Chapter 16, "Fusion Boundaries: What Can't Be Fused, and Why," closes out Part 3 by cataloguing the genuine, structural reasons a fusion opportunity has to stop -- sharing (Chapter 13), shape changes (Chapter 14), and loop incompatibility (this chapter) are three of them, but not the only ones a real tensor compiler has to recognize. Apply the Chapter 5-15 depth-level standard (more prose, more diagrams before code) there too.

## Worked Solutions

1. It reads the node's own already-inferred output `Shape` (Chapter 6's `inferShapes()`) for every computed node kind (`Add`, `Mul`, `ReLU`, `FusedElementwise`, and the per-dimension part of `FusedReduction`), and, for a reduction specifically, the `reduceElementCount` Chapter 14's own pass already bakes into the `Sum` step (`FusedReduction`) or, for a not-yet-fused node, the `elementCounts` table Chapter 14's own `evaluate()` already needed (plain `OpKind::Sum`). No new data is introduced anywhere -- both are values every earlier chapter already computed for its own reasons.
2. A `FusedElementwise` node produces exactly one output element per run of its own program -- the same per-element correspondence Chapter 13's `FusedStep` model depends on -- so its output `Shape`'s own dimensions ARE its iteration space, with nothing left over. A `FusedReduction` node's own output is always a scalar (`Shape{}`, zero dimensions) regardless of how many elements it summed, so the dimensions describing HOW MANY elements it actually iterates over live nowhere in its own output shape at all -- they only exist inside the `Sum` step's own `reduceElementCount`, which is exactly the extra loop `buildLoopNest()` has to append by hand.
3. It proves that fusion changes HOW a value is computed (one fused kernel instead of a separate `ReLU` node feeding a separate `Sum` node) without changing how many total iterations that computation actually needs (4, either way). This matters because it confirms `LoopNest` is measuring something real and stable about the underlying computation, not an artifact of whichever pass happened to run -- the same way Chapters 13-14's own byte-counting measured a real change (traffic) while `evaluate()`'s own agreement checks confirmed something else stayed exactly the same (the computed values).
4. A single fixed `innerExtent`, applied uniformly to every outer index, is correct only when `tileSize` evenly divides `originalExtent`. The moment it doesn't, the LAST outer index's own inner loop would run for the full `tileSize` regardless of how many real elements actually remain, reading or writing indices past the end of the real data -- concretely, tiling 13 elements by a tile size of 4 with a fixed inner extent visits indices 12, 13, and 14, where only index 12 is real.
5. The loop nest's own STRUCTURE changes from a 2-loop nest (`[dim0, dim1]`) to a 3-loop nest (`[dim0, dim1_outer, dim1_inner]`) -- tiling always adds exactly one loop for the one dimension being tiled. What stays exactly the same is the total iteration count (12, both before and after) and, more precisely, the full SET of `(dim0, dim1)` index pairs visited -- proved directly by enumerating both sets rather than just comparing totals, since two loop nests could in principle share a total iteration count while visiting genuinely different points.
6. Operator fusion (Chapters 13-14) says `t1` and `s` cannot share a single node's own `FusedStep` program -- `t1` because it has two consumers (Chapter 13's sharing rule), and `s` because it is a reduction (Chapter 14's own rule) -- and Chapter 14 already confirmed Graph B fuses nothing at all as a result. `loopNestsCompatibleForFusion()` asks an unrelated question: do `t1`'s own loop nest (`[dim0:4]`) and `s`'s own loop nest (`[reduce:4]`) have the same loop count and the same extents? They do, so the check reports them compatible. Neither answer is wrong -- they are answers to two different questions (can these VALUES share a program; could these LOOPS share a nest) about the same pair of nodes, which is exactly why real compilers like TVM keep them as separate schedule decisions rather than one combined rule.
7. Nest P (`[dim0:3, dim1:4]`) and nest Q (`[dim0:4, dim1:3]`) share the exact same multiset of extents (`{3, 4}`) but in a different order, and `loopNestsCompatibleForFusion()` reports them incompatible because it compares loop extents position by position, not as an unordered set. This gap wasn't fixed here for the same reason Chapter 10 left `add(a,b)` vs. `add(b,a)` unmerged: recognizing a permuted match safely would require reasoning about whether reordering the loops is actually valid for the computations involved (it usually is for a pure elementwise or reduction loop nest, but not always, once real dependencies enter the picture in later chapters), and a conservative exact check that never produces a false positive is the safer default to ship first.
8. Operator fusion (Chapters 13-14) decides what a single node's own internal `FusedStep` program contains, built on `resolveIntoGroup()` and each pass's own top-level walk. Loop nests (Section 15.1) decide how many times a given node's own program actually runs, built directly on Chapter 6's `Shape`/`inferShapes()` and Chapter 14's `reduceElementCount`. Loop-nest compatibility (Section 15.3) decides whether two SEPARATE nodes' own loops could still be merged at codegen time, built on top of Section 15.1's own `LoopNest` without touching Chapters 13-14's fusion machinery at all. Each layer reads the one below it without modifying it -- the same "extend, don't rewrite" discipline this book has followed since Chapter 8's own `TransformPass` was first introduced.

---

**Sources cited in this chapter:**

None new. The loop-nest / tiling / loop-fusion separation modeled here is the same one Chapter 3 already cited in TVM's own real, documented schedule primitives (`split`, `tile`, `reorder`, `fuse`) -- this chapter builds CUDA Hammer's own minimal version of that same idea, not a new external citation.
