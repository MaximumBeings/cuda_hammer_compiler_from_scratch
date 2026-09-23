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
