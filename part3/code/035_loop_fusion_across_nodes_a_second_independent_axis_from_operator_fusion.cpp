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
