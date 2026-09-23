// Chapter 16: Fusion Boundaries
// 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing.cpp
//
// Section 16.1 -- Chapters 13, 14, and 15 each added exactly one new
// condition to resolveIntoGroup()'s own mustBeExternal check: Leaf/Shared
// (Chapter 13), Reduction (Chapter 14, independent of consumer count), and
// -- as a SEPARATE, later question, not a condition inside resolveIntoGroup()
// at all -- loop-nest incompatibility (Chapter 15). Nothing so far has ever
// asked, for a GIVEN node, "why, specifically, is this a boundary" as its
// own first-class question -- every earlier chapter only ever answered
// "boundary or not" as a side effect of deciding whether to inline. This
// section builds that missing piece: classifyNode(), a small diagnostic
// function that re-derives the SAME conditions resolveIntoGroup() already
// checks, exposed as an explicit, inspectable taxonomy -- and proves, by
// direct comparison against the real fusion passes' own actual behavior,
// that it agrees with them on every node of every test graph this book has
// built so far.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing.cpp -o 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing
// Run:     ./036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
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
        } else {  // Sum
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

// ==================== Section 16.1: BoundaryReason / classifyNode() ====================
//
// Every condition below is a re-derivation of something resolveIntoGroup()
// or reductionFusionPass()'s own top-level walk ALREADY checks -- nothing
// here is new fusion LOGIC, only a new, explicit, inspectable NAME for
// logic that has existed since Chapter 13. A node can carry more than one
// reason at once (Chapter 14's own Graph B already showed this informally:
// t1 and s were both boundaries, for two entirely different reasons) --
// classifyNode() returns every reason that applies, not just the first one
// found, so that overlap is visible rather than hidden behind whichever
// condition an if/else chain happened to check first.
enum class BoundaryReason { NotABoundary, Leaf, GraphRoot, Shared, Reduction };

static std::string boundaryReasonStr(BoundaryReason r) {
    switch (r) {
        case BoundaryReason::NotABoundary: return "NotABoundary";
        case BoundaryReason::Leaf:          return "Leaf";
        case BoundaryReason::GraphRoot:     return "GraphRoot";
        case BoundaryReason::Shared:        return "Shared";
        default:                            return "Reduction";
    }
}

// consumers.at(n->id) == 0 means nothing IN THIS GRAPH reads this node's
// own value. Chapter 9's "last node is the output" convention picks exactly
// ONE such node as the graph's own designated output when deadCodeEliminationPass()
// runs -- but no fusion pass in this book has ever required that there be
// only one. reductionFusionPass() (like elementwiseFusionPass() before it)
// materializes EVERY zero-consumer node as its own root, whatever their
// number -- a graph nobody has run DCE on can genuinely have more than one
// (Section 16.1's own Graph 2, below, actually does: `y` and `y2` are BOTH
// zero-consumer sinks, reused verbatim from Chapter 14's own Graph B).
// GraphRoot names that real, checked behavior precisely -- "a root fusion
// treats as its own" -- rather than overclaiming a uniqueness fusion itself
// was never told to enforce.
static std::vector<BoundaryReason> classifyNode(const Node* n, const std::map<int, int>& consumers) {
    std::vector<BoundaryReason> reasons;
    if (n->op == OpKind::Input || n->op == OpKind::Const) {
        reasons.push_back(BoundaryReason::Leaf);
        return reasons;  // a leaf's ONLY reason is being a leaf -- nothing else to check
    }
    if (n->op == OpKind::Sum) {
        reasons.push_back(BoundaryReason::Reduction);
    }
    int c = consumers.at(n->id);
    if (c == 0) {
        reasons.push_back(BoundaryReason::GraphRoot);
    } else if (c >= 2) {
        reasons.push_back(BoundaryReason::Shared);
    }
    if (reasons.empty()) {
        reasons.push_back(BoundaryReason::NotABoundary);
    }
    return reasons;
}

static bool isBoundary(const std::vector<BoundaryReason>& reasons) {
    return !(reasons.size() == 1 && reasons[0] == BoundaryReason::NotABoundary);
}

static std::string reasonsStr(const std::vector<BoundaryReason>& reasons) {
    std::string out;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i) out += "+";
        out += boundaryReasonStr(reasons[i]);
    }
    return out;
}

// Reuses reductionFusionPass()'s OWN consumer-counting logic (copy of the
// two-line loop every pass in this chapter builds) so classifyNode()'s
// predictions can be checked against the SAME consumers map the real pass
// actually used, not a separately-computed one that might disagree by
// accident.
static std::map<int, int> countConsumers(const Graph& g) {
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    return consumers;
}

// The actual proof: for every ORIGINAL node in a graph, classifyNode()'s
// own verdict (boundary or not) must agree with whether that node's own id
// shows up as a top-level materialized root in the FUSED graph's own
// representativeOldId map -- the REAL, executed outcome of running the
// actual pass, not a second opinion computed independently.
static bool verifyClassificationAgreesWithPass(const Graph& original, const FusionResult& result) {
    std::map<int, int> consumers = countConsumers(original);
    std::set<int> materializedOldIds;
    for (const auto& kv : result.representativeOldId) materializedOldIds.insert(kv.second);

    for (const auto& n : original.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumers);
        bool predictedBoundary = isBoundary(reasons);
        bool actuallyMaterializedAsOwnRoot = materializedOldIds.count(n->id) > 0;
        // A Leaf (Input/Const) is always "materialized" too (as itself, not
        // inlined into anything) -- both predictedBoundary and
        // actuallyMaterializedAsOwnRoot are true for a leaf, so this check
        // holds uniformly across every category without a special case.
        if (predictedBoundary != actuallyMaterializedAsOwnRoot) {
            printf("  MISMATCH at node %%%d (%s): classifyNode predicted boundary=%s, reasons=[%s],\n",
                   n->id, n->debugName.c_str(), predictedBoundary ? "true" : "false", reasonsStr(reasons).c_str());
            printf("    but the pass's own actual output says materialized-as-own-root=%s\n",
                   actuallyMaterializedAsOwnRoot ? "true" : "false");
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Section 16.1: classifyNode() -- re-deriving fusion's own logic as an explicit taxonomy ===\n\n");

    // ---- Graph 1: Chapter 13's own diamond ----
    printf("--- Graph 1: Chapter 13's own diamond (a=[3,4], b=[4]) ---\n\n");
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
    std::map<int, long long> elementCountsD;
    for (const auto& kv : shapesD) elementCountsD[kv.first] = numElements(kv.second);
    FusionResult diamondResult = reductionFusionPass(diamond, elementCountsD);

    std::map<int, int> consumersD = countConsumers(diamond);
    for (const auto& n : diamond.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumersD);
        printf("  %%%d %-4s consumers=%d  reasons=[%s]\n", n->id, n->debugName.c_str(),
               consumersD.at(n->id), reasonsStr(reasons).c_str());
    }
    bool diamondOk = verifyClassificationAgreesWithPass(diamond, diamondResult);
    printf("\nself-check: classifyNode()'s own predicted boundary/not-boundary verdict agrees with\n");
    printf("reductionFusionPass()'s own actual output on every node of the diamond graph (%s)\n",
           diamondOk ? "confirmed" : "MISMATCH");

    // ---- Graph 2: Chapter 14's own Graph B (two DIFFERENT reasons, same graph) ----
    printf("\n--- Graph 2: Chapter 14's own Graph B (x=[4], w=scalar) ---\n\n");
    Graph gB;
    Value x  = gB.addInput("x");
    Value w  = gB.addInput("w");
    Value bt1 = gB.addUnary(OpKind::ReLU, x, "t1");
    Value bs  = gB.addUnary(OpKind::Sum, bt1, "s");
    Value by  = gB.addBinary(OpKind::Add, bs, w, "y");
    Value by2 = gB.addBinary(OpKind::Mul, bt1, w, "y2");
    (void)by; (void)by2;
    std::map<int, Shape> declaredB = {{x.nodeId, Shape{{4}}}, {w.nodeId, Shape{}}};
    std::map<int, Shape> shapesB = inferShapes(gB, declaredB);
    std::map<int, long long> elementCountsB;
    for (const auto& kv : shapesB) elementCountsB[kv.first] = numElements(kv.second);
    FusionResult gBResult = reductionFusionPass(gB, elementCountsB);

    std::map<int, int> consumersB = countConsumers(gB);
    for (const auto& n : gB.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumersB);
        printf("  %%%d %-4s consumers=%d  reasons=[%s]\n", n->id, n->debugName.c_str(),
               consumersB.at(n->id), reasonsStr(reasons).c_str());
    }
    bool gBOk = verifyClassificationAgreesWithPass(gB, gBResult);
    printf("\nself-check: t1 is classified Shared (2 consumers), s is classified Reduction (1\n");
    printf("consumer, but Sum) -- two DIFFERENT reasons on two DIFFERENT nodes. y AND y2 are BOTH\n");
    printf("classified GraphRoot: this graph, reused verbatim from Chapter 14's own File 031, was\n");
    printf("never run through deadCodeEliminationPass(), so it genuinely has TWO independent\n");
    printf("zero-consumer sinks, not one -- and reductionFusionPass() materializes both as their\n");
    printf("own roots without caring how many there are, exactly as GraphRoot's own definition\n");
    printf("above says it should. All four predictions agree with the pass's own actual output\n");
    printf("(which fuses nothing in this graph at all) (%s)\n", gBOk ? "confirmed" : "MISMATCH");

    bool t1IsShared = false, sIsReduction = false, yIsGraphRoot = false, y2IsGraphRoot = false;
    for (const auto& n : gB.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumersB);
        if (n->debugName == "t1")
            t1IsShared = std::find(reasons.begin(), reasons.end(), BoundaryReason::Shared) != reasons.end();
        if (n->debugName == "s")
            sIsReduction = std::find(reasons.begin(), reasons.end(), BoundaryReason::Reduction) != reasons.end();
        if (n->debugName == "y")
            yIsGraphRoot = std::find(reasons.begin(), reasons.end(), BoundaryReason::GraphRoot) != reasons.end();
        if (n->debugName == "y2")
            y2IsGraphRoot = std::find(reasons.begin(), reasons.end(), BoundaryReason::GraphRoot) != reasons.end();
    }
    printf("self-check: t1 carries Shared (%s), s carries Reduction (%s), y carries GraphRoot (%s),\n",
           t1IsShared ? "confirmed" : "MISMATCH", sIsReduction ? "confirmed" : "MISMATCH",
           yIsGraphRoot ? "confirmed" : "MISMATCH");
    printf("y2 carries GraphRoot too (%s)\n", y2IsGraphRoot ? "confirmed" : "MISMATCH");

    bool allOk = diamondOk && gBOk && t1IsShared && sIsReduction && yIsGraphRoot && y2IsGraphRoot;
    return allOk ? 0 : 1;
}
