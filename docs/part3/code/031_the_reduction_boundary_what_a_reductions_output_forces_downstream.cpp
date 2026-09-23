// Chapter 14: Reduction Fusion
// 031_the_reduction_boundary_what_a_reductions_output_forces_downstream.cpp
//
// Section 14.2 -- Section 14.1 built the MECHANISM (resolveIntoGroup()'s
// new guard, reductionFusionPass()'s own always-materialize-Sum rule).
// This section proves what that mechanism actually FORCES, on two graphs
// that together cover both ways a node can end up standing alone in this
// chapter's model:
//
//   Graph A: a reduction's OWN result is consumed further downstream, by
//   something that would otherwise be free to fuse. Its single consumer
//   cannot inline it -- not because of sharing (Chapter 13's own original
//   rule), but purely because the reduction CHANGED SHAPE.
//
//   Graph B: a value feeds BOTH a reduction AND something else -- Chapter
//   13's original sharing rule and this chapter's new shape-changing rule
//   apply to two DIFFERENT nodes in the SAME graph at once, and both are
//   still correctly respected together, with zero interaction bugs
//   between them.
//
// Same pass, unmodified between the two graphs -- two different real
// outcomes, because the graphs' own structures differ. Exactly Section
// 13.2's own framing, one shape-changing operator later.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 031_the_reduction_boundary_what_a_reductions_output_forces_downstream.cpp -o 031_the_reduction_boundary_what_a_reductions_output_forces_downstream
// Run:     ./031_the_reduction_boundary_what_a_reductions_output_forces_downstream
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from File 030, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input:            return "Input";
        case OpKind::Const:            return "Const";
        case OpKind::Add:              return "Add";
        case OpKind::Mul:              return "Mul";
        case OpKind::ReLU:             return "ReLU";
        case OpKind::Sum:              return "Sum";
        case OpKind::FusedElementwise: return "FusedElementwise";
        default:                       return "FusedReduction";
    }
}

enum class OperandKind { ExternalInput, PriorStep };
struct FusedOperand { OperandKind kind; int index; };
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

struct TopoResult { std::vector<int> order; bool ok = true; };

static TopoResult topologicalSort(const Graph& g) {
    std::map<int, int> inDegree;
    for (const auto& n : g.nodes()) inDegree[n->id] = static_cast<int>(n->inputs.size());
    std::deque<int> ready;
    for (const auto& n : g.nodes()) if (inDegree[n->id] == 0) ready.push_back(n->id);
    TopoResult result;
    while (!ready.empty()) {
        int id = ready.front();
        ready.pop_front();
        result.order.push_back(id);
        for (const auto& n : g.nodes())
            for (const Value& in : n->inputs)
                if (in.nodeId == id)
                    if (--inDegree[n->id] == 0) ready.push_back(n->id);
    }
    result.ok = (result.order.size() == g.size());
    return result;
}

// ==================== Shape / broadcastShapes / inferShapes (from File 030, unchanged) ====================

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
        if (n->op == OpKind::Input || n->op == OpKind::Const) shapes[id] = declaredShapes.at(id);
        else if (n->op == OpKind::Add || n->op == OpKind::Mul)
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        else if (n->op == OpKind::ReLU) shapes[id] = shapes.at(n->inputs[0].nodeId);
        else shapes[id] = Shape{};  // Sum
    }
    return shapes;
}

// ==================== resolveIntoGroup() / reductionFusionPass() (from File 030, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized,
                                      std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId,
                                      std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
                           (consumers.at(oldId) != 1);
    if (mustBeExternal) {
        auto extIt = externalIndexByOldId.find(oldId);
        if (extIt != externalIndexByOldId.end()) return FusedOperand{OperandKind::ExternalInput, extIt->second};
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

// ==================== printGraph() / evaluate() (from File 030, unchanged) ====================

static std::string formatGraph(const Graph& g) {
    std::string out;
    char buf[256];
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise && n->op != OpKind::FusedReduction) {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            out += buf;
            if (n->op == OpKind::Const) {
                snprintf(buf, sizeof(buf), "%g", n->constValue);
                out += buf;
            } else {
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    if (i) out += ", ";
                    snprintf(buf, sizeof(buf), "%%%d", n->inputs[i].nodeId);
                    out += buf;
                }
            }
            out += ")\n";
        } else {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(external inputs: ", n->id, n->debugName.c_str(),
                     opKindStr(n->op).c_str());
            out += buf;
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                snprintf(buf, sizeof(buf), "ext%zu=%%%d", i, n->inputs[i].nodeId);
                out += buf;
            }
            out += ")\n";
            for (size_t s = 0; s < n->fusedSteps.size(); ++s) {
                const FusedStep& step = n->fusedSteps[s];
                snprintf(buf, sizeof(buf), "      step%zu = %s(", s, opKindStr(step.op).c_str());
                out += buf;
                for (size_t i = 0; i < step.operands.size(); ++i) {
                    if (i) out += ", ";
                    const FusedOperand& o = step.operands[i];
                    snprintf(buf, sizeof(buf), "%s%d", o.kind == OperandKind::ExternalInput ? "ext" : "step", o.index);
                    out += buf;
                }
                out += ")";
                if (step.op == OpKind::Sum) {
                    snprintf(buf, sizeof(buf), "  [reduces %lld elements]", step.reduceElementCount);
                    out += buf;
                }
                out += (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output\n" : "\n";
            }
        }
    }
    return out;
}
static void printGraph(const Graph& g) { printf("%s", formatGraph(g).c_str()); }

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName,
                                              const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");
    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) v = inputValuesByName.at(n->debugName);
        else if (n->op == OpKind::Const) v = n->constValue;
        else if (n->op == OpKind::Add) v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::Mul) v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::ReLU) v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        else if (n->op == OpKind::Sum) v = valuesById.at(n->inputs[0].nodeId) * static_cast<float>(elementCounts.at(n->inputs[0].nodeId));
        else {
            std::vector<float> externalVals;
            for (const Value& in : n->inputs) externalVals.push_back(valuesById.at(in.nodeId));
            std::vector<float> stepVals;
            for (const FusedStep& step : n->fusedSteps) {
                auto read = [&](const FusedOperand& o) {
                    return (o.kind == OperandKind::ExternalInput) ? externalVals[o.index] : stepVals[o.index];
                };
                float sv;
                if (step.op == OpKind::ReLU) sv = std::max(0.0f, read(step.operands[0]));
                else if (step.op == OpKind::Add) sv = read(step.operands[0]) + read(step.operands[1]);
                else if (step.op == OpKind::Mul) sv = read(step.operands[0]) * read(step.operands[1]);
                else sv = read(step.operands[0]) * static_cast<float>(step.reduceElementCount);
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

int main() {
    printf("=== Section 14.2: what a reduction's OWN output forces downstream ===\n\n");

    printf("--- Graph A: s = Sum(relu(x)) feeds y = Add(s, a) -- s has exactly ONE consumer ---\n\n");

    // x, a = Input; t1 = ReLU(x) [1 consumer: s]; s = Sum(t1) [1 consumer: y];
    // y = Add(s, a) [0 consumers -- the graph's own output].
    Graph gA;
    Value ax = gA.addInput("x");
    Value aa = gA.addInput("a");
    Value at1 = gA.addUnary(OpKind::ReLU, ax, "t1");
    Value as = gA.addUnary(OpKind::Sum, at1, "s");
    Value ay = gA.addBinary(OpKind::Add, as, aa, "y");
    (void)ay;

    std::map<int, Shape> declaredA = {{ax.nodeId, Shape{{4}}}, {aa.nodeId, Shape{}}};
    std::map<int, Shape> shapesA = inferShapes(gA, declaredA);
    std::map<int, long long> countsA;
    for (const auto& kv : shapesA) countsA[kv.first] = numElements(kv.second);

    printf("Original graph (%zu nodes):\n", gA.size());
    printGraph(gA);

    FusionResult resultA = reductionFusionPass(gA, countsA);
    const Graph& fusedA = resultA.graph;
    printf("\nAfter reductionFusionPass() (%zu nodes):\n", fusedA.size());
    printGraph(fusedA);

    bool aNodeCountDropped = (gA.size() == 5 && fusedA.size() == 4);
    const Node* aFusedNode = nullptr;
    for (const auto& n : fusedA.nodes()) if (n->op == OpKind::FusedReduction) aFusedNode = n.get();
    bool aReductionAbsorbedT1 = aFusedNode && aFusedNode->fusedSteps.size() == 2;
    const Node* aYNode = nullptr;
    for (const auto& n : fusedA.nodes()) if (n->debugName == "y") aYNode = n.get();
    bool aYStaysPlain = aYNode && aYNode->op == OpKind::Add && aYNode->fusedSteps.empty();
    printf("\nself-check: 5 nodes -> 4 -- t1 absorbed into s's own FusedReduction body (2 steps),\n");
    printf("s itself stands alone as a boundary (its consumer y cannot inline it, purely because\n");
    printf("s CHANGED SHAPE, not because of sharing -- s has only ONE consumer here) (%s)\n",
           (aNodeCountDropped && aReductionAbsorbedT1 && aYStaysPlain) ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inA = {{"x", 2.0f}, {"a", 1.0f}};
    float origA = evaluate(gA, inA, countsA).at("y");
    float fusedAOut = evaluate(fusedA, inA, {}).at("y");
    printf("self-check: evaluate(original, x=2,a=1).y = %g, evaluate(fused, ...).y = %g (%s)\n",
           origA, fusedAOut, (origA == fusedAOut) ? "confirmed" : "MISMATCH");

    printf("\n--- Graph B: t1 feeds BOTH a reduction and something else -- two DIFFERENT reasons\n");
    printf("    to stay a boundary, in the SAME graph, at the SAME time ---\n\n");

    // x, w = Input; t1 = ReLU(x) [2 consumers: s AND y2]; s = Sum(t1)
    // [1 consumer: y]; y = Add(s, w) [0 consumers]; y2 = Mul(t1, w)
    // [0 consumers]. t1 stays a boundary because it is SHARED (Chapter
    // 13's own original rule); s stays a boundary because it is a
    // REDUCTION (this chapter's new rule) -- even though s itself has
    // only one consumer, same as Graph A's s.
    Graph gB;
    Value bx = gB.addInput("x");
    Value bw = gB.addInput("w");
    Value bt1 = gB.addUnary(OpKind::ReLU, bx, "t1");
    Value bs = gB.addUnary(OpKind::Sum, bt1, "s");
    Value by = gB.addBinary(OpKind::Add, bs, bw, "y");
    Value by2 = gB.addBinary(OpKind::Mul, bt1, bw, "y2");
    (void)by; (void)by2;

    std::map<int, Shape> declaredB = {{bx.nodeId, Shape{{4}}}, {bw.nodeId, Shape{}}};
    std::map<int, Shape> shapesB = inferShapes(gB, declaredB);
    std::map<int, long long> countsB;
    for (const auto& kv : shapesB) countsB[kv.first] = numElements(kv.second);

    printf("Original graph (%zu nodes):\n", gB.size());
    printGraph(gB);

    FusionResult resultB = reductionFusionPass(gB, countsB);
    const Graph& fusedB = resultB.graph;
    printf("\nAfter reductionFusionPass() (%zu nodes):\n", fusedB.size());
    printGraph(fusedB);

    bool bNodeCountUnchanged = (gB.size() == 6 && fusedB.size() == 6);
    const Node* bT1Node = nullptr;
    const Node* bSNode = nullptr;
    for (const auto& n : fusedB.nodes()) {
        if (n->debugName == "t1") bT1Node = n.get();
        if (n->debugName == "s") bSNode = n.get();
    }
    bool bT1StaysPlain = bT1Node && bT1Node->op == OpKind::ReLU && bT1Node->fusedSteps.empty();
    bool bSIsTrivialFusedReduction = bSNode && bSNode->op == OpKind::FusedReduction && bSNode->fusedSteps.size() == 1;
    printf("\nself-check: 6 nodes -> 6 -- NO fusion happens at all here, and that is the CORRECT\n");
    printf("answer: t1 stays plain (shared -- 2 consumers), s stays its own (trivial, 1-step)\n");
    printf("FusedReduction (shape-changing -- always wrapped even alone, per Section 14.1's own\n");
    printf("stated rule), y and y2 both stay plain (each only 1 step once their own shared/reduced\n");
    printf("operand is external) (%s)\n", (bNodeCountUnchanged && bT1StaysPlain && bSIsTrivialFusedReduction) ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inB = {{"x", 2.0f}, {"w", 3.0f}};
    float origBY = evaluate(gB, inB, countsB).at("y");
    float fusedBY = evaluate(fusedB, inB, {}).at("y");
    float origBY2 = evaluate(gB, inB, countsB).at("y2");
    float fusedBY2 = evaluate(fusedB, inB, {}).at("y2");
    printf("self-check: evaluate(original, x=2,w=3).y = %g, evaluate(fused, ...).y = %g (%s)\n",
           origBY, fusedBY, (origBY == fusedBY) ? "confirmed" : "MISMATCH");
    printf("self-check: evaluate(original, x=2,w=3).y2 = %g, evaluate(fused, ...).y2 = %g (%s)\n",
           origBY2, fusedBY2, (origBY2 == fusedBY2) ? "confirmed" : "MISMATCH");

    bool allOk = aNodeCountDropped && aReductionAbsorbedT1 && aYStaysPlain && (origA == fusedAOut) &&
                 bNodeCountUnchanged && bT1StaysPlain && bSIsTrivialFusedReduction &&
                 (origBY == fusedBY) && (origBY2 == fusedBY2);
    return allOk ? 0 : 1;
}
