// Chapter 13: Elementwise Fusion
// 028_fusion_boundaries_what_a_shared_value_forces.cpp
//
// Section 13.2 -- reuses Section 13.1's elementwiseFusionPass() and
// evaluate() completely unchanged, and tests it on Chapter 4's own
// diamond graph (File 005): a, b = Input; t1 = Add(a, b); t2 = Mul(t1,
// a); t3 = ReLU(t1); out = Add(t2, t3). Unlike Section 13.1's straight
// chain, t1 here has TWO consumers (t2 AND t3) -- the single-consumer
// rule that decided nothing in 13.1's chain becomes the whole story
// here: t1 CANNOT be inlined into either consumer's fused body without
// either computing it twice (wasteful) or silently letting one fused
// kernel's private copy drift out of sync with the other's (a
// correctness bug, not just a missed optimization) -- so t1 is forced
// to remain its own separate, materialized node, a genuine fusion
// BOUNDARY, while t2, t3, and out -- each read by exactly one thing --
// fuse into a single FusedElementwise node around it.
//
// A second, deliberately modified graph -- identical shape, but with
// the second consumer of t1 removed -- is run through the exact same
// pass to show the boundary disappears the moment the sharing that
// caused it disappears: same pass, same rule, different graph, provably
// different outcome.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 028_fusion_boundaries_what_a_shared_value_forces.cpp -o 028_fusion_boundaries_what_a_shared_value_forces
// Run:     ./028_fusion_boundaries_what_a_shared_value_forces
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from File 027, unchanged) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU, FusedElementwise };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input:            return "Input";
        case OpKind::Const:            return "Const";
        case OpKind::Add:              return "Add";
        case OpKind::Mul:              return "Mul";
        case OpKind::ReLU:             return "ReLU";
        default:                       return "FusedElementwise";
    }
}

enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand {
    OperandKind kind;
    int index;
};

struct FusedStep {
    OpKind op;
    std::vector<FusedOperand> operands;
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
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) || (consumers.at(oldId) != 1);
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
            if (n->op == OpKind::ReLU) {
                v = out.addUnary(OpKind::ReLU, externalInputs[0], n->debugName);
            } else {
                v = out.addBinary(n->op, externalInputs[0], externalInputs[1], n->debugName);
            }
        } else {
            v = out.addFusedElementwise(externalInputs, steps, n->debugName);
        }
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise) {
            printf("  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            if (n->op == OpKind::Const) {
                printf("%g", n->constValue);
            } else {
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    if (i) printf(", ");
                    printf("%%%d", n->inputs[i].nodeId);
                }
            }
            printf(")\n");
        } else {
            printf("  %%%d %s = FusedElementwise(external inputs: ", n->id, n->debugName.c_str());
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) printf(", ");
                printf("ext%zu=%%%d", i, n->inputs[i].nodeId);
            }
            printf(")\n");
            for (size_t s = 0; s < n->fusedSteps.size(); ++s) {
                const FusedStep& step = n->fusedSteps[s];
                printf("      step%zu = %s(", s, opKindStr(step.op).c_str());
                for (size_t i = 0; i < step.operands.size(); ++i) {
                    if (i) printf(", ");
                    const FusedOperand& o = step.operands[i];
                    printf("%s%d", o.kind == OperandKind::ExternalInput ? "ext" : "step", o.index);
                }
                printf(")%s\n", (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output" : "");
            }
        }
    }
}

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
        } else if (n->op == OpKind::ReLU) {
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        } else {
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
                else sv = read(step.operands[0]) * read(step.operands[1]);
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
    printf("=== Section 13.2: Chapter 4's diamond graph -- a SHARED value forces a boundary ===\n\n");

    // The exact diamond from Chapter 4's File 005: t1 has TWO consumers.
    Graph diamond;
    Value a  = diamond.addInput("a");
    Value b  = diamond.addInput("b");
    Value t1 = diamond.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = diamond.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = diamond.addUnary(OpKind::ReLU, t1, "t3");
    Value out = diamond.addBinary(OpKind::Add, t2, t3, "out");
    (void)out;

    printf("Original diamond (%zu nodes):\n", diamond.size());
    printGraph(diamond);

    FusionResult diamondResult = elementwiseFusionPass(diamond);
    const Graph& diamondFused = diamondResult.graph;

    printf("\nAfter elementwiseFusionPass() (%zu nodes):\n", diamondFused.size());
    printGraph(diamondFused);

    bool t1StillMaterialized = false;
    int fusedNodeCount = 0;
    for (const auto& n : diamondFused.nodes()) {
        if (n->debugName == "t1" && n->op == OpKind::Add) t1StillMaterialized = true;
        if (n->op == OpKind::FusedElementwise) fusedNodeCount++;
    }
    printf("\nself-check: 't1' (2 consumers in the original graph) survives as its OWN materialized\n");
    printf("Add node -- NOT inlined into anything (%s)\n", t1StillMaterialized ? "confirmed" : "MISMATCH");
    printf("self-check: exactly one FusedElementwise node exists, holding t2, t3, and out (%s)\n",
           fusedNodeCount == 1 ? "confirmed" : "MISMATCH");
    printf("self-check: node count is 4 (a, b, t1, fused) -- down from the original 6, but NOT\n");
    printf("down to a single node, because t1's own sharing forces a real boundary (%s)\n",
           diamondFused.size() == 4 ? "confirmed" : "MISMATCH");

    std::map<std::string, float> din = {{"a", 4.0f}, {"b", 9.0f}};
    float dOrig = evaluate(diamond, din).at("out");
    float dFused = evaluate(diamondFused, din).at("out");
    printf("\nself-check: evaluate(original, a=4,b=9).out = %g, evaluate(fused, ...).out = %g (%s)\n",
           dOrig, dFused, (dOrig == dFused) ? "confirmed" : "MISMATCH");

    printf("\n=== Control: the SAME pass, on a graph where that sharing is removed ===\n\n");

    // Identical shape, EXCEPT t3 (the second consumer of t1) is gone --
    // 'out' now reads t1 only once, through t2. t1's consumer count drops
    // to 1, and the SAME rule that kept it a boundary above now allows it
    // to be inlined.
    Graph noShare;
    Value a2  = noShare.addInput("a");
    Value b2  = noShare.addInput("b");
    Value t1b = noShare.addBinary(OpKind::Add, a2, b2, "t1");
    Value out2 = noShare.addBinary(OpKind::Mul, t1b, a2, "out");
    (void)out2;

    printf("Original (no sharing) graph (%zu nodes):\n", noShare.size());
    printGraph(noShare);

    FusionResult noShareResult = elementwiseFusionPass(noShare);
    const Graph& noShareFused = noShareResult.graph;

    printf("\nAfter elementwiseFusionPass() (%zu nodes):\n", noShareFused.size());
    printGraph(noShareFused);

    int noShareFusedCount = 0;
    for (const auto& n : noShareFused.nodes()) {
        if (n->op == OpKind::FusedElementwise) noShareFusedCount++;
    }
    bool fullyFused = (noShareFused.size() == 3 && noShareFusedCount == 1);  // a, b, fused(t1, out)
    printf("\nself-check: with the shared consumer gone, t1 AND out fuse into ONE node -- node count\n");
    printf("is 3 (a, b, fused), the SAME pass now removing the boundary entirely (%s)\n",
           fullyFused ? "confirmed" : "MISMATCH");

    std::map<std::string, float> nin = {{"a", 4.0f}, {"b", 9.0f}};
    float nOrig = evaluate(noShare, nin).at("out");
    float nFused = evaluate(noShareFused, nin).at("out");
    printf("self-check: evaluate(original, a=4,b=9).out = %g, evaluate(fused, ...).out = %g (%s)\n",
           nOrig, nFused, (nOrig == nFused) ? "confirmed" : "MISMATCH");

    bool allOk = t1StillMaterialized && (fusedNodeCount == 1) && (diamondFused.size() == 4) &&
                 (dOrig == dFused) && fullyFused && (nOrig == nFused);
    return allOk ? 0 : 1;
}
