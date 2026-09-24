// Appendix F: From torch.compile and TVM's Python API to C++: A Rosetta
// Stone
// 106_the_same_diamond_graph_through_cuda_hammers_own_real_fusion_pass.cpp
//
// Section F.1 -- the CUDA Hammer half of this appendix's first side-by-side
// pair. File 105 builds Chapter 4's own diamond graph in real TVM Relax and
// runs it through TVM's own real fusion pipeline (LegalizeOps ->
// AnnotateTIROpPattern -> FuseOps -> FuseTIR, established in Chapter 26's
// own File 067). This file builds the EXACT same graph shape -- t1=a+b
// feeds t2=t1*c and t3=relu(t1); out=t2+t3 -- through CUDA Hammer's own
// real C++ machinery instead: Value/OpKind/Node/Graph (Chapters 4/13/14,
// unchanged) and boundedReductionFusionPass() (Chapter 16.3's own capstone
// pass, also unchanged). Both sides fuse t2/t3/out into one unit while
// keeping t1 external (it has two consumers on both sides), and both sides
// report that fact with a single integer: TVM's own "PrimFuncs besides
// main" count, and CUDA Hammer's own post-fusion node count.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 106_the_same_diamond_graph_through_cuda_hammers_own_real_fusion_pass.cpp -o 106_driver
// Run:     ./106_driver
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

// ==================== boundedReductionFusionPass() and its helpers (from Chapter 16.3, unchanged) ====================

struct FusionResult {
    Graph graph;
    std::map<int, int> representativeOldId;
};

static std::map<int, long long> computeChainDepths(const Graph& g, const std::map<int, int>& consumers,
                                                     const TopoResult& topo) {
    std::map<int, long long> depth;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        bool isChainCandidate = (n->op == OpKind::Add || n->op == OpKind::Mul || n->op == OpKind::ReLU) &&
                                 consumers.at(id) == 1;
        if (!isChainCandidate) {
            depth[id] = 0;
            continue;
        }
        long long best = 0;
        for (const Value& in : n->inputs) {
            const Node* pred = g.node(in.nodeId);
            bool predIsChainMember = (pred->op == OpKind::Add || pred->op == OpKind::Mul || pred->op == OpKind::ReLU) &&
                                      consumers.at(pred->id) == 1;
            if (predIsChainMember) best = std::max(best, depth.at(pred->id));
        }
        depth[id] = best + 1;
    }
    return depth;
}
static std::set<int> computeSizeCapBoundaries(const Graph& g, const std::map<int, int>& consumers,
                                               long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("computeSizeCapBoundaries: graph is not acyclic");
    std::map<int, long long> depth = computeChainDepths(g, consumers, topo);
    std::set<int> boundaries;
    for (const auto& kv : depth) {
        if (kv.second > 0 && kv.second % maxChainLength == 0) boundaries.insert(kv.first);
    }
    return boundaries;
}

static FusedOperand resolveIntoGroupBounded(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                             const std::set<int>& sizeCapBoundaries,
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
                           (consumers.at(oldId) != 1) ||
                           (sizeCapBoundaries.count(oldId) > 0);
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
        operands.push_back(resolveIntoGroupBounded(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                     externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
    }
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}

static FusionResult boundedReductionFusionPass(const Graph& g, const std::map<int, long long>& elementCounts,
                                                long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("boundedReductionFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;

    std::set<int> sizeCapBoundaries = computeSizeCapBoundaries(g, consumers, maxChainLength);

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
            FusedOperand summedOperand = resolveIntoGroupBounded(n->inputs[0].nodeId, g, consumers, sizeCapBoundaries,
                                                                   materialized, externalInputs, externalIndexByOldId,
                                                                   steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroupBounded(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                         externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
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

int main() {
    printf("=== Appendix F.1: the same diamond graph, through CUDA Hammer's own real fusion pass ===\n\n");

    // Exactly File 105's own real Relax graph: t1=a+b (shared, 2
    // consumers), t2=t1*c, t3=relu(t1), out=t2+t3, over a (6,8)-shaped
    // tensor -- matched to TVM's own DiamondModule shape (a, b, c all
    // R.Tensor((6, 8), "float32")) so both sides fuse the SAME real
    // computation, not just a same-shaped one.
    Graph g;
    Value a = g.addInput("a");
    Value b = g.addInput("b");
    Value c = g.addInput("c");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, c, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");
    (void)out;

    printf("--- Part 1: the unfused graph, printed node by node (CUDA Hammer's own IR, Chapter 7's own printGraphAsSource() shape) ---\n\n");
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)  // \"%s\"\n", n->id, opKindStr(n->op).c_str(),
               [&]() {
                   std::string s;
                   for (size_t i = 0; i < n->inputs.size(); ++i) {
                       if (i) s += ", ";
                       s += "%" + std::to_string(n->inputs[i].nodeId);
                   }
                   return s;
               }().c_str(),
               n->debugName.c_str());
    }
    printf("\nunfused graph: %zu nodes\n", g.size());

    printf("\n--- Part 2: after Chapter 16.3's own real boundedReductionFusionPass() (maxChainLength=10, large enough to place no extra boundary) ---\n\n");
    std::map<int, long long> elementCounts;  // no Sum node in this graph -- never consulted
    FusionResult fused = boundedReductionFusionPass(g, elementCounts, 10);
    for (const auto& n : fused.graph.nodes()) {
        printf("  %%%d = %s  // \"%s\"%s\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str(),
               n->op == OpKind::FusedElementwise ? "  (fused: t2, t3, out collapsed into one FusedElementwise)" : "");
    }
    size_t fusedComputeNodes = 0;
    for (const auto& n : fused.graph.nodes()) {
        if (n->op != OpKind::Input && n->op != OpKind::Const) fusedComputeNodes++;
    }
    printf("\nfused graph: %zu nodes total, %zu of them real compute (non-Input/Const) nodes\n",
           fused.graph.size(), fusedComputeNodes);
    printf("-- t1's own computation lives INSIDE that one FusedElementwise node's own fusedSteps, "
           "never re-materialized through main's own return value: %s\n", (fusedComputeNodes == 1) ? "true" : "false");

    return 0;
}
