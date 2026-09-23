// Chapter 16: Fusion Boundaries
// 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp
//
// Section 16.2 -- every boundary Chapters 13-15 ever built exists to
// preserve CORRECTNESS: sharing (Chapter 13), a shape change (Chapter 14),
// and loop-nest incompatibility (Chapter 15) are all reasons a fusion would
// either compute the wrong answer or couldn't be expressed as one loop nest
// at all. This section builds a genuinely different KIND of boundary --
// one that exists even though fusing further would still be perfectly
// CORRECT, purely because letting a single fused kernel's own body grow
// without limit has real practical costs real compilers have to bound:
// more live values held at once (register pressure), more code generated
// per kernel (instruction cache pressure, compile time). CUDA Hammer's own
// version of that idea is a simple one: maxChainLength, a cap on how many
// steps a single FusedElementwise body may contain before something has to
// give.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp -o 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow
// Run:     ./037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow
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

// ==================== Debug printer (from Chapter 13, unchanged) ====================

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
                out += (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output\n" : "\n";
            }
        }
    }
    return out;
}
static void printGraph(const Graph& g) { printf("%s", formatGraph(g).c_str()); }

// ==================== evaluate() (from Chapter 13, unchanged) ====================

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
        } else {  // FusedElementwise
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
                else sv = read(step.operands[0]) * read(step.operands[1]);  // Mul
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== Chapter 13's own resolveIntoGroup() / elementwiseFusionPass() (unchanged, kept for comparison) ====================

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

// ==================== Section 16.2: a size-cap boundary, precomputed BEFORE fusion runs ====================
//
// A live check ("has the group I'm building already gotten too big") would
// have to be threaded through resolveIntoGroup()'s own recursion -- but
// that recursion resolves a chain's DEEPEST node FIRST (operands are always
// resolved before a node's own step is pushed), so a check made at entry to
// each recursive call would see an empty `steps` for every node in a chain,
// no matter how long the chain is, and only discover the group was too big
// after everything had already been inlined. Rather than threading extra,
// order-dependent state through that recursion, this section computes the
// cap as a SEPARATE, simple pre-pass -- a set of node ids forced external
// for a size reason, decided before fusion ever starts, checked by
// resolveIntoGroup() the exact same way it already checks Leaf/Shared/Reduction:
// as one more static, precomputed property of the node itself.
//
// chainDepth[n] counts how many single-consumer elementwise nodes deep n is
// within its own maximal run: 1 for a node whose relevant operand isn't
// itself such a node (the start of a new chain), or chainDepth[operand] + 1
// otherwise. A node whose own chainDepth is an exact multiple of
// maxChainLength is marked as a boundary -- the last member of one
// maxChainLength-sized segment, forcing the NEXT segment to start fresh.
static std::map<int, long long> computeChainDepths(const Graph& g, const std::map<int, int>& consumers,
                                                     const TopoResult& topo) {
    std::map<int, long long> depth;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        bool isChainCandidate = (n->op == OpKind::Add || n->op == OpKind::Mul || n->op == OpKind::ReLU) &&
                                 consumers.at(id) == 1;
        if (!isChainCandidate) {
            depth[id] = 0;  // Leaf, Shared, or a consumers!=1 node -- not part of any chain itself
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

// resolveIntoGroup(), reused verbatim except for ONE new condition -- the
// same shape every earlier chapter's own extension took: Chapter 14 added
// one condition for Reduction: here, sizeCapBoundaries.count(oldId) adds
// one condition for a size cap, checked as a simple, precomputed set
// lookup, exactly like every other condition in mustBeExternal.
static FusedOperand resolveIntoGroupCapped(int oldId, const Graph& g, const std::map<int, int>& consumers,
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
        operands.push_back(resolveIntoGroupCapped(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                    externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
    }
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}

// boundedFusionPass()'s own top-level walk needs the SAME companion change
// Chapter 14 needed for Sum: the "consumers==1, skip, my consumer will
// build me" shortcut is only safe when nothing ELSE also forces this node
// external. Without also checking sizeCapBoundaries here, a node the cap
// marks as a boundary but whose consumer count is still 1 would be skipped
// by this loop, then refused by resolveIntoGroupCapped() when its consumer
// tries to inline it anyway -- and, exactly like Chapter 14's own Sum
// oversight would have, it would vanish from the output graph instead of
// materializing as its own root.
static FusionResult boundedFusionPass(const Graph& g, long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("boundedFusionPass: graph is not acyclic");
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
        if (consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroupCapped(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
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

static long long maxFusedStepCount(const Graph& g) {
    long long best = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::FusedElementwise) best = std::max(best, static_cast<long long>(n->fusedSteps.size()));
    }
    return best;
}

int main() {
    printf("=== Section 16.2: boundedFusionPass() -- a boundary with nothing to do with correctness ===\n\n");

    // x = Input; c = Const(3); t1 = ReLU(x); t2 = Mul(t1,c); t3 = Add(t2,c);
    // t4 = Mul(t3,c). Every one of t1..t4 has exactly ONE consumer -- under
    // Chapter 13's own UNCAPPED elementwiseFusionPass(), this entire chain
    // fuses into ONE FusedElementwise node, 4 steps. Hand-derivation for
    // maxChainLength=2: chainDepth(t1)=1, chainDepth(t2)=2, chainDepth(t3)=3,
    // chainDepth(t4)=4. Depths divisible by 2: t2 and t4. t4 was ALREADY
    // going to be its own root regardless (zero consumers -- Section 16.1's
    // own GraphRoot category) -- t2 is the genuinely NEW boundary the cap
    // introduces, splitting the chain into [t1,t2] and [t3,t4].
    Graph chain;
    Value x  = chain.addInput("x");
    Value c  = chain.addConst(3.0f, "c");
    Value t1 = chain.addUnary(OpKind::ReLU, x, "t1");
    Value t2 = chain.addBinary(OpKind::Mul, t1, c, "t2");
    Value t3 = chain.addBinary(OpKind::Add, t2, c, "t3");
    Value t4 = chain.addBinary(OpKind::Mul, t3, c, "t4");
    (void)t4;

    printf("Original graph (%zu nodes):\n", chain.size());
    printGraph(chain);

    FusionResult uncapped = elementwiseFusionPass(chain);
    printf("\n--- UNCAPPED (Chapter 13's own elementwiseFusionPass(), no size limit) ---\n\n");
    printGraph(uncapped.graph);

    bool uncappedOk = (uncapped.graph.size() == 3 && maxFusedStepCount(uncapped.graph) == 4);
    printf("\nself-check: uncapped fusion produces ONE FusedElementwise node with all 4 steps,\n");
    printf("6 -> 3 nodes (%s)\n", uncappedOk ? "confirmed" : "MISMATCH");

    FusionResult capped = boundedFusionPass(chain, 2);
    printf("\n--- CAPPED (maxChainLength=2) ---\n\n");
    printGraph(capped.graph);

    bool cappedOk = (capped.graph.size() == 4 && maxFusedStepCount(capped.graph) == 2);
    printf("\nself-check: capped fusion produces TWO FusedElementwise nodes, each with AT MOST\n");
    printf("2 steps, 6 -> 4 nodes -- t2 becomes a NEW boundary the cap alone is responsible for\n");
    printf("(%s)\n", cappedOk ? "confirmed" : "MISMATCH");

    std::map<std::string, float> in = {{"x", 2.0f}};
    float origOut = evaluate(chain, in).at("t4");
    float uncappedOut = evaluate(uncapped.graph, in).at("t4");
    float cappedOut = evaluate(capped.graph, in).at("t4");
    bool sameAnswer = (origOut == uncappedOut) && (origOut == cappedOut);
    printf("\nself-check: evaluate(original, x=2).t4 = %g, evaluate(uncapped,...) = %g,\n",
           origOut, uncappedOut);
    printf("evaluate(capped,...) = %g -- all three agree (%s)\n", cappedOut, sameAnswer ? "confirmed" : "MISMATCH");

    // ---- Real bytes moved: the honest cost of imposing the cap ----
    printf("\n=== The cap's own real, measured cost ===\n\n");
    static constexpr double kBytesPerElement = 4.0;
    // x=[4], c=scalar (1 element); t1..t4 are all shape [4] (ReLU preserves,
    // Mul/Add broadcast [4] with a scalar unchanged).
    std::map<std::string, long long> elems = {{"x", 4}, {"c", 1}, {"t1", 4}, {"t2", 4}, {"t3", 4}, {"t4", 4}};

    auto bytesMovedByName = [&](const Graph& g, const std::map<int, std::string>& oldName) -> long long {
        long long total = 0;
        for (const auto& n : g.nodes()) {
            if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
            for (const Value& in : n->inputs) total += elems.at(oldName.at(in.nodeId));
            total += elems.at(oldName.at(n->id));
        }
        return total;
    };
    std::map<int, std::string> origNames, uncappedNames, cappedNames;
    for (const auto& n : chain.nodes()) origNames[n->id] = n->debugName;
    for (const auto& kv : uncapped.representativeOldId) uncappedNames[kv.first] = origNames.at(kv.second);
    for (const auto& kv : capped.representativeOldId) cappedNames[kv.first] = origNames.at(kv.second);

    long long unfusedElems = bytesMovedByName(chain, origNames);
    long long uncappedElems = bytesMovedByName(uncapped.graph, uncappedNames);
    long long cappedElems = bytesMovedByName(capped.graph, cappedNames);
    long long unfusedBytes = static_cast<long long>(static_cast<double>(unfusedElems) * kBytesPerElement);
    long long uncappedBytes = static_cast<long long>(static_cast<double>(uncappedElems) * kBytesPerElement);
    long long cappedBytes = static_cast<long long>(static_cast<double>(cappedElems) * kBytesPerElement);

    printf("Bytes moved, UNFUSED (4 separate kernels):              %lld\n", unfusedBytes);
    printf("Bytes moved, UNCAPPED fusion (1 kernel, all 4 steps):   %lld\n", uncappedBytes);
    printf("Bytes moved, CAPPED fusion (2 kernels, maxChainLength=2): %lld\n", cappedBytes);

    bool cappedCostsMoreThanUncapped = (cappedBytes > uncappedBytes);
    bool cappedStillBeatsUnfused = (cappedBytes < unfusedBytes);
    printf("\nself-check: capping costs real, measured bytes relative to uncapped fusion (%lld vs.\n",
           cappedBytes);
    printf("%lld) (%s), while still moving far less than staying fully unfused (%lld) (%s) --\n",
           uncappedBytes, cappedCostsMoreThanUncapped ? "confirmed" : "MISMATCH", unfusedBytes,
           cappedStillBeatsUnfused ? "confirmed" : "MISMATCH");
    printf("a real trade-off, not a free lunch: bounded kernel size costs some of fusion's own\n");
    printf("savings, in exchange for a cap real compilers impose for reasons this book's own\n");
    printf("byte-counting was never built to measure (register pressure, compile time).\n");

    bool allOk = uncappedOk && cappedOk && sameAnswer && cappedCostsMoreThanUncapped && cappedStillBeatsUnfused;
    return allOk ? 0 : 1;
}
