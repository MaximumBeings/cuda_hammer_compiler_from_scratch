// Chapter 13: Elementwise Fusion
// 027_elementwise_fusion_a_new_node_kind_for_a_fused_kernel_body.cpp
//
// Section 13.1 -- Part 3's first real fusion PASS: elementwiseFusionPass(),
// a fifth TransformPass sharing Chapter 8's exact Graph(const Graph&)
// signature, that actually BUILDS the kind of single fused kernel
// Chapter 12 only ever reasoned about by formula.
//
// Every pass through Chapter 11 rebuilt a graph whose nodes were still
// individually addressable, individually printable through Chapter 7's
// own grammar -- folding, CSE, and simplification all replace ONE node
// with ANOTHER single node. Fusion is different in kind: it replaces
// several nodes with ONE node that internally represents several
// operations. Representing that requires a genuinely new idea CUDA
// Hammer's IR has not needed until now: a node whose own body is a
// small, ordered PROGRAM -- OpKind::FusedElementwise, holding a
// std::vector<FusedStep>, mirroring exactly the "no intermediate
// storage inside the fusion is materialized" description Chapter 3
// already quoted directly from XLA's own real documentation.
//
// A FusedStep's own operands reference EITHER one of the fused node's
// EXTERNAL inputs (a value read from memory once, when the fused kernel
// starts) OR the result of an EARLIER step in the SAME fused node's own
// program (a value that lives in a register for exactly one step and
// never touches memory at all) -- the same "register vs. memory" split
// Chapter 12's own byte-counting formulas assumed, now made concrete as
// real IR.
//
// This section demonstrates the mechanism on the simplest case: a
// straight-line chain where every intermediate node has exactly ONE
// consumer, so nothing yet forces a fusion BOUNDARY (Section 13.2 adds
// that). The pass built here is reused completely unchanged in Sections
// 13.2 and 13.3.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 027_elementwise_fusion_a_new_node_kind_for_a_fused_kernel_body.cpp -o 027_elementwise_fusion_a_new_node_kind_for_a_fused_kernel_body
// Run:     ./027_elementwise_fusion_a_new_node_kind_for_a_fused_kernel_body
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind (from Chapter 4, OpKind extended) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

// FusedElementwise is the one genuinely new OpKind this chapter adds --
// every other kind is exactly Chapter 4's own enum, unchanged.
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

// ==================== FusedStep -- a fused node's own internal program ====================
//
// ExternalInput operands read one of the fused node's own `inputs`
// (resolved once, from memory, when the fused kernel starts). PriorStep
// operands read the result of an earlier FusedStep in the SAME node's
// own steps list -- never memory, always a register.
enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand {
    OperandKind kind;
    int index;
};

struct FusedStep {
    OpKind op;                           // Add, Mul, or ReLU -- never Input/Const/FusedElementwise
    std::vector<FusedOperand> operands;  // 1 operand for ReLU, 2 for Add/Mul
};

// ==================== Node / Graph (from Chapter 4, Node extended) ====================

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;          // EXTERNAL inputs when op == FusedElementwise
    float constValue = 0.0f;            // only meaningful when op == OpKind::Const
    std::vector<FusedStep> fusedSteps;  // only meaningful when op == OpKind::FusedElementwise;
                                         // fusedSteps.back() is this node's own designated output --
                                         // the same "last thing is the output" convention Chapter 9
                                         // established for a whole GRAPH, now echoed at the level of
                                         // one node's own internal program.
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

// ==================== Section 13.1: elementwiseFusionPass() ====================
//
// A node qualifies to be INLINED into whichever single node consumes it
// (rather than materialized as its own separate node) exactly when: it is
// itself Add/Mul/ReLU (not Input/Const, and not already fused), AND it has
// EXACTLY ONE consumer in the ORIGINAL graph. Chapter 4's own consumer-
// counting idiom (File 005's consumerCounts()) decides this, computed
// ONCE, up front, on the graph being fused -- fusion boundaries are a
// property of the ORIGINAL graph's sharing structure, not something the
// pass discovers as it goes.
//
// FusionResult carries the new Graph plus a side table this chapter's own
// later sections use for measurement: which ORIGINAL node id each NEW
// node's value actually represents (itself, for anything not fused; the
// group's own final/root node, for a materialized FusedElementwise).
struct FusionResult {
    Graph graph;
    std::map<int, int> representativeOldId;
};

// Recursively resolves oldId's value from the perspective of the group
// currently under construction (externalInputs/steps belong to the
// CALLER's own in-progress group). If oldId has already been resolved
// into a step earlier in THIS SAME call tree, the existing step is
// reused rather than recomputed -- so a value referenced twice within
// one fused node's own body is still only computed once.
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
        externalInputs.push_back(materialized.at(oldId));  // already built -- topo order guarantees this
        externalIndexByOldId[oldId] = idx;
        return FusedOperand{OperandKind::ExternalInput, idx};
    }

    // p is Add/Mul/ReLU with exactly one consumer -- inline it: resolve
    // ITS OWN operands first (recursively), then append p itself as a
    // new step in the SAME group.
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

        // consumers[oldId] != 1 means: nobody will inline this node into
        // their own group (either nothing reads it at all -- the graph's
        // own designated output -- or more than one node reads it, and
        // duplicating its computation into two different fused bodies
        // would be wasteful and, worse, would silently stop being "the
        // same node" at all). Either way, it must stand alone.
        if (consumers.at(oldId) == 1) continue;  // will be inlined later, by its one consumer

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
            // Nothing was inlined -- this node stands alone; emit it as
            // a plain node of its own original kind, not a trivial
            // one-step FusedElementwise wrapper.
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

// ==================== A small debug printer for the new node kind ====================
//
// Chapter 7's printGraphAsSource() cannot represent a FusedElementwise
// node at all -- Chapter 5's own text grammar has no syntax for a node
// whose body is itself a small program, and teaching it one is
// explicitly out of scope for this book. This is a new, ad hoc, this-
// chapter-only debug format instead.
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

// ==================== evaluate() (from Chapter 9, extended for FusedElementwise) ====================

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
        } else {  // FusedElementwise: run the internal program in a local register array
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
            v = stepVals.back();  // this node's own "last step is the output" convention
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

int main() {
    printf("=== Section 13.1: elementwiseFusionPass(), a straight single-consumer chain ===\n\n");

    // x = Input; a chain of Mul, Add, ReLU, each with exactly ONE
    // consumer -- nothing here forces a fusion boundary yet.
    //   t1 = mul(x, 2)
    //   t2 = add(t1, -3)
    //   t3 = relu(t2)      <- the graph's own designated output
    Graph g;
    Value x  = g.addInput("x");
    Value c2 = g.addConst(2.0f, "c2");
    Value cN3 = g.addConst(-3.0f, "cN3");
    Value t1 = g.addBinary(OpKind::Mul, x, c2, "t1");
    Value t2 = g.addBinary(OpKind::Add, t1, cN3, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t2, "t3");
    (void)t3;

    printf("Original graph (%zu nodes):\n", g.size());
    printGraph(g);

    FusionResult result = elementwiseFusionPass(g);
    const Graph& fused = result.graph;

    printf("\nAfter elementwiseFusionPass() (%zu nodes):\n", fused.size());
    printGraph(fused);

    bool nodeCountDropped = (fused.size() == 4 && g.size() == 6);  // x, c2, cN3, fused  <-  x, c2, cN3, t1, t2, t3
    printf("\nself-check: node count dropped from %zu to %zu -- x/c2/cN3 stay leaves, t1/t2/t3\n",
           g.size(), fused.size());
    printf("collapse into ONE FusedElementwise node (%s)\n", nodeCountDropped ? "confirmed" : "MISMATCH");

    const Node* fusedNode = nullptr;
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::FusedElementwise) fusedNode = n.get();
    }
    bool fusedNodeExists = (fusedNode != nullptr);
    bool threeExternalInputs = fusedNodeExists && fusedNode->inputs.size() == 3;
    bool threeSteps = fusedNodeExists && fusedNode->fusedSteps.size() == 3;
    printf("self-check: exactly one FusedElementwise node was created (%s)\n",
           fusedNodeExists ? "confirmed" : "MISMATCH");
    printf("self-check: it has 3 external inputs (x, c2, cN3) and 3 internal steps (mul, add, relu) (%s)\n",
           (threeExternalInputs && threeSteps) ? "confirmed" : "MISMATCH");

    std::map<std::string, float> in1 = {{"x", 5.0f}};
    float orig1 = evaluate(g, in1).at("t3");
    float fused1 = evaluate(fused, in1).at("t3");
    std::map<std::string, float> in2 = {{"x", 1.0f}};
    float orig2 = evaluate(g, in2).at("t3");
    float fused2 = evaluate(fused, in2).at("t3");

    printf("\nself-check: evaluate(original, x=5).t3 = %g, evaluate(fused, x=5).t3 = %g (%s)\n",
           orig1, fused1, (orig1 == fused1) ? "confirmed" : "MISMATCH");
    printf("self-check: evaluate(original, x=1).t3 = %g, evaluate(fused, x=1).t3 = %g (%s)\n",
           orig2, fused2, (orig2 == fused2) ? "confirmed" : "MISMATCH");

    bool allOk = nodeCountDropped && fusedNodeExists && threeExternalInputs && threeSteps &&
                 (orig1 == fused1) && (orig2 == fused2);
    return allOk ? 0 : 1;
}
