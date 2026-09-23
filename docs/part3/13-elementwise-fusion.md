# 13. Elementwise Fusion

**What you will understand:** `elementwiseFusionPass()`, Part 3's first real fusion pass and the fifth real optimization pass to share Chapter 8's exact `TransformPass` signature unmodified -- how it represents several operations as ONE node's own internal program (a genuinely new IR idea, `OpKind::FusedElementwise`), how a value with more than one consumer forces a real fusion boundary that a straight-line chain never has to face, and how much a real, correctness-preserving pass actually saves, measured against the two bounds Chapter 12 could only compute by formula.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph`, `topologicalSort()`, and consumer-counting idiom (`consumerCounts()`), Chapter 6's `Shape`/`inferShapes()`, Chapter 8's `TransformPass` and `runPasses()`, Chapter 9's `evaluate()` and its "last node is the output" convention, and Chapter 12's arithmetic-intensity byte-counting for the exact same diamond graph this chapter reuses.

---

Every pass through Chapter 11 shared one structural trait: it replaced a node with *another single node* -- constant folding replaced a computable `Add` with a `Const`; CSE replaced a duplicate with a pointer to the original; algebraic simplification replaced an identity with an already-existing operand. In every case, the graph's own node-for-node character never changed -- print it with Chapter 7's `printGraphAsSource()` before and after, and you'd still see one line per operation, still parseable by Chapter 5's own grammar. Fusion cannot work this way, because its entire point is to make *several* operations happen inside *one* kernel launch -- and CUDA Hammer's `Graph` has never had a way to say "this one node actually represents three operations, not one." This chapter builds that representation, and the pass that produces it.

```text
WHAT THIS CHAPTER ADDS TO PART 2's OWN PASS LINEAGE:

  Ch9's passes:        Ch10's pass:          Ch11's pass:            THIS chapter's pass:

    constantFoldPass      commonSubexpression   algebraicSimplif-       elementwiseFusionPass
    deadCodeElimination-    EliminationPass        icationPass            (replaces SEVERAL nodes
      Pass                                                                 with ONE node whose own
                                                                            BODY is a small program --
                                                                            every earlier pass replaced
                                                                            one node with one node)

  all five share the EXACT signature Chapter 8 already defined --
  Graph(const Graph&) -- plugging into runPasses() with ZERO changes
  to Chapter 8's own PassManager, across five separate chapters.
```

## 13.1 A New Node Kind for a Fused Kernel Body

### Intuition

Think of how a recipe card works. Most recipe steps stand alone: "chop the onion," written on its own line, could be read and done independently of anything else, and another recipe could reference "an already-chopped onion" as its own starting ingredient. But some steps only make sense read together: "whisk the eggs, then fold in the whisked eggs while still warm" is really one continuous action with an internal handoff -- writing "fold in the eggs" as its own separate card, to be done whenever someone gets around to it, would let them go cold first, changing the result. A `FusedElementwise` node is CUDA Hammer's way of writing several operations on ONE card, with the "handoff" between them made explicit as something that happens immediately, in place, never left to cool.

### Background

Chapter 3 already described exactly this shape in a real, production compiler: XLA's own fusion computations, where, quoting Chapter 3's own citation, "no intermediate storage inside the fusion is materialized in HBM." This chapter builds CUDA Hammer's own version of that idea. A `FusedElementwise` node's `inputs` list holds only its **external** inputs -- values that genuinely have to be read from memory when the fused kernel starts. Its own internal computation is a small, ordered program: a `std::vector<FusedStep>`, where each `FusedStep` is one `Add`, `Mul`, or `ReLU`, and each of ITS OWN operands is either an `ExternalInput` (read once, from the node's own `inputs` list) or a `PriorStep` (the result of an earlier step in the SAME program, which never touches memory at all -- it lives in a register for exactly the one step that reads it).

```text
A FusedElementwise NODE'S OWN SHAPE:

  Node {
    op = FusedElementwise
    inputs = [ external value A, external value B, ... ]   -- read from memory ONCE each
    fusedSteps = [
      step0 = SomeOp(operand, operand)     operand is ExternalInput(i) or PriorStep(j)
      step1 = SomeOp(operand, operand)     -- PriorStep(j) means "step j's own result,
      ...                                     still sitting in a register"
      stepN = SomeOp(operand, operand)     -- THIS node's own designated output
    ]
  }

  fusedSteps.back() is this node's own output -- the SAME "last thing is
  the output" convention Chapter 9 stated for a whole GRAPH, now echoed
  at the level of one node's own internal program.
```

`elementwiseFusionPass()` builds this by walking the graph in topological order, deciding at each `Add`/`Mul`/`ReLU` node whether it can be *inlined* into whatever single thing consumes it, or whether it must stand alone. The rule for inlining is exactly Chapter 4's own consumer-counting idiom (`File 005`'s `consumerCounts()`), computed once, up front: a node inlines into its consumer's own fused body if and only if it is itself `Add`/`Mul`/`ReLU` **and** it has exactly one consumer in the *original* graph. This section's own test graph is the simplest case where that rule never has to say no -- every node has exactly one consumer, so nothing yet forces a boundary; Section 13.2 is where the rule actually starts deciding something.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 027_elementwise_fusion_a_new_node_kind_for_a_fused_kernel_body.cpp -o 027_elementwise_fusion_a_new_node_kind_for_a_fused_kernel_body
./027_elementwise_fusion_a_new_node_kind_for_a_fused_kernel_body
```

**Output:**

```text
=== Section 13.1: elementwiseFusionPass(), a straight single-consumer chain ===

Original graph (6 nodes):
  %0 x = Input()
  %1 c2 = Const(2)
  %2 cN3 = Const(-3)
  %3 t1 = Mul(%0, %1)
  %4 t2 = Add(%3, %2)
  %5 t3 = ReLU(%4)

After elementwiseFusionPass() (4 nodes):
  %0 x = Input()
  %1 c2 = Const(2)
  %2 cN3 = Const(-3)
  %3 t3 = FusedElementwise(external inputs: ext0=%0, ext1=%1, ext2=%2)
      step0 = Mul(ext0, ext1)
      step1 = Add(step0, ext2)
      step2 = ReLU(step1)  <- this node's own output

self-check: node count dropped from 6 to 4 -- x/c2/cN3 stay leaves, t1/t2/t3
collapse into ONE FusedElementwise node (confirmed)
self-check: exactly one FusedElementwise node was created (confirmed)
self-check: it has 3 external inputs (x, c2, cN3) and 3 internal steps (mul, add, relu) (confirmed)

self-check: evaluate(original, x=5).t3 = 7, evaluate(fused, x=5).t3 = 7 (confirmed)
self-check: evaluate(original, x=1).t3 = 0, evaluate(fused, x=1).t3 = 0 (confirmed)
```

!!! note "A value used twice by the SAME consumer is a real, honest edge case"
    `consumers[id]` counts EDGES, not distinct consumer nodes -- if some node were written as `Mul(t1, t1)`, `t1` would be counted as having 2 consumers (it appears twice in one node's own `inputs`), even though only ONE node actually reads it. The pass would conservatively materialize `t1` as its own separate node rather than inlining it, missing a real fusion opportunity. This is a genuine, stated limitation -- safe (never wrong), just not maximally aggressive -- in the same spirit as Chapter 10's own `add(a,b)` vs. `add(b,a)` conservatism. None of this chapter's own test graphs happen to exercise it.

## 13.2 Fusion Boundaries: What a Shared Value Forces

### Intuition

Picture two chefs sharing one prepared sauce. If only ONE dish uses that sauce, the cook can fold the sauce-making right into that dish's own cooking process -- taste, adjust, plate, all in one continuous motion, nothing sitting around. But if TWO different dishes both need the same sauce, someone has to make a full batch, set it down in a container both chefs can reach, and let each dish's own cook take what they need from that one container -- the sauce becomes a real, standalone thing precisely because more than one thing depends on it. Chapter 4's own diamond graph is exactly this second case: `t1` (`a + b`) is used by both `t2` and `t3` -- it cannot be folded into either one's own fused body without either computing it twice or letting the two fused kernels silently drift out of sync with each other.

### Background

Section 13.1's chain never had to answer a hard question, because every node in it had exactly one consumer. Chapter 4's diamond graph does: `t1 = Add(a, b)` is consumed by BOTH `t2 = Mul(t1, a)` and `t3 = ReLU(t1)`. The rule from Section 13.1 -- inline only when a node has exactly one consumer -- says `t1` must stand alone: `consumers[t1] == 2`. `t2` and `t3` each have exactly one consumer (`out`), so they inline into `out`'s own fused body. The result is a real, structural fusion **boundary**: two kernels, not one -- `t1` on its own, and a `FusedElementwise` node holding `t2`, `t3`, and `out` together.

```text
CHAPTER 4's DIAMOND, THROUGH elementwiseFusionPass()'s OWN RULE:

  ORIGINAL:            a   b
                         \ /
                     t1 = Add(a,b)      -- consumers[t1] = 2 (t2 AND t3 both read it)
                        /    \
             t2=Mul(t1,a)  t3=ReLU(t1)  -- consumers[t2]=1, consumers[t3]=1 (only 'out' reads each)
                        \    /
                      out=Add(t2,t3)    -- consumers[out] = 0 (the graph's own output)

  AFTER FUSION:      a   b
                       \ /
                  t1 = Add(a,b)          -- STANDS ALONE: 2 consumers, can't be inlined
                     |      \
                     |    (read once as an external input, reused for BOTH steps below)
                      \____  \____
                           \  \
                     out = FusedElementwise(ext0=t1, ext1=a):
                             step0 = Mul(ext0, ext1)     -- was t2
                             step1 = ReLU(ext0)           -- was t3 (reuses ext0, no re-read)
                             step2 = Add(step0, step1)    -- was out, this node's own output
```

The SAME pass, unmodified, run on a graph where that sharing is absent, makes a different decision -- not because the pass changed, but because the graph's own structure changed. This section's own File 028 proves both directions: the diamond's real boundary, and a second, deliberately modified graph (same shape, minus the second consumer of `t1`) where the exact same rule now allows full fusion into one node.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 028_fusion_boundaries_what_a_shared_value_forces.cpp -o 028_fusion_boundaries_what_a_shared_value_forces
./028_fusion_boundaries_what_a_shared_value_forces
```

**Output:**

```text
=== Section 13.2: Chapter 4's diamond graph -- a SHARED value forces a boundary ===

Original diamond (6 nodes):
  %0 a = Input()
  %1 b = Input()
  %2 t1 = Add(%0, %1)
  %3 t2 = Mul(%2, %0)
  %4 t3 = ReLU(%2)
  %5 out = Add(%3, %4)

After elementwiseFusionPass() (4 nodes):
  %0 a = Input()
  %1 b = Input()
  %2 t1 = Add(%0, %1)
  %3 out = FusedElementwise(external inputs: ext0=%2, ext1=%0)
      step0 = Mul(ext0, ext1)
      step1 = ReLU(ext0)
      step2 = Add(step0, step1)  <- this node's own output

self-check: 't1' (2 consumers in the original graph) survives as its OWN materialized
Add node -- NOT inlined into anything (confirmed)
self-check: exactly one FusedElementwise node exists, holding t2, t3, and out (confirmed)
self-check: node count is 4 (a, b, t1, fused) -- down from the original 6, but NOT
down to a single node, because t1's own sharing forces a real boundary (confirmed)

self-check: evaluate(original, a=4,b=9).out = 65, evaluate(fused, ...).out = 65 (confirmed)

=== Control: the SAME pass, on a graph where that sharing is removed ===

Original (no sharing) graph (4 nodes):
  %0 a = Input()
  %1 b = Input()
  %2 t1 = Add(%0, %1)
  %3 out = Mul(%2, %0)

After elementwiseFusionPass() (3 nodes):
  %0 a = Input()
  %1 b = Input()
  %2 out = FusedElementwise(external inputs: ext0=%0, ext1=%1)
      step0 = Add(ext0, ext1)
      step1 = Mul(step0, ext0)  <- this node's own output

self-check: with the shared consumer gone, t1 AND out fuse into ONE node -- node count
is 3 (a, b, fused), the SAME pass now removing the boundary entirely (confirmed)
self-check: evaluate(original, a=4,b=9).out = 52, evaluate(fused, ...).out = 52 (confirmed)
```

!!! note "This is Part 3's first real preview of Chapter 16's own subject"
    `t1`'s two consumers forcing a real, unavoidable boundary is not a limitation of THIS pass specifically -- it is a correctness requirement any real fusion pass has to respect, in any real compiler. Chapter 16, "Fusion Boundaries," returns to this exact question in far more depth (memory limits, non-elementwise ops, control flow). This section's own diamond graph is the simplest possible case where a boundary is unavoidable, not the last word on when one is.

## 13.3 Measuring a Real Fusion Pass Against Chapter 12's Two Bounds

### Intuition

Chapter 12 computed two numbers by formula for this exact diamond graph and never had to justify either one against a real algorithm: 496 bytes for the fully unfused version (four separate kernels), and an idealized 112 bytes for a single kernel covering the *entire* graph at once -- a number that quietly assumed away `t1`'s own sharing, since a formula doesn't have to worry about whether merging everything into one kernel would actually stay correct. This section runs the actual pass and measures what it actually produces, landing somewhere real between those two numbers -- less savings than the idealized bound promised, because a real pass has to keep the answer right.

### Background

`elementwiseFusionPass()` returns a `FusionResult`, not a bare `Graph` -- its `representativeOldId` side table is genuinely useful (this section uses it directly, to carry Chapter 6's `inferShapes()` results on the original graph across the pass), but it is *analysis* information, not the transformation itself, echoing the same analysis-pass/transform-pass distinction Chapter 2 first drew and Chapter 8 formalized. A one-line wrapper, `elementwiseFusionTransform()`, adapts the pass to Chapter 8's exact `TransformPass` contract for use inside `runPasses()`, which only ever needs the new graph itself -- proving, a fifth time since Chapter 9, that this book's PassManager infrastructure still needs zero changes.

Byte-counting generalizes cleanly from Chapter 12's own formula to this chapter's real, post-fusion graph: any node that is not a leaf -- whether a plain `Add`/`Mul`/`ReLU` that never got fused, or a `FusedElementwise` node holding several -- reads every one of its own `inputs` (its external reads) from memory and writes its own single output back. A `FusedElementwise` node's internal steps never appear in this count at all, which is precisely the point of building one.

```text
BEFORE vs. AFTER, ON CHAPTER 4's DIAMOND (Chapter 12's own graph, Section 12.3):

  BEFORE (4 separate kernels -- t1, t2, t3, out):        496 bytes  (Chapter 12's own figure)
  AFTER elementwiseFusionPass() (t1 stands alone,
         {t2,t3,out} fuse into ONE kernel):               256 bytes  (this section's REAL measurement)
  Chapter 12's idealized single-kernel bound
         (assumed the WHOLE graph could fuse,
          never checked against t1's own sharing):        112 bytes  (never actually built)

              112 -- 256 -- 496   (strictly increasing, left to right)
   idealized (unreachable)   REAL PASS   fully unfused
```

```cpp
// Chapter 13: Elementwise Fusion
// 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds.cpp
//
// Section 13.3 -- closes the chapter two ways: plugging
// elementwiseFusionPass() (Sections 13.1/13.2, reused verbatim) into
// Chapter 8's own unmodified runPasses(), and measuring REAL bytes moved
// on the diamond graph both before and after the pass actually runs,
// against the two bounds Chapter 12 computed by formula (Section 12.3's
// File 026): 496 bytes for the fully unfused graph, and an IDEALIZED
// 112 bytes for a single kernel covering the whole graph at once --
// a bound Chapter 12 never had to justify was achievable, because it
// never ran an actual correctness-preserving pass to produce it.
//
// elementwiseFusionPass() returns a FusionResult, not a bare Graph --
// its representativeOldId side table is genuinely useful (this section
// uses it directly, to carry Chapter 6's inferShapes() results across
// the pass), but it is ANALYSIS information, not the transformation
// itself, the same distinction Chapter 2 drew between a pass that reads
// an IR and a pass that produces a new one. A one-line wrapper adapts
// the pass to Chapter 8's exact TransformPass contract -- Graph(const
// Graph&) -- for use inside runPasses(), which only ever needs the new
// graph itself.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds.cpp -o 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds
// Run:     ./029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Files 027/028, unchanged) ====================

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
struct FusedOperand { OperandKind kind; int index; };
struct FusedStep { OpKind op; std::vector<FusedOperand> operands; };

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
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) || (consumers.at(oldId) != 1);
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

// TransformPass-compatible wrapper -- Chapter 8's PassManager only ever
// needs the new Graph itself; the representativeOldId side table is
// extra analysis this chapter's own measurement code below asks for
// directly, calling elementwiseFusionPass() (not this wrapper).
static Graph elementwiseFusionTransform(const Graph& g) { return elementwiseFusionPass(g).graph; }

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise) {
            printf("  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            if (n->op == OpKind::Const) printf("%g", n->constValue);
            else {
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
        if (n->op == OpKind::Input) v = inputValuesByName.at(n->debugName);
        else if (n->op == OpKind::Const) v = n->constValue;
        else if (n->op == OpKind::Add) v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::Mul) v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::ReLU) v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
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

// ==================== Shape / broadcastShapes / inferShapes (from Chapter 6, unchanged) ====================

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
        } else {
            shapes[id] = shapes.at(n->inputs[0].nodeId);  // ReLU -- shape-preserving
        }
    }
    return shapes;
}

// ==================== Chapter 8's TransformPass / runPasses() (unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;

struct NamedPass {
    std::string name;
    TransformPass pass;
};

static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass (%zu nodes) ---\n", initial.size());
    printGraph(initial);

    // Graph deliberately has no copy constructor (it owns its nodes
    // through unique_ptr, exactly like Chapter 4's own Graph) -- so the
    // FIRST pass reads `initial` directly by const reference, and every
    // pass after that reads the previous pass's own returned Graph by
    // reference, moved into `current` only once it has already been
    // validated. Same discipline as Chapter 8's own runPasses().
    Graph current = passes[0].pass(initial);
    {
        TopoResult topo = topologicalSort(current);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + passes[0].name + "' produced a graph that is not acyclic");
        }
    }
    printf("\n--- after pass '%s' (%zu nodes) ---\n", passes[0].name.c_str(), current.size());
    printGraph(current);

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& np = passes[i];
        Graph next = np.pass(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + np.name + "' produced a graph that is not acyclic");
        }
        printf("\n--- after pass '%s' (%zu nodes) ---\n", np.name.c_str(), next.size());
        printGraph(next);
        current = std::move(next);
    }
    return current;
}

// ==================== Section 13.3: real bytes moved, before vs. after ====================
//
// Uniform for ANY graph this chapter's IR can build: a leaf (Input/Const)
// moves nothing; every other node -- a plain Add/Mul/ReLU OR a
// FusedElementwise -- reads every one of its own `inputs` (its EXTERNAL
// reads) from memory and writes its own single output back. A
// FusedElementwise node's internal steps never appear in this count at
// all -- exactly the point of building one.
static constexpr double kBytesPerElement = 4.0;

static long long bytesMoved(const Graph& g, const std::map<int, long long>& elementCounts) {
    long long totalElements = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        for (const Value& in : n->inputs) totalElements += elementCounts.at(in.nodeId);
        totalElements += elementCounts.at(n->id);
    }
    return static_cast<long long>(static_cast<double>(totalElements) * kBytesPerElement);
}

int main() {
    printf("=== Section 13.3: elementwiseFusionPass() through Chapter 8's own runPasses() ===\n\n");

    Graph diamond;
    Value a  = diamond.addInput("a");
    Value b  = diamond.addInput("b");
    Value t1 = diamond.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = diamond.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = diamond.addUnary(OpKind::ReLU, t1, "t3");
    Value out = diamond.addBinary(OpKind::Add, t2, t3, "out");
    (void)out;

    Graph afterPasses = runPasses(diamond, {{"elementwiseFusionPass", elementwiseFusionTransform}});

    bool passManagerAccepted = (afterPasses.size() == 4);
    printf("\nself-check: runPasses() accepted elementwiseFusionTransform with ZERO changes to\n");
    printf("Chapter 8's own PassManager -- the fifth real OPTIMIZATION pass (after constant\n");
    printf("folding, dead code elimination, CSE, and algebraic simplification) to plug into\n");
    printf("this same infrastructure unmodified (%s)\n", passManagerAccepted ? "confirmed" : "MISMATCH");

    printf("\n=== Real bytes moved: before vs. after, against Chapter 12's two bounds ===\n\n");

    std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> originalShapes = inferShapes(diamond, declared);
    std::map<int, long long> originalElementCounts;
    for (const auto& kv : originalShapes) originalElementCounts[kv.first] = numElements(kv.second);

    FusionResult fusionResult = elementwiseFusionPass(diamond);
    const Graph& fused = fusionResult.graph;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fusionResult.representativeOldId.at(n->id);
        fusedElementCounts[n->id] = originalElementCounts.at(oldId);
    }

    long long bytesBefore = bytesMoved(diamond, originalElementCounts);
    long long bytesAfter = bytesMoved(fused, fusedElementCounts);

    printf("Bytes moved, BEFORE fusion (4 separate kernels -- t1, t2, t3, out):     %lld\n", bytesBefore);
    printf("Bytes moved, AFTER elementwiseFusionPass() (t1 stays a boundary,\n");
    printf("             t2/t3/out fuse into one kernel):                          %lld\n", bytesAfter);
    printf("\nChapter 12's own two bounds for this exact graph (Section 12.3, computed by\n");
    printf("formula, never by an actual pass):\n");
    printf("  fully UNFUSED (4 kernels):                          496 bytes  (matches 'BEFORE' above)\n");
    printf("  IDEALIZED single kernel for the WHOLE graph at once: 112 bytes  (never actually built)\n");

    bool beforeMatchesCh12Unfused = (bytesBefore == 496);
    bool afterIsBetweenTheBounds = (bytesAfter > 112 && bytesAfter < 496);
    printf("\nself-check: bytes moved before fusion exactly matches Chapter 12's own 496-byte\n");
    printf("unfused figure for this graph (%s)\n", beforeMatchesCh12Unfused ? "confirmed" : "MISMATCH");
    printf("self-check: bytes moved after this REAL pass (%lld) is strictly BETWEEN Chapter 12's\n", bytesAfter);
    printf("idealized 112-byte bound and its 496-byte unfused figure -- real, substantial\n");
    printf("savings, honestly short of an ideal that never had to respect t1's own sharing (%s)\n",
           afterIsBetweenTheBounds ? "confirmed" : "MISMATCH");

    double flops = 48.0;  // identical either way -- computed by hand from Chapter 12's own File 026
    double aiBefore = flops / static_cast<double>(bytesBefore);
    double aiAfter = flops / static_cast<double>(bytesAfter);
    printf("\nAI_before = 48 / %lld bytes = %.6f FLOPs/byte\n", bytesBefore, aiBefore);
    printf("AI_after  = 48 / %lld bytes = %.6f FLOPs/byte\n", bytesAfter, aiAfter);
    bool aiImproved = aiAfter > aiBefore;
    printf("self-check: arithmetic intensity strictly improved from a REAL pass run, not just a\n");
    printf("formula (%s)\n", aiImproved ? "confirmed" : "MISMATCH");

    std::map<std::string, float> in = {{"a", 4.0f}, {"b", 9.0f}};
    float origOut = evaluate(diamond, in).at("out");
    float fusedOut = evaluate(fused, in).at("out");
    bool sameAnswer = (origOut == fusedOut);
    printf("\nself-check: evaluate(original, a=4,b=9).out = %g, evaluate(fused, ...).out = %g (%s)\n",
           origOut, fusedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = passManagerAccepted && beforeMatchesCh12Unfused && afterIsBetweenTheBounds &&
                 aiImproved && sameAnswer;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds.cpp -o 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds
./029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds
```

**Output:**

```text
=== Section 13.3: elementwiseFusionPass() through Chapter 8's own runPasses() ===

--- before any pass (6 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 t1 = Add(%0, %1)
  %3 t2 = Mul(%2, %0)
  %4 t3 = ReLU(%2)
  %5 out = Add(%3, %4)

--- after pass 'elementwiseFusionPass' (4 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 t1 = Add(%0, %1)
  %3 out = FusedElementwise(external inputs: ext0=%2, ext1=%0)
      step0 = Mul(ext0, ext1)
      step1 = ReLU(ext0)
      step2 = Add(step0, step1)  <- this node's own output

self-check: runPasses() accepted elementwiseFusionTransform with ZERO changes to
Chapter 8's own PassManager -- the fifth real OPTIMIZATION pass (after constant
folding, dead code elimination, CSE, and algebraic simplification) to plug into
this same infrastructure unmodified (confirmed)

=== Real bytes moved: before vs. after, against Chapter 12's two bounds ===

Bytes moved, BEFORE fusion (4 separate kernels -- t1, t2, t3, out):     496
Bytes moved, AFTER elementwiseFusionPass() (t1 stays a boundary,
             t2/t3/out fuse into one kernel):                          256

Chapter 12's own two bounds for this exact graph (Section 12.3, computed by
formula, never by an actual pass):
  fully UNFUSED (4 kernels):                          496 bytes  (matches 'BEFORE' above)
  IDEALIZED single kernel for the WHOLE graph at once: 112 bytes  (never actually built)

self-check: bytes moved before fusion exactly matches Chapter 12's own 496-byte
unfused figure for this graph (confirmed)
self-check: bytes moved after this REAL pass (256) is strictly BETWEEN Chapter 12's
idealized 112-byte bound and its 496-byte unfused figure -- real, substantial
savings, honestly short of an ideal that never had to respect t1's own sharing (confirmed)

AI_before = 48 / 496 bytes = 0.096774 FLOPs/byte
AI_after  = 48 / 256 bytes = 0.187500 FLOPs/byte
self-check: arithmetic intensity strictly improved from a REAL pass run, not just a
formula (confirmed)

self-check: evaluate(original, a=4,b=9).out = 65, evaluate(fused, ...).out = 65 (confirmed)
```

!!! warning "[COMMON TRAP] Assuming a real pass reaches an idealized bound"
    Chapter 12's 112-byte figure was always a *ceiling*, not a target -- it described what a single kernel over the ENTIRE graph would cost, without asking whether building that single kernel would still be correct. The moment a real pass has to respect genuine sharing (`t1`'s two consumers), the achievable number moves up, not because the pass is weak, but because the alternative (duplicating `t1`'s computation into two separate fused kernels, or letting two kernels silently disagree about its value) is not actually an option. A real measurement landing between two formula-derived bounds, rather than at the more optimistic one, is the expected and correct outcome, not a bug to chase.

## Chapter Summary

This chapter opened Part 3's real machinery with `elementwiseFusionPass()`, the fifth real optimization pass to share Chapter 8's exact `TransformPass` signature unmodified. Section 13.1 introduced the one genuinely new IR idea this book has needed since Chapter 4: `OpKind::FusedElementwise`, a node whose own body is a small ordered program of `FusedStep`s, each reading either an external input (memory, once) or a prior step's own result (a register, never memory) -- demonstrated on a simple single-consumer chain that collapses cleanly into one node. Section 13.2 showed the rule that actually decides fusion boundaries -- Chapter 4's own consumer-counting idiom -- using Chapter 4's diamond graph, where `t1`'s two consumers force it to remain a separate, materialized node even as its own two consumers fuse together around it, and proved the SAME pass fully fuses a second, deliberately modified graph once that sharing is removed. Section 13.3 closed the chapter by running the pass through Chapter 8's own unmodified `runPasses()` and measuring real bytes moved on the diamond graph both before and after: 496 bytes before, 256 after -- landing, honestly and by direct measurement, strictly between Chapter 12's two formula-derived bounds (496 fully unfused, an unreachable idealized 112), a genuine improvement that respects a genuine correctness constraint no formula alone had to face.

## Self-Check Questions

1. Every pass through Chapter 11 replaced one node with another single node. Why can't elementwise fusion work that way, and what genuinely new idea does `OpKind::FusedElementwise` add to represent it?
2. What is the difference between a `FusedStep`'s `ExternalInput` operand and its `PriorStep` operand, in terms of what actually happens at runtime?
3. What rule decides whether a node gets inlined into its consumer's fused body, versus materialized as its own separate node -- and where does that rule get computed?
4. In Section 13.2's diamond graph, `t1` has two consumers, yet its own OWN operands (`a` and `b`) both have exactly one consumer within `t1`. Why does `t1` itself still have to be materialized, even though nothing about `a` or `b` individually forces it?
5. Section 13.2's control graph (with the second consumer of `t1` removed) fully fuses into one node using the exact same pass. What changed between the two graphs to produce a different outcome?
6. Why does `elementwiseFusionPass()` return a `FusionResult` rather than a bare `Graph`, and why does Section 13.3 still need a separate one-line wrapper to use it with `runPasses()`?
7. Section 13.3 measures 256 bytes moved after fusion, strictly between Chapter 12's 112-byte idealized bound and its 496-byte unfused figure. Why was landing exactly at 112 bytes never a realistic possibility for this specific graph?
8. What would `resolveIntoGroup()` do differently if it did NOT check `stepIndexByOldId` before recursing into an already-inlined node's own operands again?

## Where We Go Next

This chapter fused chains and simple DAGs of independent elementwise operations -- `Add`, `Mul`, `ReLU` -- where every node reads and writes values shaped exactly like its neighbors, one output element at a time, with no operation needing to look at more than one input element to produce one output element. A reduction (summing an entire tensor down to one value, or one per row) breaks that assumption outright: it reads many elements to produce one, and fusing computation around a reduction raises genuinely new questions this chapter's own single-consumer rule was never built to answer. Chapter 14, "Reduction Fusion," takes up exactly that question next.

## Worked Solutions

1. Every earlier pass changed WHAT a node computed (or which existing node's value a consumer should read instead), but never how many operations one node represented -- the graph stayed one-line-per-operation printable through Chapter 7's own grammar the whole time. Fusion's entire point is to make SEVERAL operations happen inside ONE kernel launch, which requires a node that can represent more than one operation internally. `OpKind::FusedElementwise` adds exactly that: a node whose body is `std::vector<FusedStep>`, a small ordered program, rather than a single fixed operation.
2. An `ExternalInput` operand is read from the fused node's own `inputs` list -- a value that came from memory when the fused kernel started, read exactly once no matter how many internal steps reference it. A `PriorStep` operand reads the result of an earlier step in the SAME node's own program -- a value that lives in a register for exactly as long as it takes later steps to consume it, and never touches memory at all.
3. A node inlines into its single consumer's fused body if and only if it is itself `Add`/`Mul`/`ReLU` (not `Input`/`Const`) AND it has exactly one consumer in the ORIGINAL graph. Both conditions are checked against `consumers`, a map computed ONCE, up front, by walking every node's own `inputs` list on the ORIGINAL graph -- reusing Chapter 4's own consumer-counting idiom from File 005 -- before the pass builds a single new node.
4. `t1` itself has 2 consumers (`t2` AND `t3` both read it) -- that is the condition that matters for whether `t1` gets inlined into SOMETHING ELSE's fused body, and it fails: `t1` cannot be inlined into two different fused kernels without either duplicating its own computation or risking the two copies disagreeing. Whether `t1`'s OWN operands (`a`, `b`) each have one consumer only decides whether `a` and `b` themselves get inlined INTO `t1` -- a separate, independent question that doesn't change what `t1`'s own consumer count requires of `t1`.
5. In the control graph, `t3` (the second consumer of `t1`) no longer exists -- `out` is the only remaining node that reads `t1`, so `consumers[t1]` drops from 2 to 1. The exact same rule that required materializing `t1` when it had 2 consumers now allows it to be inlined, because the rule was never about `t1` specifically -- it was always about the CONSUMER COUNT, and that count genuinely changed between the two graphs.
6. `elementwiseFusionPass()` needs to report which ORIGINAL node id each NEW node's value represents (`representativeOldId`), so that later code (Section 13.3's byte-counting) can look up the right element count for each new node without re-inferring shapes for a node kind (`FusedElementwise`) that Chapter 6's own `inferShapes()` was never taught to handle. That side table is genuinely useful ANALYSIS information, but Chapter 8's `TransformPass` signature -- `Graph(const Graph&)` -- has no room for it; a thin wrapper, `elementwiseFusionTransform()`, discards the side table and returns just the `Graph`, satisfying that exact signature for use inside `runPasses()`.
7. The 112-byte figure assumed the ENTIRE graph -- including `t1` -- could be folded into one single kernel, without checking whether that assumption would still be correct. It isn't: `t1` genuinely has two consumers (`t2` and `t3`), so any pass that keeps the graph's answer correct MUST either materialize `t1` separately (what this pass does) or duplicate its computation into two different fused kernels (introducing real risk of the two copies disagreeing). 112 bytes was never an achievable target for THIS graph -- it was a ceiling computed without that constraint in view.
8. Without that check, a value referenced twice within the SAME fused node's own body (for example, `t1` being read by both the `Mul` step and the `ReLU` step in Section 13.2's diamond) would be resolved by recursing into it a SECOND time, appending a SECOND, duplicate `FusedStep` computing the exact same thing again -- wasting a step and, if that operand were itself external, potentially double-counting it in `externalInputs` too. The `stepIndexByOldId` check is what lets a value used twice within one fused body still be computed (and read from memory, if external) exactly once.

---

**Sources cited in this chapter:**

None new. This chapter's own `OpKind::FusedElementwise` design is original to this book, though its "no intermediate storage materialized" shape is explicitly modeled on XLA's own real fusion computations, already cited directly to openxla.org's own documentation in Chapter 3.
