// Chapter 9: Constant Folding and Dead Code Elimination
// 017_folding_and_dce_together_through_chapter_8s_pass_manager.cpp
//
// Section 9.3 -- running constantFoldPass and deadCodeEliminationPass
// TOGETHER, through Chapter 8's own unmodified runPasses(), and seeing
// why the ORDER genuinely matters: folding does not just simplify
// existing computation, it actively CREATES new dead code, which only
// becomes visible -- and only gets cleaned up -- once DCE runs AFTER it.
//
// Both of this chapter's own passes already have the exact signature
// Chapter 8's TransformPass requires, `Graph(const Graph&)` -- neither
// one needed any change at all to plug directly into Chapter 8's own
// runPasses(). This file is the payoff of that design choice: Part 2's
// first real, multi-pass optimization pipeline, built entirely by
// composing two independently-tested passes through infrastructure
// that was finished two chapters ago.
//
// Reuses File 015's constantFoldPass() and evaluate(), File 016's
// deadCodeEliminationPass(), Chapter 7's printGraphAsSource(), and
// Chapter 8's runPasses()/NamedPass -- all completely unchanged.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 017_folding_and_dce_together_through_chapter_8s_pass_manager.cpp -o 017_folding_and_dce_together_through_chapter_8s_pass_manager
// Run:     ./017_folding_and_dce_together_through_chapter_8s_pass_manager
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

// ==================== Value / Node / Graph (from Chapter 4) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU };

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;
    float constValue = 0.0f;
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

// ==================== printGraphAsSource() (from Chapter 7, unchanged) ====================

static std::string opKindToSourceKeyword(OpKind op) {
    switch (op) {
        case OpKind::Input: return "input";
        case OpKind::Const: return "const";
        case OpKind::Add:   return "add";
        case OpKind::Mul:   return "mul";
        default:            return "relu";
    }
}

static std::string printGraphAsSource(const Graph& g) {
    std::string out;
    for (const auto& n : g.nodes()) {
        out += n->debugName + " = " + opKindToSourceKeyword(n->op) + "(";
        if (n->op == OpKind::Const) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", n->constValue);
            out += buf;
        } else {
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                out += g.node(n->inputs[i].nodeId)->debugName;
            }
        }
        out += ")\n";
    }
    return out;
}

// ==================== evaluate() (from File 015, unchanged) ====================

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
        } else {
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== TransformPass / constantFoldPass() (from Section 9.1) ====================

using TransformPass = std::function<Graph(const Graph&)>;

static Graph constantFoldPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("constantFoldPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        Value newValue;
        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->op == OpKind::ReLU) {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            const Node* inNode = result.node(in.nodeId);
            if (inNode->op == OpKind::Const) {
                newValue = result.addConst(std::max(0.0f, inNode->constValue), n->debugName);
            } else {
                newValue = result.addUnary(OpKind::ReLU, in, n->debugName);
            }
        } else {
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            const Node* lhsNode = result.node(lhs.nodeId);
            const Node* rhsNode = result.node(rhs.nodeId);
            if (lhsNode->op == OpKind::Const && rhsNode->op == OpKind::Const) {
                float folded = (n->op == OpKind::Add) ? (lhsNode->constValue + rhsNode->constValue)
                                                        : (lhsNode->constValue * rhsNode->constValue);
                newValue = result.addConst(folded, n->debugName);
            } else {
                newValue = result.addBinary(n->op, lhs, rhs, n->debugName);
            }
        }
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ==================== deadCodeEliminationPass() (from Section 9.2) ====================

static std::set<int> computeLiveNodeIds(const Graph& g, int rootId) {
    std::set<int> live;
    std::deque<int> worklist{rootId};
    while (!worklist.empty()) {
        int id = worklist.front();
        worklist.pop_front();
        if (live.count(id)) continue;
        live.insert(id);
        for (const Value& in : g.node(id)->inputs) worklist.push_back(in.nodeId);
    }
    return live;
}

static Graph deadCodeEliminationPass(const Graph& g) {
    if (g.size() == 0) throw std::runtime_error("deadCodeEliminationPass: an empty graph has no output node");
    int rootId = static_cast<int>(g.size()) - 1;
    std::set<int> live = computeLiveNodeIds(g, rootId);

    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("deadCodeEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
        if (!live.count(oldId)) continue;
        const Node* n = g.node(oldId);
        Value newValue;
        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->inputs.size() == 1) {
            newValue = result.addUnary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId), n->debugName);
        } else {
            newValue = result.addBinary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId),
                                         oldIdToNewValue.at(n->inputs[1].nodeId), n->debugName);
        }
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ============================== Section 9.3: runPasses() (from Chapter 8, unchanged) ==============================

struct NamedPass {
    std::string name;
    TransformPass fn;
};

static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass (%zu nodes) ---\n%s\n", initial.size(), printGraphAsSource(initial).c_str());

    Graph current = passes[0].fn(initial);
    {
        TopoResult topo = topologicalSort(current);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + passes[0].name + "' produced an invalid graph");
        }
    }
    printf("--- after pass '%s' (%zu nodes) ---\n%s\n",
           passes[0].name.c_str(), current.size(), printGraphAsSource(current).c_str());

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& p = passes[i];
        Graph next = p.fn(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + p.name + "' produced an invalid graph");
        }
        printf("--- after pass '%s' (%zu nodes) ---\n%s\n",
               p.name.c_str(), next.size(), printGraphAsSource(next).c_str());
        current = std::move(next);
    }
    return current;
}

int main() {
    printf("=== Section 9.3: constant folding creates dead code; DCE only helps if it runs AFTER ===\n\n");

    // File 015's own graph: a foldable constant chain (c1+c2)*c3,
    // combined with a real runtime input x. "out" is this graph's own
    // designated output (the last node).
    Graph original;
    Value c1 = original.addConst(2.0f, "c1");
    Value c2 = original.addConst(3.0f, "c2");
    Value c3 = original.addConst(4.0f, "c3");
    Value t1 = original.addBinary(OpKind::Add, c1, c2, "t1");
    Value t2 = original.addBinary(OpKind::Mul, t1, c3, "t2");
    Value x  = original.addInput("x");
    original.addBinary(OpKind::Add, t2, x, "out");

    printf(">>> Pipeline: [constantFoldPass, deadCodeEliminationPass]\n\n");
    Graph finalResult = runPasses(original, {
        {"constantFoldPass", constantFoldPass},
        {"deadCodeEliminationPass", deadCodeEliminationPass},
    });

    // After folding, t2 becomes a plain Const(20) with NO inputs at
    // all -- which means c1, c2, c3, AND the old t1 are now reachable
    // from NOTHING "out" depends on, even though constantFoldPass
    // itself still faithfully re-emitted all of them (it rebuilds
    // EVERY node, it does not know or care which ones anything else
    // still needs -- that is precisely DCE's job, not folding's).
    printf("self-check: the graph shrank from %zu nodes to %zu nodes -- folding severed the\n",
           original.size(), finalResult.size());
    printf("dependency chain back to c1/c2/c3/t1, and DCE then dropped all four (%s)\n",
           (finalResult.size() == 3) ? "confirmed" : "MISMATCH");

    bool onlyThreeNamesRemain = finalResult.size() == 3;
    bool namesAreOutT2X = onlyThreeNamesRemain; // checked precisely below
    if (onlyThreeNamesRemain) {
        std::set<std::string> remainingNames;
        for (const auto& n : finalResult.nodes()) remainingNames.insert(n->debugName);
        namesAreOutT2X = remainingNames == std::set<std::string>{"out", "t2", "x"};
    }
    printf("self-check: exactly {t2, x, out} remain -- c1, c2, c3, and the original t1 are\n");
    printf("all gone (%s)\n", namesAreOutT2X ? "confirmed" : "MISMATCH");

    // The claim that actually matters, same as File 015: two whole
    // optimization passes later, the graph still computes the exact
    // same answer for a real input.
    std::map<std::string, float> inputs = {{"x", 6.5f}};
    float originalOut = evaluate(original, inputs).at("out");
    float finalOut = evaluate(finalResult, inputs).at("out");
    bool sameAnswer = (originalOut == finalOut);
    printf("self-check: evaluate(original, x=6.5).out = %g, evaluate(final, x=6.5).out = %g (%s)\n",
           originalOut, finalOut, sameAnswer ? "confirmed" : "MISMATCH");

    // The order-matters claim: running DCE BEFORE folding would find
    // EVERY node live (c1/c2/c3/t1 all still genuinely feed "out" in
    // the UNFOLDED graph), so it would remove nothing at all -- proving
    // the size reduction above specifically requires folding to run
    // FIRST, not just requires both passes to run in some order.
    Graph dceFirst = deadCodeEliminationPass(original);
    bool dceAloneRemovedNothing = (dceFirst.size() == original.size());
    printf("\nself-check: running deadCodeEliminationPass() ALONE, before any folding, removes\n");
    printf("NOTHING (%zu -> %zu nodes) -- every node is genuinely live in the UNFOLDED graph (%s)\n",
           original.size(), dceFirst.size(), dceAloneRemovedNothing ? "confirmed" : "MISMATCH");

    bool allOk = (finalResult.size() == 3) && namesAreOutT2X && sameAnswer && dceAloneRemovedNothing;
    return allOk ? 0 : 1;
}
