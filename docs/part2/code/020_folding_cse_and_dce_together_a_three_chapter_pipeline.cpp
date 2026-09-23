// Chapter 10: Common Subexpression Elimination for Tensor Graphs
// 020_folding_cse_and_dce_together_a_three_chapter_pipeline.cpp
//
// Section 10.3 -- constantFoldPass() (9.1), commonSubexpressionEliminationPass()
// (10.1), and deadCodeEliminationPass() (9.2), run back to back through
// Chapter 8's own, completely unmodified runPasses(). Three
// TransformPasses, from three different chapters, none of them aware
// the other two exist, composed through infrastructure that has not
// changed one line since Chapter 8 first wrote it.
//
// This file's real point is a specific instance of a question Section
// 9.3's own [COMMON TRAP] raised and deliberately left open: "could
// running this again find more?" There, the question was about folding
// and DCE alternating. Here it shows up from a genuinely different
// angle -- CAN FOLDING EXPOSE A CSE OPPORTUNITY CSE COULD NOT HAVE
// FOUND ON ITS OWN? The graph below is built so the answer is yes, and
// proves it the same way Section 9.3 proved its own order-matters
// claim: not by assuring the reader in prose, but by actually running
// commonSubexpressionEliminationPass() ALONE on the ORIGINAL (unfolded)
// graph and showing, by a real measurement, that it finds nothing.
//
// The graph: two constants, 2 and 3, get added together and named "p".
// A THIRD, separately-created constant, "c3", is independently given
// the literal value 5 -- which happens to be exactly what "p" computes,
// but "p" is an Add node and "c3" is a Const node BEFORE folding runs,
// so cseKey() (Section 10.1) gives them completely different keys:
// commonSubexpressionEliminationPass() has no arithmetic of its own,
// and cannot know an Add node's eventual VALUE without evaluating it.
// Only after constantFoldPass() turns "p" into an actual Const(5) node
// does it share a key with "c3" -- and only then can CSE recognize them
// as the same thing.
//
// Once folding and CSE have both run, "c1" and "c2" (the two constants
// that fed "p") are no longer referenced by anything -- "p" itself was
// replaced by the CSE merge, not kept -- leaving them as genuinely NEW
// dead code that only deadCodeEliminationPass() can clean up. The full
// pipeline is: 8 nodes -> (fold) 8 nodes -> (CSE) 6 nodes -> (DCE) 4
// nodes, checked at every step, with evaluate() confirming the exact
// same answer before and after all three passes.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 020_folding_cse_and_dce_together_a_three_chapter_pipeline.cpp -o 020_folding_cse_and_dce_together_a_three_chapter_pipeline
// Run:     ./020_folding_cse_and_dce_together_a_three_chapter_pipeline
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

// ==================== evaluate() (from Chapter 9, unchanged) ====================

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
        } else { // ReLU
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== constantFoldPass() (from Section 9.1, unchanged) ====================

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
        } else { // Add or Mul
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

// ==================== commonSubexpressionEliminationPass() (from Section 10.1, unchanged) ====================

static std::string cseKey(const Node* n, const std::map<int, Value>& oldIdToNewValue) {
    char buf[128];
    switch (n->op) {
        case OpKind::Input:
            return "input:" + n->debugName;
        case OpKind::Const:
            snprintf(buf, sizeof(buf), "const:%.9g", n->constValue);
            return buf;
        case OpKind::ReLU: {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            snprintf(buf, sizeof(buf), "relu:%d", in.nodeId);
            return buf;
        }
        default: { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            snprintf(buf, sizeof(buf), "%s:%d,%d", n->op == OpKind::Add ? "add" : "mul", lhs.nodeId, rhs.nodeId);
            return buf;
        }
    }
}

static Graph commonSubexpressionEliminationPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("commonSubexpressionEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    std::map<std::string, Value> keyToExistingValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string key = cseKey(n, oldIdToNewValue);

        auto found = keyToExistingValue.find(key);
        if (found != keyToExistingValue.end()) {
            oldIdToNewValue[oldId] = found->second;
            continue;
        }

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
        keyToExistingValue[key] = newValue;
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ==================== deadCodeEliminationPass() (from Section 9.2, unchanged) ====================

static std::set<int> computeLiveNodeIds(const Graph& g, int rootId) {
    std::set<int> live;
    std::deque<int> worklist{rootId};
    while (!worklist.empty()) {
        int id = worklist.front();
        worklist.pop_front();
        if (live.count(id)) continue;
        live.insert(id);
        for (const Value& in : g.node(id)->inputs) {
            worklist.push_back(in.nodeId);
        }
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

// ==================== PassManager (from Section 8.3, unchanged) ====================

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
            throw std::runtime_error("runPasses: pass '" + passes[0].name +
                                      "' produced an invalid (non-acyclic) graph");
        }
    }
    printf("--- after pass '%s' (%zu nodes) ---\n%s\n", passes[0].name.c_str(), current.size(),
           printGraphAsSource(current).c_str());

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& p = passes[i];
        Graph next = p.fn(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + p.name +
                                      "' produced an invalid (non-acyclic) graph");
        }
        printf("--- after pass '%s' (%zu nodes) ---\n%s\n", p.name.c_str(), next.size(),
               printGraphAsSource(next).c_str());
        current = std::move(next);
    }
    return current;
}

int main() {
    printf("=== Section 10.3: folding creates a CSE opportunity; CSE creates new dead code ===\n\n");

    // c1(2) + c2(3) -> "p", which folds to Const(5) -- the exact same
    // VALUE as "c3", an independently-created Const(5). Before folding,
    // "p" is an Add node and "c3" is a Const node: different cseKey()s,
    // no merge possible. "r1" and "r2" each depend on one of them plus
    // a real runtime input "x", so neither folds away entirely --
    // keeping "p" and "c3" both alive as real operands for CSE to see.
    Graph original;
    Value c1 = original.addConst(2.0f, "c1");
    Value c2 = original.addConst(3.0f, "c2");
    Value p  = original.addBinary(OpKind::Add, c1, c2, "p");   // 2 + 3 -- folds to Const(5)
    Value c3 = original.addConst(5.0f, "c3");                  // literal duplicate of p's eventual value
    Value x  = original.addInput("x");
    Value r1 = original.addBinary(OpKind::Add, p, x, "r1");    // p + x
    Value r2 = original.addBinary(OpKind::Add, c3, x, "r2");   // c3 + x
    original.addBinary(OpKind::Add, r1, r2, "out");            // r1 + r2

    // First, the control measurement: run CSE ALONE on the UNFOLDED
    // original graph. "p" (an Add) and "c3" (a Const) have different
    // keys regardless of what they'd eventually compute -- CSE has no
    // arithmetic of its own, so it cannot see they are equal in VALUE.
    Graph cseAloneOnOriginal = commonSubexpressionEliminationPass(original);
    bool cseAloneFindsNothing = (cseAloneOnOriginal.size() == original.size());
    printf(">>> Control: commonSubexpressionEliminationPass() ALONE, before any folding\n\n");
    printf("size: %zu -> %zu\n", original.size(), cseAloneOnOriginal.size());
    printf("self-check: CSE alone finds NOTHING to merge in the unfolded graph (%s)\n\n",
           cseAloneFindsNothing ? "confirmed" : "MISMATCH");

    printf(">>> Pipeline: [constantFoldPass, commonSubexpressionEliminationPass, deadCodeEliminationPass]\n\n");
    Graph finalResult = runPasses(original, {
        {"constantFoldPass", constantFoldPass},
        {"commonSubexpressionEliminationPass", commonSubexpressionEliminationPass},
        {"deadCodeEliminationPass", deadCodeEliminationPass},
    });

    bool finalSizeCorrect = (finalResult.size() == 4);
    printf("self-check: the full pipeline reduced the graph from %zu nodes to %zu nodes (%s)\n",
           original.size(), finalResult.size(), finalSizeCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"x", 6.5f}};
    float originalOut = evaluate(original, inputs).at("out");
    float finalOut = evaluate(finalResult, inputs).at("out");
    bool sameAnswer = (originalOut == finalOut);
    printf("self-check: evaluate(original, x=6.5).out = %g, evaluate(final, x=6.5).out = %g (%s)\n",
           originalOut, finalOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = cseAloneFindsNothing && finalSizeCorrect && sameAnswer;
    return allOk ? 0 : 1;
}
