// Chapter 11: Algebraic Simplification
// 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline.cpp
//
// Section 11.3 -- constantFoldPass() (9.1), commonSubexpressionEliminationPass()
// (10.1), algebraicSimplificationPass() (11.1/11.2), and
// deadCodeEliminationPass() (9.2), all four run back to back through
// Chapter 8's own, still completely unmodified runPasses(). Four
// TransformPasses, from three different chapters, closing out Part 2.
//
// PART A -- a control measurement, the same "prove it, don't assert it"
// discipline Section 9.3 and Section 10.3 both used for their own
// order-matters claims: algebraicSimplificationPass() run ALONE on the
// ORIGINAL (unfolded) graph below finds nothing to simplify at the one
// spot that matters, because the relevant operand is still an Add node,
// not yet the Const(0) constant folding will turn it into.
//
// PART B -- the full four-pass pipeline. The graph is built so that
// constant folding creates a zero, common subexpression elimination
// merges a literal duplicate, algebraic simplification uses that zero
// to discard an entire otherwise-unrelated subexpression, and dead code
// elimination cleans up everything that discarding left behind --
// shrinking a real 13-node graph down to 4, with evaluate() confirming
// the answer never changed.
//
// PART C -- a compact, deliberately separate demonstration of a real
// hazard: when algebraic simplification happens to elide the LITERAL
// last node a graph ever added, Chapter 9's own "the last node is the
// graph's own output" convention -- which deadCodeEliminationPass()
// depends on completely -- silently ends up pointing at the wrong node.
// This is not a new bug introduced by this chapter; it is Section 9.2's
// own honestly-stated limitation (Worked Solution #4: "a graph needing
// ... an output that is not the most recently created node, could not
// be expressed under this convention at all") coming true for a
// concrete, demonstrable reason. Part B's own graph was deliberately
// built to avoid this exact trap; Part C shows precisely why that care
// was necessary.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline.cpp -o 023_folding_cse_simplification_and_dce_closing_part_2s_pipeline
// Run:     ./023_folding_cse_simplification_and_dce_closing_part_2s_pipeline
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

// ==================== algebraicSimplificationPass() (from Section 11.1/11.2, unchanged) ====================

static Graph algebraicSimplificationPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("algebraicSimplificationPass: input graph is not acyclic");

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
            if (inNode->op == OpKind::ReLU) {
                newValue = in;
            } else {
                newValue = result.addUnary(OpKind::ReLU, in, n->debugName);
            }
        } else { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            const Node* lhsNode = result.node(lhs.nodeId);
            const Node* rhsNode = result.node(rhs.nodeId);
            bool lhsIsZero = (lhsNode->op == OpKind::Const && lhsNode->constValue == 0.0f);
            bool rhsIsZero = (rhsNode->op == OpKind::Const && rhsNode->constValue == 0.0f);
            bool lhsIsOne  = (lhsNode->op == OpKind::Const && lhsNode->constValue == 1.0f);
            bool rhsIsOne  = (rhsNode->op == OpKind::Const && rhsNode->constValue == 1.0f);

            if (n->op == OpKind::Add) {
                if (lhsIsZero) newValue = rhs;
                else if (rhsIsZero) newValue = lhs;
                else newValue = result.addBinary(OpKind::Add, lhs, rhs, n->debugName);
            } else { // Mul
                if (lhsIsZero) newValue = lhs;
                else if (rhsIsZero) newValue = rhs;
                else if (lhsIsOne) newValue = rhs;
                else if (rhsIsOne) newValue = lhs;
                else newValue = result.addBinary(OpKind::Mul, lhs, rhs, n->debugName);
            }
        }

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
    // ---------------- PART A: control measurement ----------------
    printf("=== Section 11.3, Part A: algebraicSimplificationPass() ALONE, before any folding ===\n\n");

    // "z" folds to Const(0) under constantFoldPass() -- but here, BEFORE
    // folding runs, it is still a plain Add node. "waste" = mul(y, z)
    // cannot be recognized as "y * 0" while z is still an Add: the
    // simplification pass only ever checks whether an operand IS
    // already a Const, never what it would eventually evaluate to.
    Graph original;
    Value c1 = original.addConst(3.0f, "c1");
    Value c2 = original.addConst(-3.0f, "c2");
    Value z  = original.addBinary(OpKind::Add, c1, c2, "z");     // 3 + (-3) -- folds to Const(0)
    Value a  = original.addInput("a");
    Value b  = original.addInput("b");
    Value y  = original.addBinary(OpKind::Mul, a, b, "y");       // an otherwise-unused subexpression
    Value waste = original.addBinary(OpKind::Mul, y, z, "waste"); // y * z -- NOT foldable/simplifiable yet
    Value x1 = original.addInput("x1");
    Value x2 = original.addInput("x2");
    Value sum1 = original.addBinary(OpKind::Add, x1, x2, "sum1");
    Value sum2 = original.addBinary(OpKind::Add, x1, x2, "sum2"); // literal duplicate of sum1
    Value combined = original.addBinary(OpKind::Add, waste, sum1, "combined");
    original.addBinary(OpKind::Mul, combined, sum2, "out");

    Graph simplifyAloneOnOriginal = algebraicSimplificationPass(original);
    bool simplifyAloneFindsNothing = (simplifyAloneOnOriginal.size() == original.size());
    printf("size: %zu -> %zu\n", original.size(), simplifyAloneOnOriginal.size());
    printf("self-check: simplification alone finds NOTHING before folding runs (%s)\n\n",
           simplifyAloneFindsNothing ? "confirmed" : "MISMATCH");

    // ---------------- PART B: the full four-pass pipeline ----------------
    printf("=== Section 11.3, Part B: [constantFoldPass, commonSubexpressionEliminationPass, algebraicSimplificationPass, deadCodeEliminationPass] ===\n\n");

    Graph finalResult = runPasses(original, {
        {"constantFoldPass", constantFoldPass},
        {"commonSubexpressionEliminationPass", commonSubexpressionEliminationPass},
        {"algebraicSimplificationPass", algebraicSimplificationPass},
        {"deadCodeEliminationPass", deadCodeEliminationPass},
    });

    bool finalSizeCorrect = (finalResult.size() == 4);
    printf("self-check: the full pipeline reduced the graph from %zu nodes to %zu nodes (%s)\n",
           original.size(), finalResult.size(), finalSizeCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"a", 2.0f}, {"b", 7.0f}, {"x1", 3.0f}, {"x2", 4.0f}};
    float originalOut = evaluate(original, inputs).at("out");
    float finalOut = evaluate(finalResult, inputs).at("out");
    bool sameAnswer = (originalOut == finalOut) && (originalOut == 49.0f);  // (x1+x2)^2 = 7^2 = 49
    printf("self-check: evaluate(original, ...).out = %g, evaluate(final, ...).out = %g, both == 49 (%s)\n",
           originalOut, finalOut, sameAnswer ? "confirmed" : "MISMATCH");

    // ---------------- PART C: the last-node-elision hazard ----------------
    printf("\n=== Section 11.3, Part C: when simplification elides the LITERAL last node ===\n\n");

    // A tiny, deliberately minimal graph: "out" itself is a pure
    // additive-identity Add, and it is also the LAST node ever added.
    Graph tiny;
    Value tx = tiny.addInput("x");
    Value tzero = tiny.addConst(0.0f, "zero");
    tiny.addBinary(OpKind::Add, tx, tzero, "out");  // x + 0 -- IS the last node, and WILL be elided

    Graph tinySimplified = algebraicSimplificationPass(tiny);
    printf("Before: %zu nodes, last node is 'out'\n", tiny.size());
    printf("After:  %zu nodes\n", tinySimplified.size());
    printf("%s\n", printGraphAsSource(tinySimplified).c_str());

    // "out" (x + 0) elides to alias x directly -- x was already added
    // FIRST, so it is NOT the new graph's last node. The new graph's
    // actual last node is "zero" -- which is emphatically NOT what "out"
    // used to mean. Treating "the last node is the output" here, the
    // way deadCodeEliminationPass() always does, would silently select
    // the WRONG node.
    bool tinyShrankByOne = (tinySimplified.size() == tiny.size() - 1);
    const Node& tinyLastNode = *tinySimplified.nodes().back();
    bool lastNodeIsWrong = (tinyLastNode.debugName == "zero");
    printf("self-check: the simplified graph's own LAST node is '%s', not the real output (%s)\n",
           tinyLastNode.debugName.c_str(), lastNodeIsWrong ? "confirmed -- this IS the hazard" : "MISMATCH");
    printf("self-check: the graph shrank by exactly one node, as expected, %zu -> %zu (%s)\n",
           tiny.size(), tinySimplified.size(), tinyShrankByOne ? "confirmed" : "MISMATCH");
    printf("(Part B's own graph was deliberately built so its final Mul survives un-elided,\n");
    printf("precisely to avoid the hazard this small graph exists purely to demonstrate.)\n");

    bool allOk = simplifyAloneFindsNothing && finalSizeCorrect && sameAnswer &&
                 tinyShrankByOne && lastNodeIsWrong;
    return allOk ? 0 : 1;
}
