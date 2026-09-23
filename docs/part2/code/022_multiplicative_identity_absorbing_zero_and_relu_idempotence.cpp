// Chapter 11: Algebraic Simplification
// 022_multiplicative_identity_absorbing_zero_and_relu_idempotence.cpp
//
// Section 11.2 -- the same algebraicSimplificationPass() from Section
// 11.1, reproduced unchanged, exercising the identities Section 11.1
// did not: Mul's own identity element (`x * 1` and `1 * x` both
// simplify to plain `x`, the multiplicative twin of Section 11.1's
// additive-zero identity), Mul's ABSORBING element (`x * 0` and `0 * x`
// both simplify to plain `0` -- a genuinely different kind of identity
// from the other three: it does not return one of the two operands
// unchanged, it returns the CONSTANT operand itself, discarding the
// other operand -- and whatever produced that other operand -- entirely,
// regardless of what it was), and ReLU's idempotence (`relu(relu(x))`
// simplifies to plain `relu(x)`, since ReLU applied twice clamps
// nothing that applying it once didn't already clamp).
//
// The graph this file builds is deliberately busy: it is built so that
// EVERY identity this pass knows fires at least once, and so that
// several of those firings CASCADE into further firings, discovered in
// the very same topological walk -- the identical "check the new graph,
// not the old one" discipline that let constantFoldPass() (9.1) fold an
// entire chain in one pass and commonSubexpressionEliminationPass()
// (10.1) discover a multi-level merge in one pass, now shown a third
// time for a third kind of pass. Two Mul-by-zero nodes both collapse to
// the SAME shared zero constant; an Add that then combines those two
// zero-aliases collapses AGAIN, to that same zero; and an Add that then
// combines THAT zero-alias with something else collapses a THIRD time,
// to whatever that something else was. Three cascade levels, one pass.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 022_multiplicative_identity_absorbing_zero_and_relu_idempotence.cpp -o 022_multiplicative_identity_absorbing_zero_and_relu_idempotence
// Run:     ./022_multiplicative_identity_absorbing_zero_and_relu_idempotence
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
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

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input: return "Input";
        case OpKind::Const: return "Const";
        case OpKind::Add:   return "Add";
        case OpKind::Mul:   return "Mul";
        default:            return "ReLU";
    }
}

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

static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

// ==================== algebraicSimplificationPass() (from Section 11.1, unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;

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

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) printf(" [value=%g]", n->constValue);
        printf("\n");
    }
}

int main() {
    printf("=== Section 11.2: multiplicative identity, absorbing zero, and ReLU idempotence, cascading in one pass ===\n\n");

    Graph g;
    Value x = g.addInput("x");
    Value y = g.addInput("y");
    Value one = g.addConst(1.0f, "one");
    Value zero = g.addConst(0.0f, "zero");
    Value t1 = g.addBinary(OpKind::Mul, x, one, "t1");     // x * 1 -> x
    Value t2 = g.addBinary(OpKind::Mul, one, x, "t2");     // 1 * x -> x
    Value t3 = g.addBinary(OpKind::Mul, x, zero, "t3");    // x * 0 -> 0
    Value t4 = g.addBinary(OpKind::Mul, zero, y, "t4");    // 0 * y -> 0
    Value t5 = g.addBinary(OpKind::Mul, x, y, "t5");       // control: no identity applies
    Value r1 = g.addUnary(OpKind::ReLU, x, "r1");
    Value r2 = g.addUnary(OpKind::ReLU, r1, "r2");         // relu(relu(x)) -> relu(x)
    Value s1 = g.addBinary(OpKind::Add, t1, t2, "s1");     // CASCADE level 1: becomes add(x, x)
    Value s2 = g.addBinary(OpKind::Add, t3, t4, "s2");     // CASCADE level 1: becomes add(zero, zero) -- itself an identity!
    Value s3 = g.addBinary(OpKind::Add, s1, s2, "s3");     // CASCADE level 2: becomes add(s1, zero) -- itself an identity!
    Value s4 = g.addBinary(OpKind::Add, t5, r2, "s4");     // becomes add(t5, r1)
    g.addBinary(OpKind::Add, s3, s4, "out");                // becomes add(s1, s4)

    printf("Graph exercising every identity this pass knows, with three cascade levels:\n");
    printGraph(g);

    Graph simplified = algebraicSimplificationPass(g);
    printf("\nAfter algebraicSimplificationPass():\n");
    printGraph(simplified);

    // Six nodes elided: t1, t2 (mul identity), t3, t4 (mul absorbing
    // zero), r2 (ReLU idempotence), s2 AND s3 (both cascades). 16 -> 9.
    bool sizeCorrect = (simplified.size() == 9);
    printf("\nself-check: seven nodes were elided across three cascade levels, %zu -> %zu (%s)\n",
           g.size(), simplified.size(), sizeCorrect ? "confirmed" : "MISMATCH");

    bool allEludedNamesGone = true;
    for (const char* name : {"t1", "t2", "t3", "t4", "r2", "s2", "s3"}) {
        for (const auto& n : simplified.nodes()) {
            if (n->debugName == name) allEludedNamesGone = false;
        }
    }
    printf("self-check: none of 't1','t2','t3','t4','r2','s2','s3' appear in the simplified graph (%s)\n",
           allEludedNamesGone ? "confirmed" : "MISMATCH");

    const Node* t5InSimplified = findNodeByName(simplified, "t5");
    const Node* r1InSimplified = findNodeByName(simplified, "r1");
    const Node* s1InSimplified = findNodeByName(simplified, "s1");
    const Node* s4InSimplified = findNodeByName(simplified, "s4");
    const Node* outInSimplified = findNodeByName(simplified, "out");
    bool survivorsCorrect =
        (t5InSimplified->op == OpKind::Mul) &&
        (r1InSimplified->op == OpKind::ReLU) &&
        (s1InSimplified->op == OpKind::Add && s1InSimplified->inputs[0].nodeId == s1InSimplified->inputs[1].nodeId) &&
        (s4InSimplified->op == OpKind::Add &&
         s4InSimplified->inputs[0].nodeId == t5InSimplified->id &&
         s4InSimplified->inputs[1].nodeId == r1InSimplified->id) &&
        (outInSimplified->op == OpKind::Add &&
         outInSimplified->inputs[0].nodeId == s1InSimplified->id &&
         outInSimplified->inputs[1].nodeId == s4InSimplified->id);
    printf("self-check: 't5','r1','s1','s4','out' all survive as real nodes, correctly rewired (%s)\n",
           survivorsCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"x", 4.0f}, {"y", 5.0f}};
    float originalOut = evaluate(g, inputs).at("out");
    float simplifiedOut = evaluate(simplified, inputs).at("out");
    bool sameAnswer = (originalOut == simplifiedOut);
    printf("self-check: evaluate(original, x=4,y=5).out = %g, evaluate(simplified, ...).out = %g (%s)\n",
           originalOut, simplifiedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = sizeCorrect && allEludedNamesGone && survivorsCorrect && sameAnswer;
    return allOk ? 0 : 1;
}
