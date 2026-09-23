// Chapter 11: Algebraic Simplification
// 021_algebraic_simplification_the_additive_identity.cpp
//
// Section 11.1 -- algebraic simplification, this book's FOURTH
// TransformPass, plugging into the exact same Chapter 8 machinery
// constantFoldPass() (9.1), deadCodeEliminationPass() (9.2), and
// commonSubexpressionEliminationPass() (10.1) already plug into, with
// zero changes to that machinery required, for the fourth chapter in a
// row.
//
// The three passes so far each answer a different question: "is this
// computation's answer already fully known?" (folding), "does anything
// still need this?" (dead code elimination), "has this exact
// computation already been done somewhere else?" (CSE). Algebraic
// simplification asks a fourth: "does basic arithmetic already tell me
// this computation is POINTLESS, regardless of what its operand turns
// out to be?" Adding 0 to ANY number -- known or not, a runtime Input
// or a compile-time Const -- always returns that same number, unchanged.
// A graph can easily contain `add(x, 0)` where `x` is a genuine runtime
// value nobody could fold away -- but the ADDITION itself is still
// provably pointless, and this pass is the one that notices.
//
// algebraicSimplificationPass() is structurally the twin of Section
// 9.1's constantFoldPass(): it rebuilds the graph in topological order,
// and at Add and Mul nodes, checks its own (already-rebuilt) operands
// against a small table of algebraic identities. Where constantFoldPass()
// responds to a match by COMPUTING a brand new answer and emitting a
// fresh Const node, this pass responds by pointing straight at an
// operand THAT ALREADY EXISTS -- no computation, no new node, not even
// a new Const. This section covers the additive identity element: `x +
// 0` and `0 + x` both simplify to plain `x`, for any `x` at all.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 021_algebraic_simplification_the_additive_identity.cpp -o 021_algebraic_simplification_the_additive_identity
// Run:     ./021_algebraic_simplification_the_additive_identity
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

// ============================== Section 11.1: algebraicSimplificationPass() (new) ==============================
//
// This file only exercises the ADD identity element (0 is what "doing
// nothing" means for addition). Section 11.2 adds Mul's own identity
// (1), Mul's absorbing element (0), and ReLU's idempotence -- all
// through the exact same function, reproduced unchanged in every file
// in this chapter, the same "one real pass, reused verbatim" discipline
// every earlier chapter in Part 2 has followed.
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
                // relu(relu(x)) -> relu(x): applying ReLU to something
                // ReLU already clamped changes nothing further.
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
                if (lhsIsZero) newValue = rhs;          // 0 + x -> x
                else if (rhsIsZero) newValue = lhs;     // x + 0 -> x
                else newValue = result.addBinary(OpKind::Add, lhs, rhs, n->debugName);
            } else { // Mul
                if (lhsIsZero) newValue = lhs;          // 0 * x -> 0 (the zero itself)
                else if (rhsIsZero) newValue = rhs;     // x * 0 -> 0
                else if (lhsIsOne) newValue = rhs;      // 1 * x -> x
                else if (rhsIsOne) newValue = lhs;      // x * 1 -> x
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
    printf("=== Section 11.1: algebraicSimplificationPass(), the additive identity element ===\n\n");

    // t1 = x + 0, t2 = 0 + x -- both should simplify to just x, pointing
    // straight at x's own node, with NO new node built at all. t3 = x +
    // y is the control: neither operand is Const(0), so it must survive
    // untouched. t12 and out both combine already-simplified operands,
    // so the pass has to work correctly through TWO more levels after
    // the leaf-level simplification, not just at the very first Add.
    Graph g;
    Value x = g.addInput("x");
    Value y = g.addInput("y");
    Value zero = g.addConst(0.0f, "zero");
    Value t1 = g.addBinary(OpKind::Add, x, zero, "t1");     // x + 0 -> x
    Value t2 = g.addBinary(OpKind::Add, zero, x, "t2");     // 0 + x -> x
    Value t3 = g.addBinary(OpKind::Add, x, y, "t3");        // control: no identity applies
    Value t12 = g.addBinary(OpKind::Add, t1, t2, "t12");
    g.addBinary(OpKind::Add, t12, t3, "out");

    printf("Graph with two additive-identity Adds ('t1', 't2') and one real Add ('t3'):\n");
    printGraph(g);

    Graph simplified = algebraicSimplificationPass(g);
    printf("\nAfter algebraicSimplificationPass():\n");
    printGraph(simplified);

    bool twoNodesDropped = (simplified.size() == g.size() - 2);
    bool t1Gone = true, t2Gone = true;
    for (const auto& n : simplified.nodes()) {
        if (n->debugName == "t1") t1Gone = false;
        if (n->debugName == "t2") t2Gone = false;
    }
    printf("\nself-check: exactly two identity Adds were dropped, %zu -> %zu (%s)\n",
           g.size(), simplified.size(), twoNodesDropped ? "confirmed" : "MISMATCH");
    printf("self-check: neither 't1' nor 't2' appears anywhere in the simplified graph (%s)\n",
           (t1Gone && t2Gone) ? "confirmed" : "MISMATCH");

    const Node* t3InSimplified = findNodeByName(simplified, "t3");
    bool t3StillAdd = (t3InSimplified->op == OpKind::Add);
    printf("self-check: 't3' (x + y, no identity applies) is still a real Add (%s)\n",
           t3StillAdd ? "confirmed" : "MISMATCH");

    const Node* t12InSimplified = findNodeByName(simplified, "t12");
    bool t12BothOperandsAreX = (t12InSimplified->inputs[0].nodeId == t12InSimplified->inputs[1].nodeId);
    printf("self-check: 't12' now reads the SAME node (x) as both of its own operands (%s)\n",
           t12BothOperandsAreX ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"x", 4.0f}, {"y", 9.0f}};
    float originalOut = evaluate(g, inputs).at("out");
    float simplifiedOut = evaluate(simplified, inputs).at("out");
    bool sameAnswer = (originalOut == simplifiedOut);
    printf("self-check: evaluate(original, x=4,y=9).out = %g, evaluate(simplified, ...).out = %g (%s)\n",
           originalOut, simplifiedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = twoNodesDropped && t1Gone && t2Gone && t3StillAdd && t12BothOperandsAreX && sameAnswer;
    return allOk ? 0 : 1;
}
