// Chapter 4: Representing Tensors and Operations as a Graph
// 006_topological_order_and_cycle_detection.cpp
//
// Section 4.3 -- a graph must be acyclic, and how CUDA Hammer's own
// construction API guarantees that by design.
//
// Reuses File 005's own Value/Node/Graph classes unchanged (a real
// compiler's later stages never re-derive the earlier ones). The only
// new code here is a genuine topological sort (Kahn's algorithm) and an
// independent second check of its own output.
//
// CUDA Hammer's addInput/addConst/addUnary/addBinary API can only ever
// take Values that some EARLIER call already returned -- there is no way
// to reference a node that does not exist yet -- so a graph built
// through that API can never contain a cycle. This file proves that
// property two ways: first, by running Kahn's algorithm on a normally
// built graph and cross-checking its result with an independent,
// differently-implemented positional check; second, by deliberately
// reaching PAST the normal API (directly mutating a Node's own inputs
// vector, something ordinary use of Graph never does) to hand-construct
// one small graph that DOES contain a cycle, and confirming the same
// checker correctly reports it as invalid rather than silently
// producing a wrong order.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 006_topological_order_and_cycle_detection.cpp -o 006_topological_order_and_cycle_detection
// Run:     ./006_topological_order_and_cycle_detection
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>

// ==================== Value / Node / Graph (from File 005) ====================

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
    Node* mutableNode(int id) { return nodes_.at(static_cast<size_t>(id)).get(); }
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

// ======================= Kahn's algorithm (new in this file) =======================
//
// Repeatedly removes a node with in-degree 0 (no unprocessed producers
// left), appending it to the order and decrementing the in-degree of
// every node that reads it. If every node gets removed this way, the
// graph is acyclic and `order` is a valid topological order. If nodes
// remain stuck with in-degree > 0 once no more zero-in-degree nodes are
// left to remove, the graph contains a cycle -- ok is set to false.
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
        // Decrement the in-degree of every node that lists `id` as an input.
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

// Independent cross-check: for a claimed order to be valid, every node
// must appear AFTER every one of its own inputs' producer nodes. This is
// checked by position lookup, never by re-running Kahn's own in-degree
// bookkeeping -- a genuinely different method arriving at the same
// verdict, the same cross-check discipline Chapter 2 used (an
// independent AST evaluator, checked against the IR interpreter).
static bool independentlyValidatesOrder(const Graph& g, const std::vector<int>& order) {
    if (order.size() != g.size()) return false;
    std::map<int, int> position;
    for (size_t i = 0; i < order.size(); ++i) position[order[i]] = static_cast<int>(i);
    for (const auto& n : g.nodes()) {
        for (const Value& in : n->inputs) {
            if (position.at(in.nodeId) >= position.at(n->id)) return false;
        }
    }
    return true;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("    %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (!n->inputs.empty()) {
            printf(" <- ");
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) printf(", ");
                printf("%%%d", n->inputs[i].nodeId);
            }
        }
        printf("\n");
    }
}

int main() {
    printf("=== Section 4.3: topological order, and detecting a graph that has no valid one ===\n\n");

    // --- Part A: the normal diamond DAG from File 005, built ONLY through
    // the graph API -- this can never contain a cycle by construction. ---
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    g.addBinary(OpKind::Add, t2, t3, "out");

    TopoResult result = topologicalSort(g);
    printf("Part A -- valid DAG (%zu nodes, built only through the graph API):\n", g.size());
    printGraph(g);
    printf("  Kahn's algorithm order: ");
    for (int id : result.order) printf("%%%d ", id);
    printf("\n  Kahn's algorithm reports valid: %s\n", result.ok ? "yes" : "no");

    bool independentCheckA = independentlyValidatesOrder(g, result.order);
    printf("  independent positional cross-check agrees: %s\n\n",
           independentCheckA ? "confirmed" : "MISMATCH");

    // --- Part B: a SECOND, tiny graph, deliberately broken by reaching
    // past the normal API to hand-construct a cycle. This is never
    // something ordinary use of Graph can produce -- addBinary/addUnary
    // only ever accept Values that an earlier call already returned, so
    // a fresh Node cannot reference a Node that does not exist yet. The
    // only way to build a cycle at all is to add the nodes first and then
    // go back and mutate an EARLIER node's own inputs afterward, which is
    // exactly what happens below, purely to test that the checker itself
    // correctly rejects it. ---
    Graph broken;
    Value x = broken.addInput("x");             // node 0, no inputs yet
    Value y = broken.addUnary(OpKind::ReLU, x, "y"); // node 1, input: node 0
    // Reach past the API: force node 0 ("x") to also depend on node 1
    // ("y") after the fact -- x -> y -> x, a genuine cycle no normal
    // sequence of addInput/addUnary/addBinary calls could ever create.
    broken.mutableNode(x.nodeId)->inputs.push_back(y);

    TopoResult brokenResult = topologicalSort(broken);
    printf("Part B -- deliberately hand-broken graph (2 nodes, x -> y -> x cycle\n");
    printf("           forced by mutating node 0's inputs directly, bypassing the graph API):\n");
    printGraph(broken);
    printf("  Kahn's algorithm processed %zu of %zu nodes before getting stuck\n",
           brokenResult.order.size(), broken.size());
    printf("  Kahn's algorithm reports valid: %s\n", brokenResult.ok ? "yes" : "no");

    bool correctlyRejected = !brokenResult.ok;
    printf("\nself-check: the normal DAG is confirmed valid AND cross-checked (%s);\n",
           (result.ok && independentCheckA) ? "confirmed" : "MISMATCH");
    printf("the hand-broken cyclic graph is correctly REJECTED, not silently misordered (%s)\n",
           correctlyRejected ? "confirmed" : "MISMATCH");

    return (result.ok && independentCheckA && correctlyRejected) ? 0 : 1;
}
