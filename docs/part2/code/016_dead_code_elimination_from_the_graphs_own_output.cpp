// Chapter 9: Constant Folding and Dead Code Elimination
// 016_dead_code_elimination_from_the_graphs_own_output.cpp
//
// Section 9.2 -- dead code elimination, the first real use of Chapter
// 8's own "rebuild rather than mutate" design, put to exactly the
// purpose Chapter 8 built it for: dropping nodes nothing actually
// depends on.
//
// Dead code elimination needs one thing Chapter 4's Graph has never
// had a concept of: an actual OUTPUT. Every example graph in this book
// so far has ended with a node named "out," by convention, but nothing
// in Graph itself has ever recorded "this is the value the caller
// actually cares about" -- Graph is just a set of nodes and edges.
// Without knowing which node's value the graph is FOR, "is this node
// dead" has no answer at all: every node in a graph with no designated
// output is trivially "used by nothing outside the graph," which would
// make deadCodeEliminationPass delete everything.
//
// This file picks the simplest possible convention that fits every
// example graph this book has built so far: the LAST node added to the
// graph (the highest id) is treated as the graph's own output. This is
// a real simplification, stated plainly rather than hidden -- a
// compiler for a language with multiple return values, or explicit
// "these N values are live outputs" annotations, would need something
// more general. For a graph with exactly one output, which is every
// graph this book has built through Chapter 8, "the last node is the
// output" is enough, and it has one very convenient side effect: it
// keeps deadCodeEliminationPass's own signature EXACTLY `Graph(const
// Graph&)`, identical to every other TransformPass this chapter and
// Chapter 8 define -- no extra "which node is the output" parameter
// ever needs to be threaded through Chapter 8's own PassManager.
//
// "Dead," precisely: a node is LIVE if it is the designated output, or
// if some other LIVE node depends on it (directly or transitively).
// Everything else is dead. This is computed by walking BACKWARD from
// the output through each node's own `inputs` -- the same field every
// other pass in this book already reads, just walked in the opposite
// direction from topologicalSort()'s own forward walk.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 016_dead_code_elimination_from_the_graphs_own_output.cpp -o 016_dead_code_elimination_from_the_graphs_own_output
// Run:     ./016_dead_code_elimination_from_the_graphs_own_output
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
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

static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

// ============================== Section 9.2: liveness and DCE (new) ==============================

using TransformPass = std::function<Graph(const Graph&)>;

// Walks BACKWARD from rootId through each node's own `inputs` -- the
// opposite direction from topologicalSort()'s forward walk -- marking
// every node reached along the way as live. A node with two live
// consumers (like this book's own diamond graph's own t1) is still
// visited and inserted only once, since `live` is a set: reachability
// doesn't care HOW MANY paths lead to a node, only whether at least
// one does.
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

// Rebuilds the graph keeping only LIVE nodes -- a dead node is simply
// never re-added, exactly the "rebuild rather than mutate" pattern
// Chapter 8 established, now finally used for the one thing it was
// always meant to make possible: this book's Graph still has no way to
// REMOVE a node from an existing graph, so dropping one has always
// meant "don't add it to the new graph in the first place."
static Graph deadCodeEliminationPass(const Graph& g) {
    if (g.size() == 0) throw std::runtime_error("deadCodeEliminationPass: an empty graph has no output node");
    int rootId = static_cast<int>(g.size()) - 1;  // convention: the last node IS the graph's own output
    std::set<int> live = computeLiveNodeIds(g, rootId);

    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("deadCodeEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
        if (!live.count(oldId)) continue;  // DEAD -- skipped, never added to the new graph

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

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
    }
}

int main() {
    printf("=== Section 9.2: deadCodeEliminationPass(), dropping a node nothing depends on ===\n\n");

    // The familiar diamond graph, PLUS one genuinely dead node ("junk")
    // inserted in the middle -- computed from real graph nodes (a, b),
    // so it LOOKS legitimate, but nothing that "out" (the last node,
    // this graph's own designated output) depends on ever reaches it.
    Graph withJunk;
    Value a  = withJunk.addInput("a");
    Value b  = withJunk.addInput("b");
    Value t1 = withJunk.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = withJunk.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = withJunk.addUnary(OpKind::ReLU, t1, "t3");
    withJunk.addBinary(OpKind::Add, a, b, "junk");             // DEAD: computed, never consumed
    withJunk.addBinary(OpKind::Add, t2, t3, "out");            // the graph's own output (last node)

    printf("Graph with one dead node ('junk'):\n");
    printGraph(withJunk);

    Graph cleaned = deadCodeEliminationPass(withJunk);
    printf("\nAfter deadCodeEliminationPass():\n");
    printGraph(cleaned);

    bool junkRemoved = (cleaned.size() == withJunk.size() - 1);
    bool everyRemainingNodeIsLive = true;
    for (const auto& n : cleaned.nodes()) {
        if (n->debugName == "junk") everyRemainingNodeIsLive = false;
    }
    printf("\nself-check: exactly one node was dropped, %zu -> %zu (%s)\n",
           withJunk.size(), cleaned.size(), junkRemoved ? "confirmed" : "MISMATCH");
    printf("self-check: 'junk' does not appear anywhere in the cleaned graph (%s)\n",
           everyRemainingNodeIsLive ? "confirmed" : "MISMATCH");

    // t1 has TWO live consumers (t2 and t3) -- confirm it was kept
    // exactly ONCE, not duplicated, and that both of its consumers
    // still correctly reference it in the new graph.
    const Node* t1InCleaned = findNodeByName(cleaned, "t1");
    const Node* t2InCleaned = findNodeByName(cleaned, "t2");
    const Node* t3InCleaned = findNodeByName(cleaned, "t3");
    int t1OccurrenceCount = 0;
    for (const auto& n : cleaned.nodes()) if (n->debugName == "t1") ++t1OccurrenceCount;
    bool t2ReferencesT1 = t2InCleaned->inputs[0].nodeId == t1InCleaned->id;
    bool t3ReferencesT1 = t3InCleaned->inputs[0].nodeId == t1InCleaned->id;
    printf("self-check: 't1' (2 live consumers) appears exactly once in the cleaned graph,\n");
    printf("and both 't2' and 't3' still correctly reference that ONE copy (%s)\n",
           (t1OccurrenceCount == 1 && t2ReferencesT1 && t3ReferencesT1) ? "confirmed" : "MISMATCH");

    // Control case: a graph with NO dead nodes at all should come back
    // completely unchanged in size -- DCE must not misfire on clean input.
    Graph clean;
    Value ca = clean.addInput("a");
    Value cb = clean.addInput("b");
    Value ct1 = clean.addBinary(OpKind::Add, ca, cb, "t1");
    Value ct2 = clean.addBinary(OpKind::Mul, ct1, ca, "t2");
    Value ct3 = clean.addUnary(OpKind::ReLU, ct1, "t3");
    clean.addBinary(OpKind::Add, ct2, ct3, "out");
    Graph stillClean = deadCodeEliminationPass(clean);
    bool nothingRemoved = (stillClean.size() == clean.size());
    printf("\nself-check: a graph with NO dead nodes comes back the same size, %zu -> %zu\n",
           clean.size(), stillClean.size());
    printf("(DCE does not misfire on already-clean input) (%s)\n",
           nothingRemoved ? "confirmed" : "MISMATCH");

    bool allOk = junkRemoved && everyRemainingNodeIsLive && t1OccurrenceCount == 1 &&
                 t2ReferencesT1 && t3ReferencesT1 && nothingRemoved;
    return allOk ? 0 : 1;
}
