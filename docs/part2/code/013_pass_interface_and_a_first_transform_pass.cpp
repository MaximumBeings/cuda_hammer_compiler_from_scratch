// Chapter 8: The Pass Manager: Structuring Compiler Transformations
// 013_pass_interface_and_a_first_transform_pass.cpp
//
// Section 8.1 -- what a PASS actually is, made concrete for THIS book's
// own Graph -- and Section 8.2 -- the very first real transform pass,
// one deliberately simple enough not to overlap with Chapter 9's
// constant folding, Chapter 10's common subexpression elimination, or
// Chapter 11's algebraic simplification, while still being a genuine,
// useful rewrite rather than a placeholder.
//
// Chapter 2 already gave this book the word "pass": something that
// reads an IR and produces a result, without touching source text
// again. Chapters 6 and 7 already wrote two: inferShapes() (an ANALYSIS
// pass -- reads a Graph, produces a side table, never touches the
// Graph itself) and printGraphAsSource() (arguably a pass too, though
// its result is text rather than a data structure). This file adds the
// other kind: a TRANSFORM pass, one that reads a Graph and produces a
// DIFFERENT Graph.
//
// One design decision worth stating up front, because it shapes
// everything from here through Chapter 11: this book's own Graph (from
// Chapter 4) has never had a way to REMOVE a node once added -- addInput/
// addConst/addUnary/addBinary are strictly append-only. Chapter 9's dead
// code elimination is going to need to drop nodes; a real in-place
// removal API is more machinery than this book needs to add just to
// support that. Instead, a TransformPass in this book has the signature
// `Graph(const Graph&)` -- it reads an old graph and BUILDS AND RETURNS
// a brand new one, keeping only what it wants to keep. This "rebuild
// rather than mutate" style is a real, legitimate design some real
// compiler IRs use too (this is a general architectural fact, not a
// specific claim needing its own citation) -- it trades a bit of extra
// copying for never needing a removal API at all.
//
// This file's own first real transform pass, canonicalizeNodeNames(),
// renumbers every node's debugName to "%<id>" in topological order.
// That might look cosmetic, but it solves a real problem: two graphs
// that compute the exact same thing, built through different paths
// (by hand, parsed from text, or already passed through some earlier
// pass), can have completely different human-chosen names -- which
// means Chapter 7's own printGraphAsSource() would print them
// differently even though they are structurally identical. A canonical
// naming pass makes "print two graphs and diff the text" a meaningful
// way to compare them, which Chapter 9 onward will lean on constantly
// when showing a pass's own before/after effect.
//
// Reuses Chapter 4's Value/Node/Graph and topologicalSort(), and
// Chapter 5's graphsStructurallyEqual() (which compares op kind and
// input ids -- NEVER debugName -- making it exactly the right tool to
// confirm this pass changes names and nothing else).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 013_pass_interface_and_a_first_transform_pass.cpp -o 013_pass_interface_and_a_first_transform_pass
// Run:     ./013_pass_interface_and_a_first_transform_pass
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
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

// ==================== Structural equality (from Chapter 5, unchanged) ====================
//
// Deliberately compares op kind and input ids only -- NEVER debugName.
// That is exactly what makes it the right tool to check a renaming
// pass: two graphs this function calls equal may still have completely
// different names on every single node.
static bool graphsStructurallyEqual(const Graph& a, const Graph& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const Node* na = a.node(static_cast<int>(i));
        const Node* nb = b.node(static_cast<int>(i));
        if (na->op != nb->op) return false;
        if (na->inputs.size() != nb->inputs.size()) return false;
        for (size_t j = 0; j < na->inputs.size(); ++j) {
            if (na->inputs[j].nodeId != nb->inputs[j].nodeId) return false;
        }
    }
    return true;
}

// ============================== Section 8.1: the Pass interface (new) ==============================
//
// A TransformPass reads an existing Graph and returns a BRAND NEW one.
// It never mutates the Graph it was given -- the caller's own original
// graph is always still intact after a pass runs, which is exactly what
// lets Chapter 8's own PassManager (File 014) print a "before" dump,
// run the pass, and print an "after" dump without the "before" dump
// having become a lie partway through.
using TransformPass = std::function<Graph(const Graph&)>;

// ============================== Section 8.2: canonicalizeNodeNames() (new) ==============================
//
// Walks the input graph in topological order (reusing Chapter 4's own
// topologicalSort() explicitly, rather than assuming id order already
// matches it -- the same discipline Chapter 6's inferShapes() and
// Chapter 7's own [COMMON TRAP]-adjacent note both already established:
// don't rely on a coincidence a future pass might not preserve) and
// rebuilds every node into a fresh Graph with a canonical name, "%<id>"
// matching that node's OWN id in the new graph. A map from the OLD
// graph's node ids to the Values the NEW graph produced is threaded
// through so every input reference gets correctly remapped, even if the
// new graph's ids ever ended up different from the old ones.
static Graph canonicalizeNodeNames(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("canonicalizeNodeNames: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string canonicalName = "%" + std::to_string(result.size());
        Value newValue;

        if (n->op == OpKind::Input) {
            newValue = result.addInput(canonicalName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, canonicalName);
        } else if (n->inputs.size() == 1) {
            Value remappedIn = oldIdToNewValue.at(n->inputs[0].nodeId);
            newValue = result.addUnary(n->op, remappedIn, canonicalName);
        } else {
            Value remappedLhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value remappedRhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            newValue = result.addBinary(n->op, remappedLhs, remappedRhs, canonicalName);
        }

        oldIdToNewValue[oldId] = newValue;
    }

    return result;
}

int main() {
    printf("=== Section 8.2: canonicalizeNodeNames(), a real first transform pass ===\n\n");

    // The familiar diamond graph, with human-chosen names -- exactly
    // what Chapters 4 through 7 have all used.
    Graph original;
    Value a  = original.addInput("a");
    Value b  = original.addInput("b");
    Value t1 = original.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = original.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = original.addUnary(OpKind::ReLU, t1, "t3");
    original.addBinary(OpKind::Add, t2, t3, "out");

    printf("Original graph (human-chosen names):\n");
    for (const auto& n : original.nodes()) {
        printf("  %%%d = %s(%s)\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
    }

    Graph canonical = canonicalizeNodeNames(original);

    printf("\nAfter canonicalizeNodeNames() (canonical names):\n");
    for (const auto& n : canonical.nodes()) {
        printf("  %%%d = %s(%s)\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
    }

    // Every node's debugName should now be exactly "%" + its own id.
    bool everyNameCanonical = true;
    for (const auto& n : canonical.nodes()) {
        if (n->debugName != "%" + std::to_string(n->id)) everyNameCanonical = false;
    }
    printf("\nself-check: every node's debugName is exactly \"%%<its own id>\" (%s)\n",
           everyNameCanonical ? "confirmed" : "MISMATCH");

    // The real claim: canonicalization changed NAMES ONLY. Structural
    // equality (which never looks at debugName at all) must still hold.
    bool stillStructurallyEqual = graphsStructurallyEqual(original, canonical);
    printf("self-check: the canonicalized graph is still graphsStructurallyEqual() to the\n");
    printf("original -- only names changed, the computation itself did not (%s)\n",
           stillStructurallyEqual ? "confirmed" : "MISMATCH");

    // The original graph must be completely untouched -- a TransformPass
    // reads, it never mutates.
    bool originalNamesUntouched = (original.node(t1.nodeId)->debugName == "t1") &&
                                   (original.node(t2.nodeId)->debugName == "t2") &&
                                   (original.node(t3.nodeId)->debugName == "t3");
    printf("self-check: the ORIGINAL graph's own names are completely untouched --\n");
    printf("canonicalizeNodeNames() read from it but never mutated it (%s)\n",
           originalNamesUntouched ? "confirmed" : "MISMATCH");

    bool allOk = everyNameCanonical && stillStructurallyEqual && originalNamesUntouched;
    return allOk ? 0 : 1;
}
