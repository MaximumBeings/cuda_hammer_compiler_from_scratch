// Chapter 10: Common Subexpression Elimination for Tensor Graphs
// 018_common_subexpression_elimination_reuses_what_already_exists.cpp
//
// Section 10.1 -- common subexpression elimination (CSE), this book's
// THIRD TransformPass, plugging into the exact same Chapter 8 machinery
// Section 9.1's constantFoldPass() and Section 9.2's
// deadCodeEliminationPass() already plug into, with zero changes to
// that machinery required.
//
// Constant folding (9.1) asks "is this computation's answer already
// fully known?" Dead code elimination (9.2) asks "does anything still
// need this computation at all?" Common subexpression elimination asks
// a third, different question: "has this EXACT computation already been
// done somewhere else in this graph?" A real tensor program can easily
// compute the same thing twice without anyone intending it -- the same
// normalization applied along two different code paths that happen to
// read the same two tensors, say -- and CSE's job is to notice and
// collapse that duplication down to one computation, reused everywhere
// it was needed.
//
// "Exact computation," here, means something very specific: the SAME
// operation, applied to the SAME operands, where "the same operands"
// means the same already-deduplicated NODES in the graph being rebuilt
// -- not merely operands that would happen to evaluate to equal numbers
// at runtime. A Const(5) and an Input node that happens to receive 5.0
// at runtime are not "the same" to this pass; it has no way to know
// what an Input will be fed at runtime, and correctly does not try to
// guess. This file characterizes exactly what CSE built this way DOES
// catch. Section 10.2 characterizes what it deliberately does NOT.
//
// commonSubexpressionEliminationPass() rebuilds the graph in
// topological order, exactly like every earlier TransformPass in this
// book -- except before adding a new node, it computes a CANONICAL KEY
// describing exactly what that node computes, and checks whether some
// EARLIER node in this same walk already produced an identical key. If
// one did, the duplicate is never built at all: every later consumer is
// pointed at the node that already exists. This is the same "just don't
// add it to the new graph" idiom Section 9.2 used to drop nodes nothing
// needed -- here, a node is left out not because nothing needs it, but
// because something IDENTICAL already exists for everything to share.
//
// Because a node's key is built from the NEW graph's own (already
// deduplicated) ids -- via oldIdToNewValue, populated as the walk
// proceeds, the exact "check the new graph, not the old one" discipline
// constantFoldPass() used to fold an entire chain in a single pass --
// a merge discovered early in this walk is immediately visible to
// whatever gets keyed later. That lets a single CSE pass discover a
// CASCADE of merges: two duplicate leaves merging first can make two
// nodes that each consume one of them duplicates of each other too,
// discovered in the very same walk, with no need to run the pass twice.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 018_common_subexpression_elimination_reuses_what_already_exists.cpp -o 018_common_subexpression_elimination_reuses_what_already_exists
// Run:     ./018_common_subexpression_elimination_reuses_what_already_exists
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
//
// Keyed by node NAME, not id, for the exact reason Chapter 9's own
// evaluate() comment explains: a rebuilt graph's ids are not guaranteed
// to match the original's once a graph has more than one
// simultaneously-ready leaf, and this chapter's own passes are no
// exception to that rule.
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

// ============================== Section 10.1: commonSubexpressionEliminationPass() (new) ==============================

using TransformPass = std::function<Graph(const Graph&)>;

// The canonical key describing exactly what a node computes, in terms
// of the NEW (already-deduplicated) graph being rebuilt: an Input's key
// is its own name; a Const's key is its literal value; an Add/Mul/
// ReLU's key is its op plus the NEW ids of its own (already-processed)
// operands. Two nodes with an identical key compute, by construction,
// the identical thing -- one of them is redundant.
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
    std::map<std::string, Value> keyToExistingValue;  // canonical key -> the node already built for it

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string key = cseKey(n, oldIdToNewValue);

        auto found = keyToExistingValue.find(key);
        if (found != keyToExistingValue.end()) {
            // Something already built in THIS walk computes exactly this
            // -- reuse it. oldId is remapped, but nothing new is added.
            oldIdToNewValue[oldId] = found->second;
            continue;
        }

        // First time this exact key has been seen -- actually build it.
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

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) printf(" [value=%g]", n->constValue);
        printf("\n");
    }
}

int main() {
    printf("=== Section 10.1: commonSubexpressionEliminationPass(), reusing an already-computed value ===\n\n");

    // t1 and t3 compute the SAME thing -- add(a, b), in the SAME operand
    // order -- built independently, as two different names. t4 consumes
    // t3; after CSE, it should consume t1's node instead, and t3 itself
    // should not exist in the result at all.
    Graph withDuplicateAdd;
    Value a  = withDuplicateAdd.addInput("a");
    Value b  = withDuplicateAdd.addInput("b");
    Value t1 = withDuplicateAdd.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = withDuplicateAdd.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = withDuplicateAdd.addBinary(OpKind::Add, a, b, "t3");  // duplicate of t1
    Value t4 = withDuplicateAdd.addBinary(OpKind::Mul, t3, b, "t4");
    withDuplicateAdd.addBinary(OpKind::Add, t2, t4, "out");

    printf("Graph with one duplicate computation ('t3' duplicates 't1'):\n");
    printGraph(withDuplicateAdd);

    Graph deduped = commonSubexpressionEliminationPass(withDuplicateAdd);
    printf("\nAfter commonSubexpressionEliminationPass():\n");
    printGraph(deduped);

    bool oneNodeDropped = (deduped.size() == withDuplicateAdd.size() - 1);
    bool t3Gone = true;
    for (const auto& n : deduped.nodes()) if (n->debugName == "t3") t3Gone = false;
    printf("\nself-check: exactly one duplicate node was dropped, %zu -> %zu (%s)\n",
           withDuplicateAdd.size(), deduped.size(), oneNodeDropped ? "confirmed" : "MISMATCH");
    printf("self-check: 't3' does not appear anywhere in the deduplicated graph (%s)\n",
           t3Gone ? "confirmed" : "MISMATCH");

    const Node* t1InDeduped = findNodeByName(deduped, "t1");
    const Node* t4InDeduped = findNodeByName(deduped, "t4");
    bool t4NowReferencesT1 = t4InDeduped->inputs[0].nodeId == t1InDeduped->id;
    printf("self-check: 't4' now reads its first operand from the SAME node as 't1' (%s)\n",
           t4NowReferencesT1 ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs1 = {{"a", 3.0f}, {"b", 5.0f}};
    float originalOut1 = evaluate(withDuplicateAdd, inputs1).at("out");
    float dedupedOut1 = evaluate(deduped, inputs1).at("out");
    bool sameAnswer1 = (originalOut1 == dedupedOut1);
    printf("self-check: evaluate(original, a=3,b=5).out = %g, evaluate(deduped, a=3,b=5).out = %g (%s)\n",
           originalOut1, dedupedOut1, sameAnswer1 ? "confirmed" : "MISMATCH");

    // A second, independent case: two LITERALLY duplicate Const nodes.
    // Merging them is a plain instance of the same key mechanism (no
    // folding involved -- both are already Const when the pass sees
    // them) -- and it CASCADES: once the two Const(7) nodes collapse
    // into one, the two Add nodes that each consumed one of them turn
    // out to have an identical key too, and collapse in the same walk.
    Graph withDuplicateConst;
    Value k1 = withDuplicateConst.addConst(7.0f, "k1");
    Value m  = withDuplicateConst.addInput("m");
    Value r1 = withDuplicateConst.addBinary(OpKind::Add, k1, m, "r1");
    Value k2 = withDuplicateConst.addConst(7.0f, "k2");  // literal duplicate of k1's value
    Value r2 = withDuplicateConst.addBinary(OpKind::Add, k2, m, "r2");
    withDuplicateConst.addBinary(OpKind::Add, r1, r2, "out");

    printf("\nGraph with two literally-duplicate Const nodes ('k1' and 'k2', both 7):\n");
    printGraph(withDuplicateConst);

    Graph constDeduped = commonSubexpressionEliminationPass(withDuplicateConst);
    printf("\nAfter commonSubexpressionEliminationPass():\n");
    printGraph(constDeduped);

    // 6 nodes -> 4: k2 merges into k1 (same key), which makes r2's own
    // key collapse onto r1's, all discovered in this ONE pass.
    bool constCascadeCorrect = (constDeduped.size() == 4);
    printf("\nself-check: the Const merge CASCADED into an Add merge in the same pass, %zu -> %zu (%s)\n",
           withDuplicateConst.size(), constDeduped.size(), constCascadeCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs2 = {{"m", 4.5f}};
    float originalOut2 = evaluate(withDuplicateConst, inputs2).at("out");
    float dedupedOut2 = evaluate(constDeduped, inputs2).at("out");
    bool sameAnswer2 = (originalOut2 == dedupedOut2);
    printf("self-check: evaluate(original, m=4.5).out = %g, evaluate(deduped, m=4.5).out = %g (%s)\n",
           originalOut2, dedupedOut2, sameAnswer2 ? "confirmed" : "MISMATCH");

    // Control case: a graph with NO duplicate computations at all should
    // come back completely unchanged in size -- CSE must not misfire.
    Graph clean;
    Value ca = clean.addInput("a");
    Value cb = clean.addInput("b");
    Value ct1 = clean.addBinary(OpKind::Add, ca, cb, "t1");
    Value ct2 = clean.addBinary(OpKind::Mul, ct1, ca, "t2");
    Value ct3 = clean.addUnary(OpKind::ReLU, ct1, "t3");
    clean.addBinary(OpKind::Add, ct2, ct3, "out");
    Graph stillClean = commonSubexpressionEliminationPass(clean);
    bool nothingMerged = (stillClean.size() == clean.size());
    printf("\nself-check: a graph with NO duplicate computations comes back the same size, %zu -> %zu\n",
           clean.size(), stillClean.size());
    printf("(CSE does not misfire on already-distinct input) (%s)\n",
           nothingMerged ? "confirmed" : "MISMATCH");

    bool allOk = oneNodeDropped && t3Gone && t4NowReferencesT1 && sameAnswer1 &&
                 constCascadeCorrect && sameAnswer2 && nothingMerged;
    return allOk ? 0 : 1;
}
