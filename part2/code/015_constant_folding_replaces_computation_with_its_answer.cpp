// Chapter 9: Constant Folding and Dead Code Elimination
// 015_constant_folding_replaces_computation_with_its_answer.cpp
//
// Section 9.1 -- constant folding, this book's first genuinely
// OPTIMIZING transform pass, plugging directly into Chapter 8's own
// TransformPass signature and design.
//
// Chapter 2's File 004 already built a constant-folding pass -- but for
// a completely different, much simpler IR (a flat sequence of
// three-address-code instructions over an arithmetic expression, with
// no notion of Input/Const/Add/Mul/ReLU nodes or a graph structure at
// all). This file builds the real thing for CUDA Hammer's own Graph,
// reusing Chapter 2's own CORE IDEA -- if every operand a computation
// needs is already known at compile time, replace the computation with
// its answer -- but built entirely from this chapter's own Value/Node/
// Graph/topologicalSort()/TransformPass machinery.
//
// constantFoldPass() walks the input graph in topological order (via
// Chapter 4's own topologicalSort(), the same discipline Chapters 6, 7,
// and 8 all already established) and rebuilds every node into a new
// Graph, exactly the way Chapter 8's canonicalizeNodeNames() did --
// except at Add, Mul, and ReLU, it first checks whether its own
// (already-rebuilt) inputs turned out to be Const nodes. If they did,
// it computes the real arithmetic answer right now and emits a single
// Const node instead of the original operation. Because this check
// looks at the NEW graph's own nodes (not the old graph's), a whole
// CHAIN of constant computations folds down to one single Const node
// in a single pass -- no need to run the pass repeatedly until nothing
// changes, for the graphs this chapter's own tests use.
//
// Correctness here means something stronger than "the fold happened" --
// it means the folded graph computes the EXACT SAME ANSWER as the
// original, for every actual input. This file adds a small evaluate()
// interpreter (new here, playing the same verification role Chapter
// 2's own IR interpreter played for ITS constant folding pass) and
// checks the original and folded graphs agree, for concrete input
// values, at the node both graphs still share by id (constantFoldPass
// never drops or reorders nodes -- only Section 9.2's dead code
// elimination will do that).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 015_constant_folding_replaces_computation_with_its_answer.cpp -o 015_constant_folding_replaces_computation_with_its_answer
// Run:     ./015_constant_folding_replaces_computation_with_its_answer
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

// ============================== evaluate() (new): a small interpreter ==============================
//
// Plays the same role Chapter 2's own IR interpreter played for ITS
// constant folding pass: an independent way to compute what a graph
// ACTUALLY produces for concrete input values, so folding can be
// checked against real execution rather than merely trusted by
// construction. Walks topological order, exactly like inferShapes()
// (Chapter 6) and every transform pass in this chapter.
//
// Deliberately keyed by NAME (a node's own debugName), never by node
// id, for both the caller-supplied input values AND the returned
// result -- a real lesson this file's own tests surfaced directly
// while being written: constantFoldPass() rebuilds a graph in
// TOPOLOGICAL order, and topological order is not guaranteed to match
// the ORIGINAL graph's id order once a graph has more than one
// simultaneously-ready leaf (this file's own test graph does -- c1,
// c2, c3, AND x are all leaves, so Kahn's algorithm visits all four
// before it can visit anything that depends on them, which does not
// necessarily preserve their original relative id order once other,
// non-leaf nodes were created in between). A node's id can silently
// shift across a rebuild; comparing two graphs' evaluate() results by
// id would be comparing the wrong nodes half the time. Comparing by
// NAME sidesteps the problem entirely, since every pass in this
// chapter (unlike Chapter 8's own canonicalizeNodeNames()) preserves
// debugName exactly.
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

// ============================== Section 9.1: constantFoldPass() (new) ==============================
//
// Rebuilds the graph node by node, in topological order -- structurally
// the same walk Chapter 8's canonicalizeNodeNames() used -- except at
// Add, Mul, and ReLU, it checks whether its (already-rebuilt) operands
// are Const nodes IN THE NEW GRAPH. Because a node folded earlier in
// this same walk becomes a Const in the new graph immediately, a later
// node that consumes it sees a Const operand too -- letting an entire
// chain of constant computation fold down to one Const in a single
// topological pass, with no need to iterate until nothing changes.
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

// Finds a node by its own debugName -- the SAME "look up by name, not
// id" discipline evaluate() above already needs, for the identical
// reason: after constantFoldPass() rebuilds a graph in topological
// order, a node's NEW id is not guaranteed to match its OLD id (this
// file's own three-constant-plus-one-input test graph is a genuine
// case where it does not -- see evaluate()'s own comment). Any code
// that wants to find "the node that used to be called t1" after a
// rebuild has to search by name, never reuse the old Value's nodeId.
static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) printf(" [value=%g]", n->constValue);
        printf("\n");
    }
}

int main() {
    printf("=== Section 9.1: constantFoldPass(), folding a whole constant chain in one pass ===\n\n");

    // c1 + c2 -> foldable; (c1+c2) * c3 -> ALSO foldable, using the
    // already-folded result; the final add brings in a real runtime
    // input "x", so folding necessarily stops there.
    Graph original;
    Value c1 = original.addConst(2.0f, "c1");
    Value c2 = original.addConst(3.0f, "c2");
    Value c3 = original.addConst(4.0f, "c3");
    Value t1 = original.addBinary(OpKind::Add, c1, c2, "t1");   // 2 + 3 = 5
    Value t2 = original.addBinary(OpKind::Mul, t1, c3, "t2");   // 5 * 4 = 20 (chained fold)
    Value x  = original.addInput("x");
    original.addBinary(OpKind::Add, t2, x, "out");  // 20 + x -- NOT foldable (looked up by name below)

    printf("Original graph:\n");
    printGraph(original);

    Graph folded = constantFoldPass(original);
    printf("\nAfter constantFoldPass():\n");
    printGraph(folded);

    // Looked up by NAME, not by the original t1/t2/out Values' own
    // nodeId -- constantFoldPass() rebuilds in topological order, and
    // (as evaluate()'s own comment explains) this graph's ids genuinely
    // shift across that rebuild, since c1/c2/c3 AND x are all
    // simultaneously-ready leaves. Using the stale old ids here would
    // silently check the WRONG nodes in the new graph.
    const Node* t1InFolded = findNodeByName(folded, "t1");
    const Node* t2InFolded = findNodeByName(folded, "t2");
    const Node* outInFolded = findNodeByName(folded, "out");
    bool t1Folded = t1InFolded->op == OpKind::Const && t1InFolded->constValue == 5.0f;
    bool t2Folded = t2InFolded->op == OpKind::Const && t2InFolded->constValue == 20.0f;
    bool outNotFolded = outInFolded->op == OpKind::Add;
    printf("\nself-check: t1 (2+3) folded to Const(5) (%s)\n", t1Folded ? "confirmed" : "MISMATCH");
    printf("self-check: t2 ((2+3)*4), using t1's ALREADY-folded value, folded to Const(20) (%s)\n",
           t2Folded ? "confirmed" : "MISMATCH");
    printf("self-check: 'out' (20 + x) stays a real Add -- x is not a compile-time constant (%s)\n",
           outNotFolded ? "confirmed" : "MISMATCH");

    // The claim that actually matters: folding did not change what the
    // graph COMPUTES. Evaluate both graphs at a real input value --
    // keyed and looked up by NAME, not id, for the reason explained
    // above evaluate()'s own definition.
    std::map<std::string, float> inputs = {{"x", 6.5f}};
    float originalOut = evaluate(original, inputs).at("out");
    float foldedOut = evaluate(folded, inputs).at("out");
    bool sameAnswer = (originalOut == foldedOut);
    printf("self-check: evaluate(original, x=6.5).out = %g, evaluate(folded, x=6.5).out = %g (%s)\n",
           originalOut, foldedOut, sameAnswer ? "confirmed" : "MISMATCH");

    // A second, independent case: ReLU of a constant.
    Graph reluGraph;
    Value negC = reluGraph.addConst(-2.5f, "negC");
    reluGraph.addUnary(OpKind::ReLU, negC, "reluOut");  // looked up by name below
    Graph reluFolded = constantFoldPass(reluGraph);
    const Node* reluOutInFolded = findNodeByName(reluFolded, "reluOut");
    bool reluFoldedCorrectly = reluOutInFolded->op == OpKind::Const && reluOutInFolded->constValue == 0.0f;
    printf("self-check: relu(const(-2.5)) folds to Const(0) (%s)\n",
           reluFoldedCorrectly ? "confirmed" : "MISMATCH");

    bool allOk = t1Folded && t2Folded && outNotFolded && sameAnswer && reluFoldedCorrectly;
    return allOk ? 0 : 1;
}
