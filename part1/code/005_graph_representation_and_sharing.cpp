// Chapter 4: Representing Tensors and Operations as a Graph
// 005_graph_representation_and_sharing.cpp
//
// Section 4.1 -- why a tensor computation needs a graph, not a tree or a
// linear list -- and Section 4.2 -- Node and Value as the actual building
// blocks CUDA Hammer's IR is made of, from here through the rest of the
// book.
//
// Chapter 2's toy compiler always produced a tree (as an AST) and then a
// straight-line list (as linear IR), because both were built by walking
// PARSED TEXT: a subexpression written once in source produces exactly
// one AST node, so nothing in that pipeline could ever make one computed
// result feed two different downstream operations without writing that
// subexpression's text out twice (and recomputing it). A real tensor
// program does not have this restriction -- a value computed once is
// routinely consumed by more than one later operation (a normalization's
// output feeding both a residual-add and a following layer, for
// instance) -- so CUDA Hammer's own IR needs a data structure that lets
// one computed result be referenced by more than one consumer without
// duplicating the operation that produced it. That data structure is a
// graph: Value identifies a specific node's output, Node holds a list of
// Values as its own inputs (edges pointing backward at its producers),
// and Graph owns every Node.
//
// This file builds a small, genuinely shared DAG by hand (through
// CUDA Hammer's own graph-construction API, not by parsing text) and
// counts two real things: how many consumers each node's output
// actually has, and how many nodes an equivalent TREE (no sharing --
// every multi-consumer value inlined at every one of its use sites
// instead) would need to represent the identical computation. That
// second count is computed by a genuinely recursive, UNMEMOIZED
// expansion -- which, not coincidentally, redoes the shared node's own
// work once per consumer while computing it, the IR-size version of
// exactly the "eager execution recomputes what fusion would have
// shared" argument Chapter 1 already made about runtime memory traffic.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 005_graph_representation_and_sharing.cpp -o 005_graph_representation_and_sharing
// Run:     ./005_graph_representation_and_sharing
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <stdexcept>

// ============================ Value and Node ============================
//
// A Value never points at a Node directly (no pointer, no reference) --
// it is just a small, cheap-to-copy pair of integers: which node
// produced it, and which of that node's outputs. Every op in this
// chapter has exactly one output, so outputIndex is always 0 here, but
// keeping it explicit now means a future op with two real outputs (for
// instance, a max-with-argmax op producing both a value and an index)
// needs no change to Value's own shape later, only a new node kind that
// happens to report more than one output.
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
    std::vector<Value> inputs;  // edges INTO this node, i.e. this node's own producers
    float constValue = 0.0f;    // only meaningful when op == OpKind::Const
};

// Graph owns every Node through a vector<unique_ptr<Node>> -- Node
// objects are individually heap-allocated, so a Value's nodeId stays
// valid forever even if the owning vector itself reallocates (which is
// exactly why Value stores an id, resolved through Graph::node(), rather
// than a raw Node* that a vector reallocation could otherwise leave
// dangling if Node were stored by value instead of by unique_ptr).
class Graph {
public:
    Value addInput(const std::string& name) {
        return addNode(OpKind::Input, {}, name);
    }
    Value addConst(float v, const std::string& name) {
        Value out = addNode(OpKind::Const, {}, name);
        nodes_.back()->constValue = v;
        return out;
    }
    Value addUnary(OpKind op, Value in, const std::string& name) {
        return addNode(op, {in}, name);
    }
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) {
        return addNode(op, {lhs, rhs}, name);
    }

    const Node* node(int id) const { return nodes_.at(static_cast<size_t>(id)).get(); }
    size_t size() const { return nodes_.size(); }
    const std::vector<std::unique_ptr<Node>>& nodes() const { return nodes_; }

    // Genuinely counted: how many other nodes' input lists reference
    // each node's single output, across the whole graph.
    std::map<int, int> consumerCounts() const {
        std::map<int, int> counts;
        for (const auto& n : nodes_) counts[n->id] = 0;
        for (const auto& n : nodes_) {
            for (const Value& in : n->inputs) {
                counts[in.nodeId]++;
            }
        }
        return counts;
    }

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

// Recursively expands a value into the node count an equivalent TREE
// would need -- every multi-consumer node's own subtree gets counted
// again, once per visit, exactly mirroring what inlining it at every use
// site (instead of sharing it) would actually cost. Deliberately NOT
// memoized: a memoized version would just be re-deriving the graph's own
// sharing, defeating the point of counting what NOT sharing costs.
static long long expandToTreeNodeCount(const Graph& g, Value v) {
    const Node* n = g.node(v.nodeId);
    long long count = 1;
    for (const Value& in : n->inputs) {
        count += expandToTreeNodeCount(g, in);
    }
    return count;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) {
            printf(" [value=%.1f]", n->constValue);
        }
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
    printf("=== Section 4.1 / 4.2: a genuinely shared DAG, built through CUDA Hammer's own graph API ===\n\n");

    // Builds the diamond below directly through addInput/addBinary/addUnary
    // -- never by parsing text, so this is real graph CONSTRUCTION, not
    // Chapter 2's kind of parsing-driven tree building:
    //
    //   a, b = Input, Input
    //   t1   = Add(a, b)
    //   t2   = Mul(t1, a)      <- t1 used again here
    //   t3   = ReLU(t1)        <- and again here: t1 has TWO consumers
    //   out  = Add(t2, t3)
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");

    printf("Graph (%zu nodes):\n", g.size());
    printGraph(g);

    printf("\nConsumer counts (how many other nodes' inputs reference each node's output):\n");
    std::map<int, int> counts = g.consumerCounts();
    bool t1HasTwoConsumers = false;
    for (const auto& n : g.nodes()) {
        int c = counts[n->id];
        printf("  %%%d (%s %s): %d consumer%s\n", n->id, opKindStr(n->op).c_str(),
               n->debugName.c_str(), c, c == 1 ? "" : "s");
        if (n->id == t1.nodeId) t1HasTwoConsumers = (c == 2);
    }

    // Note that %0 (a) ALSO has 2 consumers -- and that alone would not
    // distinguish this from Chapter 2's tree, since a tree can "reuse" a
    // leaf value too (parsing "a + a" produces two separate Variable AST
    // nodes that both look `a` up, with no sharing required). What a tree
    // genuinely cannot do is share a COMPUTED, non-leaf result -- t1 is
    // an Add's own output, not a leaf, and its 2 consumers below (t2 and
    // t3) are exactly the case a tree would have to duplicate the entire
    // Add operation to represent.
    printf("\nself-check: t1 (a computed, non-leaf node) has exactly 2 consumers (%s) -- sharing\n",
           t1HasTwoConsumers ? "confirmed" : "MISMATCH");
    printf("a COMPUTED result this way, without recomputing it, is what a tree cannot do.\n");

    long long treeNodeCount = expandToTreeNodeCount(g, out);
    printf("\nReal DAG node count:                    %zu\n", g.size());
    printf("Equivalent unshared-tree node count:    %lld  (t1's own subtree -- {a, b, t1}, 3 nodes --\n",
           treeNodeCount);
    printf("                                          gets counted twice: once while expanding t2,\n");
    printf("                                          once more while expanding t3)\n");

    bool treeIsLarger = treeNodeCount > static_cast<long long>(g.size());
    printf("self-check: the unshared-tree expansion is strictly larger than the real DAG (%s)\n",
           treeIsLarger ? "confirmed" : "MISMATCH");

    return (t1HasTwoConsumers && treeIsLarger) ? 0 : 1;
}
