// Chapter 6: Graph Validation and Shape Inference
// 009_shape_inference_by_topological_walk.cpp
//
// Section 6.1 -- what "valid" means for a tensor graph (the broadcasting
// rule two operand shapes must satisfy) -- and Section 6.2 -- inferring
// every node's own output shape by walking the graph in the exact
// topological order Chapter 4's File 006 already built.
//
// A Node in this book's Graph has never stored a shape -- Chapter 4 and
// 5 both built and parsed graphs entirely in terms of ids and edges,
// with no notion of "how big" any value actually is. Shape inference is
// a separate PASS (Chapter 2's own vocabulary: something that reads an
// IR and computes a result, without ever touching source text again)
// that walks an already-built Graph and produces a SIDE TABLE --
// map<nodeId, Shape> -- rather than modifying Node itself. Keeping shape
// data external like this is deliberate: Chapter 8's pass manager will
// run many different analyses over the same Graph, and none of them
// need to change Node's own definition to do it.
//
// Reuses Chapter 4's own Value/Node/Graph AND its topologicalSort()
// (Kahn's algorithm) completely unchanged -- shape inference's only new
// idea is WHAT to compute at each node once the order to compute it in
// is already known.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 009_shape_inference_by_topological_walk.cpp -o 009_shape_inference_by_topological_walk
// Run:     ./009_shape_inference_by_topological_walk
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
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

// ==================== Topological sort (from Chapter 4's File 006, unchanged) ====================

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

// ============================== Shape (new) ==============================

struct Shape {
    std::vector<int> dims;
};

static std::string shapeStr(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s.dims[i]);
    }
    out += "]";
    return out;
}

static bool shapesEqual(const Shape& a, const Shape& b) { return a.dims == b.dims; }

// ======================= Section 6.1: the broadcasting rule =======================
//
// The same elementwise broadcasting rule NumPy, PyTorch, and TensorFlow
// all use: compare two shapes dimension by dimension, starting from the
// TRAILING (rightmost) dimension and working left; a dimension missing
// on the shorter shape (because it has fewer dimensions) is treated as
// size 1; two aligned dimensions are compatible when they are EQUAL, or
// when AT LEAST ONE of them is 1 -- in which case the result takes
// whichever size is not 1. If any aligned pair is neither equal nor has
// a 1, the two shapes cannot be broadcast together at all.
static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        // Index from the trailing end on each side independently.
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        int outDim;
        if (da == db) {
            outDim = da;
        } else if (da == 1) {
            outDim = db;
        } else if (db == 1) {
            outDim = da;
        } else {
            throw std::runtime_error(context + ": shapes " + shapeStr(a) + " and " + shapeStr(b) +
                                      " are not broadcast-compatible (" + std::to_string(da) + " vs " +
                                      std::to_string(db) + " at trailing position " + std::to_string(i) + ")");
        }
        result[rank - 1 - i] = outDim;
    }
    return Shape{result};
}

// ======================= Section 6.2: shape inference over a graph =======================
//
// declaredShapes supplies the shape for every Input/Const node -- there
// is nothing to INFER for a leaf, its shape is given, not computed.
// Every other node's shape is computed from its own already-known
// inputs' shapes, in the exact order topologicalSort() already proved
// is valid: by the time this loop reaches any node, every one of its
// inputs has ALREADY been visited (that is precisely what a topological
// order guarantees), so `shapes.at(...)` below can never fail to find
// an input's shape it needs.
static std::map<int, Shape> inferShapes(const Graph& g, const std::map<int, Shape>& declaredShapes) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("inferShapes: graph is not acyclic");

    std::map<int, Shape> shapes;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        std::string context = "node %" + std::to_string(id) + " (" + n->debugName + ")";

        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            auto it = declaredShapes.find(id);
            if (it == declaredShapes.end()) {
                throw std::runtime_error(context + ": no declared shape was provided for this leaf node");
            }
            shapes[id] = it->second;
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            const Shape& lhs = shapes.at(n->inputs[0].nodeId);
            const Shape& rhs = shapes.at(n->inputs[1].nodeId);
            shapes[id] = broadcastShapes(lhs, rhs, context);
        } else { // ReLU -- shape-preserving: output shape equals input shape exactly.
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

int main() {
    printf("=== Section 6.1: the broadcasting rule, tested on its own ===\n\n");

    struct BroadcastCase { Shape a, b, expected; };
    std::vector<BroadcastCase> cases = {
        {{{3, 4}},    {{3, 4}},    {{3, 4}}},   // identical shapes
        {{{3, 4}},    {{4}},       {{3, 4}}},   // trailing match, leading dim padded with 1
        {{{3, 1}},    {{1, 4}},    {{3, 4}}},   // mutual broadcasting, both directions
        {{{5}},       {{1}},       {{5}}},      // scalar-like broadcasting
        {{{2, 3, 4}}, {{4}},       {{2, 3, 4}}} // higher rank + trailing match
    };

    bool allBroadcastCasesMatch = true;
    for (const auto& c : cases) {
        Shape result = broadcastShapes(c.a, c.b, "test case");
        bool matches = shapesEqual(result, c.expected);
        allBroadcastCasesMatch = allBroadcastCasesMatch && matches;
        printf("  %s broadcast %s -> %s  (expected %s, %s)\n", shapeStr(c.a).c_str(), shapeStr(c.b).c_str(),
               shapeStr(result).c_str(), shapeStr(c.expected).c_str(), matches ? "confirmed" : "MISMATCH");
    }
    printf("\nself-check: every broadcasting case matches its hand-computed expected shape (%s)\n\n",
           allBroadcastCasesMatch ? "confirmed" : "MISMATCH");

    printf("=== Section 6.2: inferring every node's shape over Chapter 4's diamond graph ===\n\n");

    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");

    // Only the two leaves need a DECLARED shape -- "a" is a full [3, 4]
    // tensor, "b" is a [4]-shaped tensor that broadcasts against it,
    // deliberately exercising broadcasting on the very first Add.
    std::map<int, Shape> declared = {
        {a.nodeId, Shape{{3, 4}}},
        {b.nodeId, Shape{{4}}},
    };

    std::map<int, Shape> inferred = inferShapes(g, declared);

    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s): shape %s\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str(),
               shapeStr(inferred.at(n->id)).c_str());
    }

    bool outShapeCorrect = shapesEqual(inferred.at(out.nodeId), Shape{{3, 4}});
    printf("\nself-check: 'out' -- reached through two broadcasts (t1) and one shape-preserving\n");
    printf("ReLU (t3) -- correctly infers to [3, 4] (%s)\n", outShapeCorrect ? "confirmed" : "MISMATCH");

    return (allBroadcastCasesMatch && outShapeCorrect) ? 0 : 1;
}
