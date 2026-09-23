// Chapter 12: Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels
// 026_measuring_fusions_effect_on_cuda_hammers_own_graph.cpp
//
// Section 12.3 -- everything Sections 12.1 and 12.2 derived by formula,
// on a hand-picked straight-line CHAIN, measured here instead on a real
// CUDA Hammer Graph -- specifically the exact diamond graph Chapter 4's
// File 005 built by hand and Chapter 6's File 009 already inferred
// shapes over. A diamond is a strictly harder case than a chain: node
// 'a' and node 't1' each have TWO consumers, so an unfused kernel for
// each of their consumers reads that shared value from memory AGAIN,
// while a fused kernel reads it from memory exactly ONCE no matter how
// many downstream ops consume it -- fusion's benefit on a real DAG comes
// from eliminating both kinds of repeated traffic (repeated-tensor-in-a-
// chain AND shared-value-with-multiple-consumers), not just the first.
//
// Reuses Chapter 4's Value/Node/Graph and topologicalSort() (Kahn's
// algorithm) and Chapter 6's Shape/broadcastShapes()/inferShapes()
// completely unchanged -- this file's only new idea is walking that
// already-inferred shape table to count bytes moved, the same way
// Chapter 9's deadCodeEliminationPass() walked node->inputs to count
// liveness instead of shape.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 026_measuring_fusions_effect_on_cuda_hammers_own_graph.cpp -o 026_measuring_fusions_effect_on_cuda_hammers_own_graph
// Run:     ./026_measuring_fusions_effect_on_cuda_hammers_own_graph
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / Node / Graph (from Chapter 4, unchanged) ====================

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

// ==================== Shape / broadcastShapes / inferShapes (from Chapter 6, unchanged) ====================

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

static long long numElements(const Shape& s) {
    long long n = 1;
    for (int d : s.dims) n *= d;
    return n;
}

static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
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
                                      " are not broadcast-compatible");
        }
        result[rank - 1 - i] = outDim;
    }
    return Shape{result};
}

static std::map<int, Shape> inferShapes(const Graph& g, const std::map<int, Shape>& declaredShapes) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("inferShapes: graph is not acyclic");
    std::map<int, Shape> shapes;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        std::string context = "node %" + std::to_string(id) + " (" + n->debugName + ")";
        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            shapes[id] = declaredShapes.at(id);
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        } else {  // ReLU -- shape-preserving
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

// ==================== Section 12.2's ridge point (from File 025, unchanged) ====================

static constexpr double kBytesPerElement = 4.0;
static constexpr double kA100Fp32PeakFlopsPerSec = 19.5e12;
static constexpr double kA100BandwidthBytesPerSec = 1555.0e9;
static double ridgePointFlopsPerByte() { return kA100Fp32PeakFlopsPerSec / kA100BandwidthBytesPerSec; }
static const char* classify(double ai, double ridge) { return (ai >= ridge) ? "compute-bound" : "memory-bound"; }

// ==================== Section 12.3: counting bytes moved, unfused vs. fused ====================
//
// UNFUSED: each non-leaf node is its own kernel -- it reads every one of
// its own operands from memory (even if some other node already read
// that SAME operand a moment ago) and writes its own output back to
// memory. A shared value like 'a' or 't1' therefore gets re-read from
// memory once per consumer.
static long long unfusedBytesMoved(const Graph& g, const std::map<int, Shape>& shapes) {
    long long totalElements = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;  // a leaf has nothing to compute
        for (const Value& in : n->inputs) {
            totalElements += numElements(shapes.at(in.nodeId));  // read each operand from memory
        }
        totalElements += numElements(shapes.at(n->id));  // write this node's own output
    }
    return static_cast<long long>(static_cast<double>(totalElements) * kBytesPerElement);
}

// FUSED: a single kernel reads each DISTINCT leaf (Input/Const) from
// memory exactly once -- no matter how many downstream nodes consume it
// -- and writes only the graph's own designated output (Chapter 9's
// "last node is the output" convention) back to memory exactly once.
// Every intermediate value (t1, t2, t3 here) lives in a register for the
// one fused kernel body and never touches memory at all.
static long long fusedBytesMoved(const Graph& g, const std::map<int, Shape>& shapes) {
    long long leafElements = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            leafElements += numElements(shapes.at(n->id));
        }
    }
    const Node* outputNode = g.nodes().back().get();  // Chapter 9's own convention
    long long outputElements = numElements(shapes.at(outputNode->id));
    return static_cast<long long>(static_cast<double>(leafElements + outputElements) * kBytesPerElement);
}

// Total FLOPs is IDENTICAL either way -- fusion changes where and how
// often data crosses memory, never how much arithmetic the graph does.
static long long totalFlops(const Graph& g, const std::map<int, Shape>& shapes) {
    long long total = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        total += numElements(shapes.at(n->id));  // one FLOP per output element, same simplification as File 024
    }
    return total;
}

int main() {
    printf("=== Section 12.3: Chapter 4's diamond graph, with Chapter 6's own declared shapes ===\n\n");

    // The exact diamond from Chapter 4's File 005 / Chapter 6's File 009:
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
    (void)out;

    // Same declared leaf shapes as Chapter 6's own File 009 -- 'a' is
    // [3, 4], 'b' is [4] and broadcasts against it at the very first Add.
    std::map<int, Shape> declared = {
        {a.nodeId, Shape{{3, 4}}},
        {b.nodeId, Shape{{4}}},
    };
    std::map<int, Shape> shapes = inferShapes(g, declared);

    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s): shape %s (%lld elements)\n", n->id, opKindStr(n->op).c_str(),
               n->debugName.c_str(), shapeStr(shapes.at(n->id)).c_str(), numElements(shapes.at(n->id)));
    }

    long long flops = totalFlops(g, shapes);
    long long unfusedBytes = unfusedBytesMoved(g, shapes);
    long long fusedBytes = fusedBytesMoved(g, shapes);

    printf("\nTotal FLOPs (identical either way, fusion doesn't change the arithmetic): %lld\n", flops);
    printf("Bytes moved, UNFUSED (each of t1/t2/t3/out is its own kernel):  %lld\n", unfusedBytes);
    printf("Bytes moved, FUSED   (one kernel, each leaf read once, 'out' written once): %lld\n", fusedBytes);

    bool fusedMovesFewerBytes = fusedBytes < unfusedBytes;
    printf("\nself-check: the fused kernel moves strictly fewer bytes than the unfused version (%s)\n",
           fusedMovesFewerBytes ? "confirmed" : "MISMATCH");

    double aiUnfused = static_cast<double>(flops) / static_cast<double>(unfusedBytes);
    double aiFused = static_cast<double>(flops) / static_cast<double>(fusedBytes);
    double ridge = ridgePointFlopsPerByte();

    printf("\nAI_unfused = %lld / %lld bytes = %.6f FLOPs/byte  (%s)\n", flops, unfusedBytes, aiUnfused,
           classify(aiUnfused, ridge));
    printf("AI_fused   = %lld / %lld bytes = %.6f FLOPs/byte  (%s)\n", flops, fusedBytes, aiFused,
           classify(aiFused, ridge));
    printf("(ridge point, same A100 numbers as Section 12.2: %.4f FLOPs/byte)\n", ridge);

    bool fusedAIHigher = aiFused > aiUnfused;
    printf("\nself-check: AI_fused is strictly higher than AI_unfused on this real DAG, not just on\n");
    printf("File 024's hand-picked chain (%s) -- fusion's benefit here comes from BOTH eliminating\n",
           fusedAIHigher ? "confirmed" : "MISMATCH");
    printf("chain-style intermediate traffic (t1->t2, t1->t3) AND reading a SHARED value ('a', 't1')\n");
    printf("from memory once instead of once per consumer.\n");

    printf("\nBoth land in the same '%s' classification here -- this diamond is tiny (12 elements\n",
           classify(aiUnfused, ridge));
    printf("per node), far below the K=101-per-chain scale Section 12.2 found necessary to cross this\n");
    printf("GPU's ridge point at all. The ratio improvement is real and measured either way; reaching\n");
    printf("compute-bound territory in practice needs the far larger tensors Part 3's actual fusion\n");
    printf("passes (starting next chapter) will be applied to, not a change to this result.\n");

    bool allOk = fusedMovesFewerBytes && fusedAIHigher;
    return allOk ? 0 : 1;
}
