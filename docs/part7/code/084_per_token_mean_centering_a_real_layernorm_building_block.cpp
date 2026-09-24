// Chapter 32: NLP/Transformers -- A Real Bag-of-Embeddings Classifier
// 084_per_token_mean_centering_a_real_layernorm_building_block.cpp
//
// Section 32.1 -- Part 7's own promise (no new OpKind, no new pass, no
// new backend) meets a real gap this time, not just a missing primitive
// Section 31.2's im2col-style staging could route around. Real LayerNorm
// (Ba et al. 2016) is `(x - mean) / sqrt(variance + eps) * gamma + beta`,
// computed PER TOKEN, independently, across that token's own feature
// dimension. CUDA Hammer's `Sum` op has reduced a WHOLE input tensor
// down to ONE scalar since Chapter 14 -- it has never had a per-row or
// "reduce along one axis of a batch" mode. A real [4 tokens x 8
// features] tensor needs 4 INDEPENDENT reductions, not one.
//
// This section is honest about that gap rather than inventing a batched-
// Sum op to paper over it (which would break Part 7's own opening
// promise). The real workaround: run Chapter 30/31's own two-pass
// "measure with Sum, then bake the result into a fresh graph as a Const"
// calibration pattern ONCE PER TOKEN, in a plain host-side C++ loop --
// four small, independent CUDA Hammer graphs, not one big one. This is
// also the honest reason `sqrt`/division are left out entirely: this
// section computes the MEAN-CENTERING half of LayerNorm only (`x -
// mean`), the half `Add`/`Mul`/`Sum` can actually express; the variance/
// sqrt/divide half needs primitives this IR does not have, named here
// directly rather than faked with a hardcoded constant standing in for
// a real standard deviation.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 084_per_token_mean_centering_a_real_layernorm_building_block.cpp -o 084_driver
// Run:     ./084_driver
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / Node / Graph (from File 081, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };
enum class OpKind { Input, Const, Add, Mul, ReLU, Sum };

class Graph {
public:
    Value addInput(const std::string& name) { return addNode(OpKind::Input, {}, name); }
    Value addConst(float v, const std::string& name) {
        Value out = addNode(OpKind::Const, {}, name);
        nodes_.back()->constValue = v;
        return out;
    }
    Value addUnary(OpKind op, Value in, const std::string& name) { return addNode(op, {in}, name); }
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) { return addNode(op, {lhs, rhs}, name); }

    struct Node {
        int id;
        OpKind op;
        std::string debugName;
        std::vector<Value> inputs;
        float constValue = 0.0f;
    };
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
using Node = Graph::Node;

// ==================== Topological sort / Shape / inferShapes / evaluateArrays (from File 081, unchanged) ====================

struct TopoResult { std::vector<int> order; bool ok = true; };
static TopoResult topologicalSort(const Graph& g) {
    std::map<int, int> inDegree;
    for (const auto& n : g.nodes()) inDegree[n->id] = static_cast<int>(n->inputs.size());
    std::deque<int> ready;
    for (const auto& n : g.nodes()) if (inDegree[n->id] == 0) ready.push_back(n->id);
    TopoResult result;
    while (!ready.empty()) {
        int id = ready.front();
        ready.pop_front();
        result.order.push_back(id);
        for (const auto& n : g.nodes())
            for (const Value& in : n->inputs)
                if (in.nodeId == id) { if (--inDegree[n->id] == 0) ready.push_back(n->id); }
    }
    result.ok = (result.order.size() == g.size());
    return result;
}

struct Shape { std::vector<int> dims; };
static long long numElements(const Shape& s) { long long n = 1; for (int d : s.dims) n *= d; return n; }
static Shape broadcastShapes(const Shape& a, const Shape& b) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        result[rank - 1 - i] = (da == db) ? da : (da == 1 ? db : da);
    }
    return Shape{result};
}
static std::map<int, Shape> inferShapes(const Graph& g, const std::map<int, Shape>& declaredShapes) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("inferShapes: graph is not acyclic");
    std::map<int, Shape> shapes;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) shapes[id] = declaredShapes.at(id);
        else if (n->op == OpKind::Add || n->op == OpKind::Mul)
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId));
        else if (n->op == OpKind::ReLU) shapes[id] = shapes.at(n->inputs[0].nodeId);
        else shapes[id] = Shape{};
    }
    return shapes;
}
static std::map<std::string, std::vector<float>> evaluateArrays(
        const Graph& g, const std::map<std::string, std::vector<float>>& inputArraysByName,
        const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluateArrays: graph is not acyclic");
    std::map<int, std::vector<float>> buffersById;
    std::map<std::string, std::vector<float>> buffersByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        std::vector<float> buf;
        if (n->op == OpKind::Input) {
            buf = inputArraysByName.at(n->debugName);
        } else if (n->op == OpKind::Const) {
            buf = {n->constValue};
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            long long count = elementCounts.at(id);
            buf.resize(static_cast<size_t>(count));
            const std::vector<float>& lhs = buffersById.at(n->inputs[0].nodeId);
            const std::vector<float>& rhs = buffersById.at(n->inputs[1].nodeId);
            long long lc = static_cast<long long>(lhs.size()), rc = static_cast<long long>(rhs.size());
            for (long long i = 0; i < count; ++i) {
                float lv = lhs[i % lc], rv = rhs[i % rc];
                buf[i] = (n->op == OpKind::Add) ? (lv + rv) : (lv * rv);
            }
        } else if (n->op == OpKind::ReLU) {
            const std::vector<float>& in = buffersById.at(n->inputs[0].nodeId);
            buf.resize(in.size());
            for (size_t i = 0; i < in.size(); ++i) buf[i] = std::max(0.0f, in[i]);
        } else {
            const std::vector<float>& in = buffersById.at(n->inputs[0].nodeId);
            float acc = 0.0f;
            for (float v : in) acc += v;
            buf = {acc};
        }
        buffersById[id] = buf;
        buffersByName[n->debugName] = buf;
    }
    return buffersByName;
}

// ==================== Section 32.1: the sentence's own real (illustrative) token embeddings ====================
//
// 4 tokens, embedding dimension 8 -- small enough to hand-check, deliberately
// including two edge cases: token1 is a CONSTANT vector (every real LayerNorm
// implementation must send a constant vector's own variance to exactly 0,
// which this section's mean-centering half must send to exactly the zero
// vector); token2 is already exactly zero-mean (centering must be a no-op).
static std::vector<std::vector<float>> buildTokenEmbeddings() {
    return {
        {1, 2, 3, 4, 5, 6, 7, 8},                  // token0: mean 4.5
        {10, 10, 10, 10, 10, 10, 10, 10},          // token1: constant vector, mean 10.0
        {-4, -3, -2, -1, 1, 2, 3, 4},               // token2: already zero-mean
        {0, 0, 0, 0, 100, 100, 100, 100},           // token3: mean 50.0
    };
}

// One token's own real two-pass calibration graph, extending Chapter
// 30/31's Sum-then-Const pattern: Pass 1 measures this token's own mean
// with a real Sum; Pass 2 builds a FRESH graph applying x + (-mean) --
// gain fixed at 1.0 (this section centers only, it does not rescale);
// Pass 3 re-measures the CENTERED vector's own mean with Sum a second
// time, checking the real invariant mean(centered) == 0.
struct CenteringResult {
    float measuredMean;
    std::vector<float> centered;
    float centeredMean;
};
static CenteringResult centerOneToken(const std::vector<float>& tokenVals, int dim) {
    Graph statsGraph;
    Value sx = statsGraph.addInput("x");
    Value ssum = statsGraph.addUnary(OpKind::Sum, sx, "sum_x");
    Value srecipN = statsGraph.addConst(1.0f / static_cast<float>(dim), "recipN");
    Value smean = statsGraph.addBinary(OpKind::Mul, ssum, srecipN, "mean");
    (void)smean;
    std::map<int, Shape> statsDeclared = {{sx.nodeId, Shape{{dim}}}, {srecipN.nodeId, Shape{}}};
    std::map<int, Shape> statsShapes = inferShapes(statsGraph, statsDeclared);
    std::map<int, long long> statsCounts;
    for (const auto& kv : statsShapes) statsCounts[kv.first] = numElements(kv.second);
    auto statsResult = evaluateArrays(statsGraph, {{"x", tokenVals}}, statsCounts);
    float mean = statsResult.at("mean")[0];

    Graph centerGraph;
    Value cx = centerGraph.addInput("x");
    Value cnegMean = centerGraph.addConst(-mean, "neg_mean");
    Value ccentered = centerGraph.addBinary(OpKind::Add, cx, cnegMean, "centered");
    (void)ccentered;
    std::map<int, Shape> centerDeclared = {{cx.nodeId, Shape{{dim}}}, {cnegMean.nodeId, Shape{}}};
    std::map<int, Shape> centerShapes = inferShapes(centerGraph, centerDeclared);
    std::map<int, long long> centerCounts;
    for (const auto& kv : centerShapes) centerCounts[kv.first] = numElements(kv.second);
    auto centerResult = evaluateArrays(centerGraph, {{"x", tokenVals}}, centerCounts);
    std::vector<float> centered = centerResult.at("centered");

    Graph verifyGraph;
    Value vx = verifyGraph.addInput("centered");
    Value vsum = verifyGraph.addUnary(OpKind::Sum, vx, "sum_centered");
    Value vrecipN = verifyGraph.addConst(1.0f / static_cast<float>(dim), "recipN");
    Value vmean = verifyGraph.addBinary(OpKind::Mul, vsum, vrecipN, "mean_centered");
    (void)vmean;
    std::map<int, Shape> verifyDeclared = {{vx.nodeId, Shape{{dim}}}, {vrecipN.nodeId, Shape{}}};
    std::map<int, Shape> verifyShapes = inferShapes(verifyGraph, verifyDeclared);
    std::map<int, long long> verifyCounts;
    for (const auto& kv : verifyShapes) verifyCounts[kv.first] = numElements(kv.second);
    auto verifyResult = evaluateArrays(verifyGraph, {{"centered", centered}}, verifyCounts);
    float centeredMean = verifyResult.at("mean_centered")[0];

    return CenteringResult{mean, centered, centeredMean};
}

int main() {
    printf("=== Chapter 32 (Part 7): Section 32.1 -- per-token mean-centering, a real LayerNorm building block ===\n\n");
    bool allOk = true;
    const int dim = 8;

    std::vector<std::vector<float>> tokens = buildTokenEmbeddings();
    printf("4 tokens, embedding dimension %d -- token1 is a CONSTANT vector (an edge case: every\n", dim);
    printf("real variance computation must send it to exactly 0); token2 is already exactly zero-mean\n");
    printf("(centering must be a real no-op).\n\n");

    printf("--- running Chapter 30/31's own two-pass calibration graph ONCE PER TOKEN (host-orchestrated,\n");
    printf("    since CUDA Hammer's Sum has never supported a per-row/batched reduction) ---\n\n");

    std::vector<std::vector<float>> allCentered;
    for (size_t t = 0; t < tokens.size(); ++t) {
        CenteringResult r = centerOneToken(tokens[t], dim);
        allCentered.push_back(r.centered);
        printf("token%zu: measured mean (real Sum, Pass 1) = %8.4f | centered[0..%d] =", t, static_cast<double>(r.measuredMean), dim - 1);
        for (float v : r.centered) printf(" %7.3f", v);
        printf("\n         re-measured mean of centered vector (real Sum, Pass 3) = %.6f\n", r.centeredMean);
        bool invariantOk = std::fabs(r.centeredMean) < 1e-4f;
        printf("         self-check: mean(centered) == 0 (%s)\n\n", invariantOk ? "confirmed" : "MISMATCH");
        allOk = allOk && invariantOk;
    }

    bool token1AllZero = true;
    for (float v : allCentered[1]) if (std::fabs(v) > 1e-4f) token1AllZero = false;
    printf("self-check: token1 (a constant vector) centers to EXACTLY the zero vector, the real edge\n");
    printf("case a variance computation must also send to 0 (%s)\n", token1AllZero ? "confirmed" : "MISMATCH");
    allOk = allOk && token1AllZero;

    bool token2Unchanged = true;
    for (int i = 0; i < dim; ++i) if (std::fabs(allCentered[2][static_cast<size_t>(i)] - tokens[2][static_cast<size_t>(i)]) > 1e-4f) token2Unchanged = false;
    printf("self-check: token2 (already zero-mean) is UNCHANGED by centering, a real no-op (%s)\n\n",
           token2Unchanged ? "confirmed" : "MISMATCH");
    allOk = allOk && token2Unchanged;

    printf("--- the real gap this section works around, named directly ---\n\n");
    printf("this section computes x - mean only. Real LayerNorm also divides by sqrt(variance + eps)\n");
    printf("and applies a learned gamma/beta -- this IR has no sqrt and no division (Chapters 4-30 never\n");
    printf("added either), so that half is left out here rather than faked with a hardcoded constant\n");
    printf("standing in for a real standard deviation. Section 32.2 continues without it.\n\n");

    printf("=== Section 32.1 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
