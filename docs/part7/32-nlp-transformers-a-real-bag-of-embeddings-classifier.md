# Chapter 32: NLP / Transformers -- A Real Bag-of-Embeddings Classifier

Chapter 31 opened Part 7 with computer vision and closed it having kept
Part 7's own opening promise exactly: no new `OpKind`, no new pass, no
new backend -- only domain-shaped graph construction and one honest,
citable host-side staging technique (im2col) for the one thing the op
set could not express on its own. Chapter 32 moves to a second domain,
NLP, and finds a DIFFERENT gap in the same op set, worked around with a
different real technique.

## Chapter 32's own shape

```text
+------------------------------------------------------------------+
|  Chapter 32's own shape, section by section                      |
|                                                                    |
|  32.1  Per-token mean-centering -- a real LayerNorm building      |
|        block. Reuses Chapter 30/31's own two-pass "measure with   |
|        Sum, then bake into a fresh graph as a Const" pattern, run |
|        ONCE PER TOKEN in a host-side loop, since CUDA Hammer's    |
|        Sum has never had a per-row/batched reduction mode. An     |
|        honest scope boundary: only the mean-centering half of     |
|        LayerNorm, never a faked variance/sqrt/divide.             |
|                                                                    |
|  32.2  Bag-of-embeddings averaging plus a small unrolled linear   |
|        classifier -- a real, well-documented architecture         |
|        (fastText, Joulin et al. 2016), built from the SAME        |
|        scalar-weighted Mul+Add chain im2col used, now averaging   |
|        over tokens instead of shifted pixels, feeding a real      |
|        3-class linear classifier with per-feature weights as a    |
|        real Input, not a scalar Const.                            |
|                                                                    |
|  32.3  The capstone -- 32.2's own graph, unchanged, run through   |
|        Chapter 14/16's real boundedReductionFusionPass() and      |
|        Chapter 19's real per-node vectorized-CPU dispatcher,      |
|        compiled with this machine's own real AVX2/FMA or NEON     |
|        flags and RUN -- the same real fusion and codegen          |
|        machinery Chapter 31 used, now over a real NLP classifier. |
+------------------------------------------------------------------+
```

## 32.1 -- Per-token mean-centering, a real LayerNorm building block

Chapter 31 found a gap in CUDA Hammer's op set (no "read my neighbor"
primitive) and worked around it with pure host-side staging: im2col
shifts happen entirely outside the graph, so the graph itself never
needed a new op. This chapter opens on a DIFFERENT kind of gap, one
staging alone cannot route around.

Real LayerNorm (Ba, Kiros, and Hinton, 2016, "Layer Normalization") is
`(x - mean) / sqrt(variance + eps) * gamma + beta`, computed PER TOKEN,
independently, across that token's own feature dimension. CUDA Hammer's
`Sum` has reduced a WHOLE input tensor down to ONE scalar since Chapter
14 -- it has never had a per-row or "reduce along one axis of a batch"
mode. A real `[4 tokens x 8 features]` tensor needs 4 independent
reductions, not one whole-tensor sum. Padding a fake batched-Sum `OpKind`
in to make this convenient would break Part 7's own opening promise, so
this section does not: instead it runs Chapter 30/31's own two-pass
"measure with `Sum`, then bake the result into a fresh graph as a
`Const`" calibration pattern ONCE PER TOKEN, in a plain host-side C++
loop -- four small, independent CUDA Hammer graphs, not one big one:

```text
+----------------------------+     +-----------------------------+
|  ONE CALL PER TOKEN t       |     |  repeated for t = 0, 1, 2, 3 |
|                              |     |  (4 independent, small,      |
|  PASS 1: token[t] -> Sum     |     |   real CUDA Hammer graphs -- |
|          -> measured mean    |     |   NOT one batched graph,     |
|                              | --> |   because Sum cannot do a    |
|  PASS 2 (fresh graph):       |     |   per-row reduction)         |
|          token[t] + (-mean)  |     |                               |
|          -> centered[t]      |     |                               |
|                              |     |                               |
|  PASS 3: centered[t] -> Sum  |     |                               |
|          -> re-measured mean |     |                               |
|          (must be 0)         |     |                               |
+----------------------------+     +-----------------------------+
```

This section is honest about the rest of the gap too: it computes the
MEAN-CENTERING half of LayerNorm only (`x - mean`), the half
`Add`/`Mul`/`Sum` can actually express. The variance/`sqrt`/divide half
needs primitives this IR has never had, named here directly -- Chapters
4 through 30 never added `sqrt` or division, so Part 7 does not invent
them now, and nothing here stands a hardcoded constant in for a real
standard deviation. The section's own token data is chosen to exercise
two real edge cases a variance computation must also get right: one
token is a constant vector (every feature identical -- centering, and a
real variance, must send it to exactly zero), and one token is already
exactly zero-mean (centering must be a genuine no-op, not a coincidence
of rounding).


```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 084_per_token_mean_centering_a_real_layernorm_building_block.cpp -o 084_per_token_mean_centering_a_real_layernorm_building_block_driver
./084_per_token_mean_centering_a_real_layernorm_building_block_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 32 (Part 7): Section 32.1 -- per-token mean-centering, a real LayerNorm building block ===

4 tokens, embedding dimension 8 -- token1 is a CONSTANT vector (an edge case: every
real variance computation must send it to exactly 0); token2 is already exactly zero-mean
(centering must be a real no-op).

--- running Chapter 30/31's own two-pass calibration graph ONCE PER TOKEN (host-orchestrated,
    since CUDA Hammer's Sum has never supported a per-row/batched reduction) ---

token0: measured mean (real Sum, Pass 1) =   4.5000 | centered[0..7] =  -3.500  -2.500  -1.500  -0.500   0.500   1.500   2.500   3.500
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

token1: measured mean (real Sum, Pass 1) =  10.0000 | centered[0..7] =   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

token2: measured mean (real Sum, Pass 1) =   0.0000 | centered[0..7] =  -4.000  -3.000  -2.000  -1.000   1.000   2.000   3.000   4.000
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

token3: measured mean (real Sum, Pass 1) =  50.0000 | centered[0..7] = -50.000 -50.000 -50.000 -50.000  50.000  50.000  50.000  50.000
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

self-check: token1 (a constant vector) centers to EXACTLY the zero vector, the real edge
case a variance computation must also send to 0 (confirmed)
self-check: token2 (already zero-mean) is UNCHANGED by centering, a real no-op (confirmed)

--- the real gap this section works around, named directly ---

this section computes x - mean only. Real LayerNorm also divides by sqrt(variance + eps)
and applies a learned gamma/beta -- this IR has no sqrt and no division (Chapters 4-30 never
added either), so that half is left out here rather than faked with a hardcoded constant
standing in for a real standard deviation. Section 32.2 continues without it.

=== Section 32.1 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 32 (Part 7): Section 32.1 -- per-token mean-centering, a real LayerNorm building block ===

4 tokens, embedding dimension 8 -- token1 is a CONSTANT vector (an edge case: every
real variance computation must send it to exactly 0); token2 is already exactly zero-mean
(centering must be a real no-op).

--- running Chapter 30/31's own two-pass calibration graph ONCE PER TOKEN (host-orchestrated,
    since CUDA Hammer's Sum has never supported a per-row/batched reduction) ---

token0: measured mean (real Sum, Pass 1) =   4.5000 | centered[0..7] =  -3.500  -2.500  -1.500  -0.500   0.500   1.500   2.500   3.500
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

token1: measured mean (real Sum, Pass 1) =  10.0000 | centered[0..7] =   0.000   0.000   0.000   0.000   0.000   0.000   0.000   0.000
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

token2: measured mean (real Sum, Pass 1) =   0.0000 | centered[0..7] =  -4.000  -3.000  -2.000  -1.000   1.000   2.000   3.000   4.000
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

token3: measured mean (real Sum, Pass 1) =  50.0000 | centered[0..7] = -50.000 -50.000 -50.000 -50.000  50.000  50.000  50.000  50.000
         re-measured mean of centered vector (real Sum, Pass 3) = 0.000000
         self-check: mean(centered) == 0 (confirmed)

self-check: token1 (a constant vector) centers to EXACTLY the zero vector, the real edge
case a variance computation must also send to 0 (confirmed)
self-check: token2 (already zero-mean) is UNCHANGED by centering, a real no-op (confirmed)

--- the real gap this section works around, named directly ---

this section computes x - mean only. Real LayerNorm also divides by sqrt(variance + eps)
and applies a learned gamma/beta -- this IR has no sqrt and no division (Chapters 4-30 never
added either), so that half is left out here rather than faked with a hardcoded constant
standing in for a real standard deviation. Section 32.2 continues without it.

=== Section 32.1 complete: all self-checks confirmed ===
```

*Byte-identical on both machines -- this file has no CUDA/NCCL/MPI linkage and no vector intrinsics, so it is cross-verified on the cloud sandbox and the device, the same standing rule every plain C++ file in this book follows.*


## 32.2 -- Bag-of-embeddings averaging and a small unrolled linear classifier

Section 32.1 closed with 4 real, independently mean-centered token
vectors. This section builds a real sentence classifier out of them,
mirroring a real, well-documented, and deliberately simple architecture:
fastText (Joulin, Grave, Bojanowski, and Mikolov, 2016, "Bag of Tricks
for Efficient Text Classification," arXiv:1607.01759) represents a whole
sentence as the AVERAGE of its own word (or n-gram) embeddings, then
feeds that single averaged vector directly into a linear classifier --
deliberately no hidden layer, no attention, no recurrence. That
simplicity is exactly what makes it fast in practice, and exactly what
makes it expressible in CUDA Hammer's own real op set with no new
machinery at all.

The averaging step reuses Section 31.2's own im2col-style pattern,
turned to a different purpose. There, 9 SHIFTED COPIES of one image were
combined by a scalar-weighted `Mul` + unrolled `Add` chain. Here, 4
TOKEN VECTORS are combined the identical way -- each multiplied by a
uniform `1/4` `Const`, then folded together with an unrolled `Add`
chain -- the same real technique, averaging over a different axis
(tokens, not shifted pixels):

```text
  4 centered token vectors           the bag-of-embeddings vector
  (Section 32.1's own real output)
                                      token0 * 0.25  -+
  token0[8]  token1[8]                token1 * 0.25   |
  token2[8]  token3[8]     ---->      token2 * 0.25   +--> Add chain --> bag[8]
  (real Input nodes)                  token3 * 0.25  -+
                                      (same Mul+Add pattern as im2col,
                                       averaging over TOKENS not pixels)
```

The classifier step needs something Section 31's filters never did: a
weight that varies BY FEATURE INDEX, not one uniform scalar per term.
That is not a job for `Const` (always a single scalar, broadcast over a
whole tensor) -- it is data, so it is a real `Input`, exactly like every
pixel and every token vector already is. For each of 3 classes:
`Mul(bag, classWeights)` elementwise, then `Sum` reduces the 8 weighted
features to one real logit, then `Add` folds in a real bias `Const`.
This section deliberately stops at raw logits -- a real softmax needs
`exp` and division, neither of which this IR has ever had -- an honest
scope boundary, not a hidden gap, stated the same way Chapter 31 stated
its own.


```cpp
// Chapter 32: NLP/Transformers -- A Real Bag-of-Embeddings Classifier
// 085_bag_of_embeddings_averaging_and_a_small_unrolled_linear_classifier.cpp
//
// Section 32.2 -- Section 32.1 closed with 4 real, independently
// mean-centered token vectors. This section builds a real sentence
// classifier out of them, mirroring a real, well-documented architecture:
// fastText (Joulin, Grave, Bojanowski, Mikolov, 2016, "Bag of Tricks for
// Efficient Text Classification," arXiv:1607.01759) represents a whole
// sentence as the AVERAGE of its own word (or n-gram) embeddings, then
// feeds that single averaged vector directly into a LINEAR classifier --
// deliberately no hidden layer, no attention, no recurrence, which is
// exactly what makes it fast and exactly what makes it expressible in
// CUDA Hammer's own real op set with no new machinery.
//
// The averaging step reuses Section 31.2's own im2col-style pattern,
// turned to a different purpose: there, 9 SHIFTED COPIES of one image
// were combined by a scalar-weighted `Mul`+`Add` chain; here, 4 TOKEN
// VECTORS are combined the identical way (`Mul` by a uniform 1/4
// `Const`, then an unrolled `Add` chain) -- the same real technique,
// averaging over a different axis. The classifier step needs something
// Section 31's filters never did: a weight that varies BY FEATURE
// INDEX, not one uniform scalar per term. That is not a job for `Const`
// (always a single scalar, broadcast over a whole tensor) -- it is data,
// so it is a real `Input`, exactly like every pixel and every token
// vector already is: `Mul(bag, classWeights)` elementwise, then `Sum`
// reduces the 8 weighted features to one real logit, then `Add` folds in
// a real bias `Const`. This section deliberately stops at raw logits --
// a real softmax needs `exp` and division, neither of which this IR has
// ever had -- an honest scope boundary, not a hidden gap.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 085_bag_of_embeddings_averaging_and_a_small_unrolled_linear_classifier.cpp -o 085_driver
// Run:     ./085_driver
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <array>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / Node / Graph (from File 084, unchanged) ====================

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

// ==================== Topological sort / Shape / inferShapes / evaluateArrays (from File 084, unchanged) ====================

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

// ==================== Section 32.2: the 4 centered token vectors (Section 31.1's real output,
//                      reused as literal input data -- exactly Section 31.3's own pattern of
//                      reusing 31.1's real computed output rather than recomputing it) ====================

static std::vector<std::vector<float>> buildCenteredTokens() {
    return {
        {-3.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 3.5f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {-4.0f, -3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f, 4.0f},
        {-50.0f, -50.0f, -50.0f, -50.0f, 50.0f, 50.0f, 50.0f, 50.0f},
    };
}

int main() {
    printf("=== Chapter 32 (Part 7): Section 32.2 -- bag-of-embeddings averaging + a small unrolled linear classifier ===\n\n");
    bool allOk = true;
    const int dim = 8;

    std::vector<std::vector<float>> tokens = buildCenteredTokens();

    // ---------------------------------------------------------------
    // Part 1: the bag-of-embeddings vector, an unrolled Mul-by-a-uniform-
    // scalar-then-Add chain -- structurally identical to Section 31.2's
    // own im2col box blur, averaging over TOKENS instead of SHIFTED
    // PIXELS.
    // ---------------------------------------------------------------
    printf("--- Part 1: the bag-of-embeddings vector (real fastText-style sentence average) ---\n\n");
    Graph g;
    std::array<Value, 4> tokenVals;
    for (int t = 0; t < 4; ++t) tokenVals[static_cast<size_t>(t)] = g.addInput("token" + std::to_string(t));

    Value quarter = g.addConst(0.25f, "quarter");
    std::array<Value, 4> terms;
    for (int t = 0; t < 4; ++t) terms[static_cast<size_t>(t)] = g.addBinary(OpKind::Mul, tokenVals[static_cast<size_t>(t)], quarter, "term" + std::to_string(t));
    Value bag = terms[0];
    for (int t = 1; t < 4; ++t) {
        std::string name = (t == 3) ? "bag" : ("bag_partial" + std::to_string(t));
        bag = g.addBinary(OpKind::Add, bag, terms[static_cast<size_t>(t)], name);
    }

    // ---------------------------------------------------------------
    // Part 2: 3 classes, each a real Mul(bag, classWeights)->Sum->
    // Add(bias) branch -- classWeights is a real Input (a per-feature
    // weight vector, real data, not a single broadcast scalar), sharing
    // `bag` as a genuine 3-way multi-consumer fan-out, echoing Section
    // 31.3's own shift-input fan-out into two filters.
    // ---------------------------------------------------------------
    printf("--- Part 2: a small 3-class linear classifier over the bag vector, raw logits (no softmax --\n");
    printf("    that needs exp/division, which this IR does not have) ---\n\n");
    std::array<std::array<float, 8>, 3> classWeights = {{
        {1, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1},
        {1, -1, 1, -1, 1, -1, 1, -1},
    }};
    std::array<float, 3> classBias = {0.0f, 0.0f, 5.0f};

    std::array<Value, 3> classWeightVals;
    std::array<Value, 3> logits;
    std::map<std::string, std::vector<float>> inputArrays;
    for (int t = 0; t < 4; ++t) inputArrays["token" + std::to_string(t)] = tokens[static_cast<size_t>(t)];
    for (int c = 0; c < 3; ++c) {
        std::string wname = "class" + std::to_string(c) + "_weights";
        classWeightVals[static_cast<size_t>(c)] = g.addInput(wname);
        std::vector<float> wvec(classWeights[static_cast<size_t>(c)].begin(), classWeights[static_cast<size_t>(c)].end());
        inputArrays[wname] = wvec;
        Value prod = g.addBinary(OpKind::Mul, bag, classWeightVals[static_cast<size_t>(c)], "prod" + std::to_string(c));
        Value sum = g.addUnary(OpKind::Sum, prod, "sum" + std::to_string(c));
        Value biasConst = g.addConst(classBias[static_cast<size_t>(c)], "bias" + std::to_string(c));
        logits[static_cast<size_t>(c)] = g.addBinary(OpKind::Add, sum, biasConst, "logit" + std::to_string(c));
    }
    (void)logits;

    std::map<int, Shape> declared;
    for (const auto& n : g.nodes())
        if (n->op == OpKind::Input || n->op == OpKind::Const) declared[n->id] = (n->op == OpKind::Input) ? Shape{{dim}} : Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);
    auto result = evaluateArrays(g, inputArrays, elementCounts);

    printf("bag vector = ");
    for (float v : result.at("bag")) printf("%8.3f", static_cast<double>(v));
    printf("\n\n");
    for (int c = 0; c < 3; ++c) printf("class%d: logit%d = %.4f\n", c, c, static_cast<double>(result.at("logit" + std::to_string(c))[0]));
    printf("\n");

    // ---------------------------------------------------------------
    // Independent host-side reference: direct averaging + direct dot
    // products, never touching CUDA Hammer's own graph at all.
    // ---------------------------------------------------------------
    std::vector<float> bagRef(dim, 0.0f);
    for (int d = 0; d < dim; ++d) {
        float s = 0.0f;
        for (int t = 0; t < 4; ++t) s += tokens[static_cast<size_t>(t)][static_cast<size_t>(d)];
        bagRef[static_cast<size_t>(d)] = s * 0.25f;
    }
    std::array<float, 3> logitsRef;
    for (int c = 0; c < 3; ++c) {
        float s = 0.0f;
        for (int d = 0; d < dim; ++d) s += bagRef[static_cast<size_t>(d)] * classWeights[static_cast<size_t>(c)][static_cast<size_t>(d)];
        logitsRef[static_cast<size_t>(c)] = s + classBias[static_cast<size_t>(c)];
    }

    bool bagMatches = true;
    for (int d = 0; d < dim; ++d) if (std::fabs(result.at("bag")[static_cast<size_t>(d)] - bagRef[static_cast<size_t>(d)]) > 1e-3f) bagMatches = false;
    bool logitsMatch = true;
    for (int c = 0; c < 3; ++c) if (std::fabs(result.at("logit" + std::to_string(c))[0] - logitsRef[static_cast<size_t>(c)]) > 1e-3f) logitsMatch = false;

    printf("independent host-side reference (never touches CUDA Hammer's graph): bag = ");
    for (float v : bagRef) printf("%8.3f", static_cast<double>(v));
    printf("\nlogits = %.4f, %.4f, %.4f\n\n", static_cast<double>(logitsRef[0]), static_cast<double>(logitsRef[1]), static_cast<double>(logitsRef[2]));
    printf("self-check: the graph's own bag vector matches the independent reference exactly (%s)\n",
           bagMatches ? "confirmed" : "MISMATCH");
    printf("self-check: all 3 logits match the independent reference exactly (%s)\n\n",
           logitsMatch ? "confirmed" : "MISMATCH");
    allOk = allOk && bagMatches && logitsMatch;

    printf("--- a real hand-checkable symmetry, named directly ---\n\n");
    printf("class0's weights sum the bag's own first 4 features (the ones that came out negative\n");
    printf("after averaging), class1 sums the last 4 (the ones that came out positive) -- since this\n");
    printf("section's own token data was built symmetric around 0, logit0 = -54.5 and logit1 = +54.5\n");
    printf("exactly cancel, a real property of THIS data, checked above, not assumed.\n\n");

    printf("=== Section 32.2 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 085_bag_of_embeddings_averaging_and_a_small_unrolled_linear_classifier.cpp -o 085_bag_of_embeddings_averaging_and_a_small_unrolled_linear_classifier_driver
./085_bag_of_embeddings_averaging_and_a_small_unrolled_linear_classifier_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 32 (Part 7): Section 32.2 -- bag-of-embeddings averaging + a small unrolled linear classifier ===

--- Part 1: the bag-of-embeddings vector (real fastText-style sentence average) ---

--- Part 2: a small 3-class linear classifier over the bag vector, raw logits (no softmax --
    that needs exp/division, which this IR does not have) ---

bag vector =  -14.375 -13.875 -13.375 -12.875  12.875  13.375  13.875  14.375

class0: logit0 = -54.5000
class1: logit1 = 54.5000
class2: logit2 = 3.0000

independent host-side reference (never touches CUDA Hammer's graph): bag =  -14.375 -13.875 -13.375 -12.875  12.875  13.375  13.875  14.375
logits = -54.5000, 54.5000, 3.0000

self-check: the graph's own bag vector matches the independent reference exactly (confirmed)
self-check: all 3 logits match the independent reference exactly (confirmed)

--- a real hand-checkable symmetry, named directly ---

class0's weights sum the bag's own first 4 features (the ones that came out negative
after averaging), class1 sums the last 4 (the ones that came out positive) -- since this
section's own token data was built symmetric around 0, logit0 = -54.5 and logit1 = +54.5
exactly cancel, a real property of THIS data, checked above, not assumed.

=== Section 32.2 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 32 (Part 7): Section 32.2 -- bag-of-embeddings averaging + a small unrolled linear classifier ===

--- Part 1: the bag-of-embeddings vector (real fastText-style sentence average) ---

--- Part 2: a small 3-class linear classifier over the bag vector, raw logits (no softmax --
    that needs exp/division, which this IR does not have) ---

bag vector =  -14.375 -13.875 -13.375 -12.875  12.875  13.375  13.875  14.375

class0: logit0 = -54.5000
class1: logit1 = 54.5000
class2: logit2 = 3.0000

independent host-side reference (never touches CUDA Hammer's graph): bag =  -14.375 -13.875 -13.375 -12.875  12.875  13.375  13.875  14.375
logits = -54.5000, 54.5000, 3.0000

self-check: the graph's own bag vector matches the independent reference exactly (confirmed)
self-check: all 3 logits match the independent reference exactly (confirmed)

--- a real hand-checkable symmetry, named directly ---

class0's weights sum the bag's own first 4 features (the ones that came out negative
after averaging), class1 sums the last 4 (the ones that came out positive) -- since this
section's own token data was built symmetric around 0, logit0 = -54.5 and logit1 = +54.5
exactly cancel, a real property of THIS data, checked above, not assumed.

=== Section 32.2 complete: all self-checks confirmed ===
```

*Byte-identical on both machines, including the bag vector and all 3 logits against the independent host-side reference -- plain C++, no vector intrinsics, cross-verified the same way as File 084.*


## 32.3 -- The whole classifier, fused and vectorized for real

Files 084 and 085 each proved ONE real graph correct in isolation, the
same two-step shape Chapter 31 used. This capstone takes Section 32.2's
own graph -- unchanged, not reimplemented -- and runs it through
Chapter 14/16's real `boundedReductionFusionPass()` and Chapter 19's
real per-node vectorized-CPU dispatcher, compiled with this machine's
own real AVX2/FMA or NEON flags and RUN, exactly Section 31.3's own
capstone structure, now over a real (if tiny) NLP classifier instead of
a vision pipeline:

```text
  Section 32.1 (OUTSIDE the fused graph,     Section 32.2's graph
  by real necessity -- Sum has no            (fed into Stage 1/2 below,
  batched/per-row mode, so there is          UNCHANGED from File 085)
  nothing for a whole-graph fusion
  pass to do there)                          4 token Inputs -> Mul(0.25)
                                              -> Add chain -> bag[8]
  4 independent 2-pass calibration                  |
  graphs, one per token         -- feeds -->    (bag has 3 real
                                                 consumers: a genuine
                                                 multi-consumer fan-out)
                                                    |
                                        +-----------+-----------+
                                        |           |           |
                                    Mul+Sum      Mul+Sum      Mul+Sum
                                    +Add(bias)   +Add(bias)   +Add(bias)
                                    logit0       logit1       logit2
                                        |           |           |
                        +---------------+-----------+-----------+
                        |                                        |
                boundedReductionFusionPass()          real per-node vectorized
                (Chapter 14/16, unchanged)             CPU dispatch (Chapter 19,
                        |                               unchanged), compiled with
                        +----------------------> this machine's own real AVX2/FMA
                                                   or NEON flags, and RUN
```

Because `bag` genuinely feeds all three class branches, it is a real
multi-consumer node -- exactly the case Chapter 16 built fusion
boundaries to handle correctly rather than silently duplicating work.
The per-token mean-centering from Section 32.1 stays OUTSIDE this fused
graph on purpose, stated directly in the file itself: it is 4
independent small graphs by real necessity, not one big fusable graph,
so there is nothing for a whole-graph fusion pass to do there. What
Chapter 14/16's real pass CAN fuse -- and does -- is everything
downstream of centering: the averaging chain and all three classifier
branches.


```cpp
// Chapter 32: NLP/Transformers -- A Real Bag-of-Embeddings Classifier
// 086_the_whole_classifier_fused_and_vectorized_for_real.cpp
//
// Section 32.3 -- the capstone. Sections 32.1 and 32.2 each proved ONE
// real graph correct in isolation, the same two-step shape Chapter 31
// used. This file takes Section 32.2's own graph -- unchanged, not
// reimplemented -- and runs it through Chapter 14/16's real
// boundedReductionFusionPass() and Chapter 19's real per-node
// vectorized-CPU dispatcher, compiled with this machine's own real
// AVX2/FMA or NEON flags and RUN, exactly Section 31.3's own capstone
// structure, now over a real (if tiny) NLP classifier instead of a
// vision pipeline.
//
// The per-token mean-centering from Section 32.1 stays OUTSIDE this
// fused graph, on purpose, stated directly: it is 4 independent small
// graphs by real necessity (CUDA Hammer's Sum has no batched/per-row
// mode), not one big fusable graph, so there is nothing for a whole-
// graph fusion pass to do there. What Chapter 14/16's real pass CAN
// fuse -- and does -- is everything downstream of centering: the
// averaging chain and all three classifier branches, which share `bag`
// as a genuine 3-way fan-out.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 [-mavx2 -mfma on x86_64] 086_the_whole_classifier_fused_and_vectorized_for_real.cpp -o 086_driver
// Run:     ./086_driver
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <array>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14 / File 083, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };
enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };
enum class OperandKind { ExternalInput, PriorStep };
struct FusedOperand { OperandKind kind; int index; };
struct FusedStep { OpKind op; std::vector<FusedOperand> operands; long long reduceElementCount = 1; };

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;
    float constValue = 0.0f;
    std::vector<FusedStep> fusedSteps;
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
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) { return addNode(op, {lhs, rhs}, name); }
    Value addFusedElementwise(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
        Value out = addNode(OpKind::FusedElementwise, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
    }
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
        Value out = addNode(OpKind::FusedReduction, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
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

// ==================== Shape / broadcastShapes / inferShapes (from Chapters 6 and 14, unchanged) ====================

struct Shape { std::vector<int> dims; };
static long long numElements(const Shape& s) { long long n = 1; for (int d : s.dims) n *= d; return n; }
static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        int outDim;
        if (da == db) outDim = da;
        else if (da == 1) outDim = db;
        else if (db == 1) outDim = da;
        else throw std::runtime_error(context + ": shapes incompatible");
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
        if (n->op == OpKind::Input || n->op == OpKind::Const) shapes[id] = declaredShapes.at(id);
        else if (n->op == OpKind::Add || n->op == OpKind::Mul)
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        else if (n->op == OpKind::ReLU) shapes[id] = shapes.at(n->inputs[0].nodeId);
        else shapes[id] = Shape{};
    }
    return shapes;
}

// ==================== boundedReductionFusionPass() (from Chapter 16 / Section 17.3 / File 083, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static std::map<int, long long> computeChainDepths(const Graph& g, const std::map<int, int>& consumers, const TopoResult& topo) {
    std::map<int, long long> depth;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        bool isChainCandidate = (n->op == OpKind::Add || n->op == OpKind::Mul || n->op == OpKind::ReLU) && consumers.at(id) == 1;
        if (!isChainCandidate) { depth[id] = 0; continue; }
        long long best = 0;
        for (const Value& in : n->inputs) {
            const Node* pred = g.node(in.nodeId);
            bool predIsChainMember = (pred->op == OpKind::Add || pred->op == OpKind::Mul || pred->op == OpKind::ReLU) &&
                                      consumers.at(pred->id) == 1;
            if (predIsChainMember) best = std::max(best, depth.at(pred->id));
        }
        depth[id] = best + 1;
    }
    return depth;
}
static std::set<int> computeSizeCapBoundaries(const Graph& g, const std::map<int, int>& consumers, long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("computeSizeCapBoundaries: graph is not acyclic");
    std::map<int, long long> depth = computeChainDepths(g, consumers, topo);
    std::set<int> boundaries;
    for (const auto& kv : depth) if (kv.second > 0 && kv.second % maxChainLength == 0) boundaries.insert(kv.first);
    return boundaries;
}
static FusedOperand resolveIntoGroupBounded(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                             const std::set<int>& sizeCapBoundaries, const std::map<int, Value>& materialized,
                                             std::vector<Value>& externalInputs, std::map<int, int>& externalIndexByOldId,
                                             std::vector<FusedStep>& steps, std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) || (p->op == OpKind::Sum) ||
                           (consumers.at(oldId) != 1) || (sizeCapBoundaries.count(oldId) > 0);
    if (mustBeExternal) {
        auto extIt = externalIndexByOldId.find(oldId);
        if (extIt != externalIndexByOldId.end()) return FusedOperand{OperandKind::ExternalInput, extIt->second};
        int idx = static_cast<int>(externalInputs.size());
        externalInputs.push_back(materialized.at(oldId));
        externalIndexByOldId[oldId] = idx;
        return FusedOperand{OperandKind::ExternalInput, idx};
    }
    std::vector<FusedOperand> operands;
    for (const Value& in : p->inputs)
        operands.push_back(resolveIntoGroupBounded(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                     externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}
static FusionResult boundedReductionFusionPass(const Graph& g, const std::map<int, long long>& elementCounts, long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("boundedReductionFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes()) for (const Value& in : n->inputs) consumers[in.nodeId]++;
    std::set<int> sizeCapBoundaries = computeSizeCapBoundaries(g, consumers, maxChainLength);

    FusionResult result;
    Graph& out = result.graph;
    std::map<int, Value> materialized;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        if (n->op == OpKind::Input) { Value v = out.addInput(n->debugName); materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue; }
        if (n->op == OpKind::Const) { Value v = out.addConst(n->constValue, n->debugName); materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue; }
        if (n->op == OpKind::Sum) {
            std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
            std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
            FusedOperand summedOperand = resolveIntoGroupBounded(n->inputs[0].nodeId, g, consumers, sizeCapBoundaries,
                                                                   materialized, externalInputs, externalIndexByOldId, steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0) continue;
        std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs)
            operands.push_back(resolveIntoGroupBounded(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                         externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
        steps.push_back(FusedStep{n->op, operands});
        Value v;
        if (steps.size() == 1) {
            if (n->op == OpKind::ReLU) v = out.addUnary(OpKind::ReLU, externalInputs[0], n->debugName);
            else v = out.addBinary(n->op, externalInputs[0], externalInputs[1], n->debugName);
        } else {
            v = out.addFusedElementwise(externalInputs, steps, n->debugName);
        }
        materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

// ==================== evaluateArrays() (from Section 17.1 / File 083, unchanged) ====================

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
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul || n->op == OpKind::ReLU) {
            long long count = elementCounts.at(id);
            buf.resize(static_cast<size_t>(count));
            const std::vector<float>& lhs = buffersById.at(n->inputs[0].nodeId);
            long long lhsCount = static_cast<long long>(lhs.size());
            if (n->op == OpKind::ReLU) {
                for (long long i = 0; i < count; ++i) buf[i] = std::max(0.0f, lhs[i % lhsCount]);
            } else {
                const std::vector<float>& rhs = buffersById.at(n->inputs[1].nodeId);
                long long rhsCount = static_cast<long long>(rhs.size());
                for (long long i = 0; i < count; ++i) {
                    float lv = lhs[i % lhsCount], rv = rhs[i % rhsCount];
                    buf[i] = (n->op == OpKind::Add) ? (lv + rv) : (lv * rv);
                }
            }
        } else if (n->op == OpKind::Sum) {
            const std::vector<float>& in = buffersById.at(n->inputs[0].nodeId);
            float acc = 0.0f;
            for (float v : in) acc += v;
            buf = {acc};
        } else {
            std::vector<const std::vector<float>*> ext;
            for (const Value& in : n->inputs) ext.push_back(&buffersById.at(in.nodeId));
            bool isReduction = (n->op == OpKind::FusedReduction);
            auto readOperand = [&](const FusedOperand& o, const std::vector<float>& stepVals, long long idx) -> float {
                if (o.kind == OperandKind::ExternalInput) {
                    const std::vector<float>& b = *ext[o.index];
                    return b[idx % static_cast<long long>(b.size())];
                }
                return stepVals[o.index];
            };
            if (!isReduction) {
                long long count = elementCounts.at(id);
                buf.resize(static_cast<size_t>(count));
                for (long long i = 0; i < count; ++i) {
                    std::vector<float> stepVals;
                    for (const FusedStep& step : n->fusedSteps) {
                        float sv;
                        if (step.op == OpKind::ReLU) sv = std::max(0.0f, readOperand(step.operands[0], stepVals, i));
                        else if (step.op == OpKind::Add) sv = readOperand(step.operands[0], stepVals, i) + readOperand(step.operands[1], stepVals, i);
                        else sv = readOperand(step.operands[0], stepVals, i) * readOperand(step.operands[1], stepVals, i);
                        stepVals.push_back(sv);
                    }
                    buf[i] = stepVals.back();
                }
            } else {
                long long reduceCount = n->fusedSteps.back().reduceElementCount;
                float acc = 0.0f;
                for (long long r = 0; r < reduceCount; ++r) {
                    std::vector<float> stepVals;
                    for (size_t s = 0; s + 1 < n->fusedSteps.size(); ++s) {
                        const FusedStep& step = n->fusedSteps[s];
                        float sv;
                        if (step.op == OpKind::ReLU) sv = std::max(0.0f, readOperand(step.operands[0], stepVals, r));
                        else if (step.op == OpKind::Add) sv = readOperand(step.operands[0], stepVals, r) + readOperand(step.operands[1], stepVals, r);
                        else sv = readOperand(step.operands[0], stepVals, r) * readOperand(step.operands[1], stepVals, r);
                        stepVals.push_back(sv);
                    }
                    acc += readOperand(n->fusedSteps.back().operands[0], stepVals, r);
                }
                buf = {acc};
            }
        }
        buffersById[id] = buf;
        buffersByName[n->debugName] = buf;
    }
    return buffersByName;
}

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2 / File 083, unchanged) ====================

struct Loop { std::string dimName; long long extent; };
struct LoopNest { std::vector<Loop> loops; };
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes, const std::map<int, long long>& elementCounts) {
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    if (n->op == OpKind::Sum) nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    else if (n->op == OpKind::FusedReduction) nest.loops.push_back(Loop{"reduce", n->fusedSteps.back().reduceElementCount});
    return nest;
}
struct LoweredNode { std::vector<FusedStep> steps; LoopNest nest; };
static LoweredNode lowerNode(const Node* n, const std::map<int, Shape>& shapes, const std::map<int, long long>& elementCounts) {
    LoweredNode result;
    if (n->op == OpKind::FusedElementwise || n->op == OpKind::FusedReduction) {
        result.steps = n->fusedSteps;
    } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
        result.steps = {FusedStep{n->op, {FusedOperand{OperandKind::ExternalInput, 0}, FusedOperand{OperandKind::ExternalInput, 1}}}};
    } else if (n->op == OpKind::ReLU) {
        result.steps = {FusedStep{OpKind::ReLU, {FusedOperand{OperandKind::ExternalInput, 0}}}};
    } else {
        long long count = elementCounts.at(n->inputs[0].nodeId);
        result.steps = {FusedStep{OpKind::Sum, {FusedOperand{OperandKind::ExternalInput, 0}}, count}};
    }
    result.nest = buildLoopNest(n, shapes, elementCounts);
    return result;
}

// ==================== emitSteps() / generateCpuElementwiseFunction() (from Section 18.1 / File 083, unchanged) ====================

enum class Target { Cpu, Cuda };
static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExprScalar(Target target, const std::string& x) {
    return (target == Target::Cpu) ? ("std::max(0.0f, " + x + ")") : ("fmaxf(0.0f, " + x + ")");
}
static std::vector<std::string> emitSteps(const std::vector<FusedStep>& steps, bool isReduction, const std::string& idxExpr,
                                           Target target, std::string& finalValueExpr) {
    std::vector<std::string> lines;
    std::vector<std::string> stepVars;
    size_t stepCount = isReduction ? steps.size() - 1 : steps.size();
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return readExternal(target, o.index, idxExpr);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < stepCount; ++s) {
        const FusedStep& step = steps[s];
        std::string expr;
        if (step.op == OpKind::ReLU) expr = maxExprScalar(target, operandExpr(step.operands[0]));
        else if (step.op == OpKind::Add) expr = operandExpr(step.operands[0]) + " + " + operandExpr(step.operands[1]);
        else expr = operandExpr(step.operands[0]) + " * " + operandExpr(step.operands[1]);
        std::string varName = "step" + std::to_string(s);
        lines.push_back("float " + varName + " = " + expr + ";");
        stepVars.push_back(varName);
    }
    if (isReduction) {
        const FusedOperand& sumOperand = steps.back().operands[0];
        finalValueExpr = (sumOperand.kind == OperandKind::ExternalInput) ? readExternal(target, sumOperand.index, idxExpr) : stepVars[static_cast<size_t>(sumOperand.index)];
    } else {
        finalValueExpr = stepVars.back();
    }
    return lines;
}
static std::string generateCpuElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n    }\n}\n";
    return src;
}

// ==================== Isa / vector primitives / FMA fold / vectorized reduction / dispatcher
//                      (from Sections 19.1-19.3 / File 083, unchanged) ====================

enum class Isa { Avx2, Neon };
static int vectorWidth(Isa isa) { return (isa == Isa::Avx2) ? 8 : 4; }
static std::string vecType(Isa isa) { return (isa == Isa::Avx2) ? "__m256" : "float32x4_t"; }
static std::string isaHeaderInclude(Isa isa) { return (isa == Isa::Avx2) ? "#include <immintrin.h>" : "#include <arm_neon.h>"; }
static std::string isaName(Isa isa) { return (isa == Isa::Avx2) ? "AVX2/FMA" : "NEON"; }
static std::string vecLoadOrBroadcast(Isa isa, int index, const std::string& baseExpr, bool isScalar) {
    std::string ext = "ext[" + std::to_string(index) + "]";
    if (isa == Isa::Avx2) return isScalar ? ("_mm256_set1_ps(" + ext + "[0])") : ("_mm256_loadu_ps(" + ext + " + " + baseExpr + ")");
    return isScalar ? ("vdupq_n_f32(" + ext + "[0])") : ("vld1q_f32(" + ext + " + " + baseExpr + ")");
}
static std::string vecAdd(Isa isa, const std::string& a, const std::string& b) { return (isa == Isa::Avx2) ? ("_mm256_add_ps(" + a + ", " + b + ")") : ("vaddq_f32(" + a + ", " + b + ")"); }
static std::string vecMul(Isa isa, const std::string& a, const std::string& b) { return (isa == Isa::Avx2) ? ("_mm256_mul_ps(" + a + ", " + b + ")") : ("vmulq_f32(" + a + ", " + b + ")"); }
static std::string vecMax(Isa isa, const std::string& x) { return (isa == Isa::Avx2) ? ("_mm256_max_ps(" + x + ", _mm256_setzero_ps())") : ("vmaxq_f32(" + x + ", vdupq_n_f32(0.0f))"); }
static std::string vecStore(Isa isa, const std::string& dst, const std::string& val) { return (isa == Isa::Avx2) ? ("_mm256_storeu_ps(" + dst + ", " + val + ");") : ("vst1q_f32(" + dst + ", " + val + ");"); }
static bool canVectorizeElementwise(const Node* n, const std::map<int, long long>& elementCounts) {
    long long outCount = elementCounts.at(n->id);
    for (const Value& in : n->inputs) { long long c = elementCounts.at(in.nodeId); if (c != outCount && c != 1) return false; }
    return true;
}
static std::string vecFma(Isa isa, const std::string& mulLhs, const std::string& mulRhs, const std::string& addend) {
    return (isa == Isa::Avx2) ? ("_mm256_fmadd_ps(" + mulLhs + ", " + mulRhs + ", " + addend + ")") : ("vfmaq_f32(" + addend + ", " + mulLhs + ", " + mulRhs + ")");
}
static std::vector<int> computeStepUseCounts(const std::vector<FusedStep>& steps, bool isReduction) {
    size_t stepCount = isReduction ? steps.size() - 1 : steps.size();
    std::vector<int> useCounts(stepCount, 0);
    for (size_t s = 0; s < stepCount; ++s) for (const FusedOperand& o : steps[s].operands) if (o.kind == OperandKind::PriorStep) useCounts[o.index]++;
    if (isReduction) { const FusedOperand& sumOperand = steps.back().operands[0]; if (sumOperand.kind == OperandKind::PriorStep) useCounts[sumOperand.index]++; }
    else if (stepCount > 0) useCounts[stepCount - 1]++;
    return useCounts;
}
static bool stepUsesOperand(const FusedStep& step, OperandKind kind, int index) {
    for (const FusedOperand& o : step.operands) if (o.kind == kind && o.index == index) return true;
    return false;
}
static std::string generateVectorElementwiseFunctionFma(const std::string& funcName, const LoweredNode& lowered, Isa isa,
                                                          const std::vector<bool>& isScalarInput, int& foldCount, int& rawStepCount) {
    const std::vector<FusedStep>& steps = lowered.steps;
    std::vector<int> useCounts = computeStepUseCounts(steps, false);
    rawStepCount = static_cast<int>(steps.size());
    foldCount = 0;
    std::vector<bool> foldedAway(steps.size(), false);
    for (size_t s = 0; s < steps.size(); ++s) {
        if (steps[s].op != OpKind::Add || s == 0) continue;
        if (steps[s - 1].op != OpKind::Mul) continue;
        if (useCounts[s - 1] != 1) continue;
        if (!stepUsesOperand(steps[s], OperandKind::PriorStep, static_cast<int>(s - 1))) continue;
        foldedAway[s - 1] = true; foldCount++;
    }
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {\n";
    int vw = vectorWidth(isa);
    src += "    long long i = 0;\n    for (; i + " + std::to_string(vw) + " <= n; i += " + std::to_string(vw) + ") {\n";
    std::vector<std::string> stepVars(steps.size());
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < steps.size(); ++s) {
        if (foldedAway[s]) continue;
        const FusedStep& step = steps[s];
        std::string varName = "v" + std::to_string(s);
        if (step.op == OpKind::Add && s > 0 && foldedAway[s - 1]) {
            const FusedStep& mulStep = steps[s - 1];
            std::string mulLhs = operandExpr(mulStep.operands[0]);
            std::string mulRhs = operandExpr(mulStep.operands[1]);
            std::string addend;
            for (const FusedOperand& o : step.operands) if (!(o.kind == OperandKind::PriorStep && o.index == static_cast<int>(s - 1))) addend = operandExpr(o);
            src += "        " + vecType(isa) + " " + varName + " = " + vecFma(isa, mulLhs, mulRhs, addend) + ";  // fused: v" + std::to_string(s - 1) + " folded in\n";
        } else if (step.op == OpKind::ReLU) {
            src += "        " + vecType(isa) + " " + varName + " = " + vecMax(isa, operandExpr(step.operands[0])) + ";\n";
        } else if (step.op == OpKind::Add) {
            src += "        " + vecType(isa) + " " + varName + " = " + vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        } else {
            src += "        " + vecType(isa) + " " + varName + " = " + vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        }
        stepVars[s] = varName;
    }
    src += "        " + vecStore(isa, "out + i", stepVars.back()) + "\n    }\n";
    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(steps, false, "i", Target::Cpu, finalExpr);
    src += "    for (; i < n; ++i) {\n";
    for (const std::string& line : tailLines) src += "        " + line + "\n";
    src += "        out[i] = " + finalExpr + ";\n    }\n}\n";
    return src;
}
static std::string generateVectorReductionFunction(const std::string& funcName, const LoweredNode& lowered, Isa isa, const std::vector<bool>& isScalarInput) {
    const std::vector<FusedStep>& steps = lowered.steps;
    size_t stepCount = steps.size() - 1;
    long long reduceExtent = lowered.nest.loops.back().extent;
    int vw = vectorWidth(isa);
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out) {\n";
    src += "    (void)extCounts;\n    " + vecType(isa) + " acc = " + (isa == Isa::Avx2 ? "_mm256_setzero_ps()" : "vdupq_n_f32(0.0f)") + ";\n";
    src += "    long long i = 0;\n    for (; i + " + std::to_string(vw) + " <= " + std::to_string(reduceExtent) + "; i += " + std::to_string(vw) + ") {\n";
    std::vector<std::string> stepVars(stepCount);
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < stepCount; ++s) {
        const FusedStep& step = steps[s];
        std::string varName = "v" + std::to_string(s);
        if (step.op == OpKind::ReLU) src += "        " + vecType(isa) + " " + varName + " = " + vecMax(isa, operandExpr(step.operands[0])) + ";\n";
        else if (step.op == OpKind::Add) src += "        " + vecType(isa) + " " + varName + " = " + vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        else src += "        " + vecType(isa) + " " + varName + " = " + vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        stepVars[s] = varName;
    }
    const FusedOperand& sumOperand = steps.back().operands[0];
    std::string sumExpr = (sumOperand.kind == OperandKind::ExternalInput) ? vecLoadOrBroadcast(isa, sumOperand.index, "i", isScalarInput[sumOperand.index]) : stepVars[static_cast<size_t>(sumOperand.index)];
    src += "        acc = " + vecAdd(isa, "acc", sumExpr) + ";\n    }\n";
    src += "    float lanes[" + std::to_string(vw) + "];\n    " + vecStore(isa, "lanes", "acc") + "\n";
    src += "    float hsum = 0.0f;\n    for (int lane = 0; lane < " + std::to_string(vw) + "; ++lane) hsum += lanes[lane];\n";
    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(steps, true, "i", Target::Cpu, finalExpr);
    src += "    for (; i < " + std::to_string(reduceExtent) + "; ++i) {\n";
    for (const std::string& line : tailLines) src += "        " + line + "\n";
    src += "        hsum += " + finalExpr + ";\n    }\n    out[0] = hsum;\n}\n";
    return src;
}
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }
enum class Backend { VectorFma, VectorReduction, ScalarFallback };
static std::string generateFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered, Isa isa,
                                            const std::map<int, long long>& elementCounts, Backend& backendUsed, int& foldCount) {
    foldCount = 0;
    std::vector<bool> isScalarInput;
    for (const Value& in : n->inputs) isScalarInput.push_back(elementCounts.at(in.nodeId) == 1);
    if (isReductionNode(n)) { backendUsed = Backend::VectorReduction; return generateVectorReductionFunction(funcName, lowered, isa, isScalarInput); }
    if (canVectorizeElementwise(n, elementCounts)) {
        backendUsed = Backend::VectorFma;
        int rawStepCount = 0;
        return generateVectorElementwiseFunctionFma(funcName, lowered, isa, isScalarInput, foldCount, rawStepCount);
    }
    backendUsed = Backend::ScalarFallback;
    return generateCpuElementwiseFunction(funcName, lowered);
}
static const char* backendName(Backend b) { return b == Backend::VectorFma ? "VECTOR+FMA" : b == Backend::VectorReduction ? "VECTOR-REDUCTION" : "SCALAR-FALLBACK"; }

static std::string formatLiteralArray(const std::vector<float>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
    out += "}";
    return out;
}
static std::string generateFullVectorizedProgram(const Graph& g, const std::map<int, Shape>& shapes, const std::map<int, long long>& elementCounts,
                                                   const std::map<std::string, std::vector<float>>& inputArrays, Isa isa,
                                                   std::vector<std::pair<std::string, Backend>>& dispatchLog, int& totalFolds) {
    TopoResult topo = topologicalSort(g);
    std::string prog = std::string(isaHeaderInclude(isa)) + "\n#include <cstdio>\n#include <vector>\n#include <map>\n#include <string>\n#include <algorithm>\n\n";
    std::map<int, std::string> funcNameById;
    totalFolds = 0;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        funcNameById[id] = funcName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        Backend backendUsed; int foldCount = 0;
        prog += generateFunctionForNode(funcName, n, lowered, isa, elementCounts, backendUsed, foldCount) + "\n";
        dispatchLog.push_back({n->debugName, backendUsed});
        totalFolds += foldCount;
    }
    prog += "int main() {\n    std::map<std::string, std::vector<float>> buf;\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input) prog += "    buf[\"" + n->debugName + "\"] = " + formatLiteralArray(inputArrays.at(n->debugName)) + ";\n";
        else if (n->op == OpKind::Const) prog += "    buf[\"" + n->debugName + "\"] = {" + std::to_string(n->constValue) + "f};\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        bool isReduction = isReductionNode(n);
        prog += "    buf[\"" + n->debugName + "\"] = std::vector<float>(" + std::to_string(elementCounts.at(id)) + ");\n";
        prog += "    {\n        std::vector<const float*> ext; std::vector<long long> extCounts;\n";
        for (const Value& in : n->inputs) {
            const Node* inNode = g.node(in.nodeId);
            prog += "        ext.push_back(buf[\"" + inNode->debugName + "\"].data()); extCounts.push_back(" + std::to_string(elementCounts.at(in.nodeId)) + ");\n";
        }
        if (isReduction) prog += "        " + funcNameById.at(id) + "(ext, extCounts, buf[\"" + n->debugName + "\"].data());\n    }\n";
        else prog += "        " + funcNameById.at(id) + "(ext, extCounts, buf[\"" + n->debugName + "\"].data(), " + std::to_string(elementCounts.at(id)) + ");\n    }\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        prog += "    printf(\"" + n->debugName + ":\");\n";
        prog += "    for (float v : buf[\"" + n->debugName + "\"]) printf(\" %.6f\", v);\n";
        prog += "    printf(\"\\n\");\n";
    }
    prog += "    return 0;\n}\n";
    return prog;
}

static std::string runShellCaptureAll(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed for: " + cmd);
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}
static void writeFile(const std::string& path, const std::string& content) { std::ofstream f(path); f << content; }
static std::map<std::string, std::vector<float>> parseNamedBuffers(const std::string& text) {
    std::map<std::string, std::vector<float>> result;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string name = line.substr(0, colon);
        std::istringstream rest(line.substr(colon + 1));
        std::vector<float> vals; float v;
        while (rest >> v) vals.push_back(v);
        result[name] = vals;
    }
    return result;
}
static bool arraysMatch(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

// ==================== Section 32.3: the 4 centered tokens and 3 class weight vectors (Sections
//                      32.1/32.2's own real data, reused unchanged) ====================

static std::vector<std::vector<float>> buildCenteredTokens() {
    return {
        {-3.5f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 3.5f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {-4.0f, -3.0f, -2.0f, -1.0f, 1.0f, 2.0f, 3.0f, 4.0f},
        {-50.0f, -50.0f, -50.0f, -50.0f, 50.0f, 50.0f, 50.0f, 50.0f},
    };
}
static std::array<std::array<float, 8>, 3> buildClassWeights() {
    return {{
        {1, 1, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 1, 1},
        {1, -1, 1, -1, 1, -1, 1, -1},
    }};
}

int main() {
    printf("=== Chapter 32 (Part 7): Section 32.3 -- the whole classifier, fused and vectorized for real ===\n\n");
    bool allOk = true;
    const int dim = 8;

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    // ---------------------------------------------------------------
    // Stage 0: Section 32.2's own graph, rebuilt exactly (bag-of-
    // embeddings average, then 3 classifier branches sharing `bag` as a
    // real multi-consumer fan-out).
    // ---------------------------------------------------------------
    printf("--- Stage 0: rebuilding Section 32.2's own graph exactly ---\n\n");
    std::vector<std::vector<float>> tokens = buildCenteredTokens();
    std::array<std::array<float, 8>, 3> classWeights = buildClassWeights();
    std::array<float, 3> classBias = {0.0f, 0.0f, 5.0f};

    Graph g;
    std::array<Value, 4> tokenVals;
    for (int t = 0; t < 4; ++t) tokenVals[static_cast<size_t>(t)] = g.addInput("token" + std::to_string(t));
    Value quarter = g.addConst(0.25f, "quarter");
    std::array<Value, 4> terms;
    for (int t = 0; t < 4; ++t) terms[static_cast<size_t>(t)] = g.addBinary(OpKind::Mul, tokenVals[static_cast<size_t>(t)], quarter, "term" + std::to_string(t));
    Value bag = terms[0];
    for (int t = 1; t < 4; ++t) {
        std::string name = (t == 3) ? "bag" : ("bag_partial" + std::to_string(t));
        bag = g.addBinary(OpKind::Add, bag, terms[static_cast<size_t>(t)], name);
    }
    std::array<Value, 3> classWeightVals;
    std::array<Value, 3> logits;
    std::map<std::string, std::vector<float>> inputArrays;
    for (int t = 0; t < 4; ++t) inputArrays["token" + std::to_string(t)] = tokens[static_cast<size_t>(t)];
    for (int c = 0; c < 3; ++c) {
        std::string wname = "class" + std::to_string(c) + "_weights";
        classWeightVals[static_cast<size_t>(c)] = g.addInput(wname);
        std::vector<float> wvec(classWeights[static_cast<size_t>(c)].begin(), classWeights[static_cast<size_t>(c)].end());
        inputArrays[wname] = wvec;
        Value prod = g.addBinary(OpKind::Mul, bag, classWeightVals[static_cast<size_t>(c)], "prod" + std::to_string(c));
        Value sum = g.addUnary(OpKind::Sum, prod, "sum" + std::to_string(c));
        Value biasConst = g.addConst(classBias[static_cast<size_t>(c)], "bias" + std::to_string(c));
        logits[static_cast<size_t>(c)] = g.addBinary(OpKind::Add, sum, biasConst, "logit" + std::to_string(c));
    }
    (void)logits;

    std::map<int, Shape> declared;
    for (const auto& n : g.nodes())
        if (n->op == OpKind::Input || n->op == OpKind::Const) declared[n->id] = (n->op == OpKind::Input) ? Shape{{dim}} : Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    auto originalResult = evaluateArrays(g, inputArrays, elementCounts);
    printf("evaluateArrays() on the ORIGINAL (unfused) graph: logit0=%.4f logit1=%.4f logit2=%.4f\n\n",
           static_cast<double>(originalResult.at("logit0")[0]), static_cast<double>(originalResult.at("logit1")[0]),
           static_cast<double>(originalResult.at("logit2")[0]));

    // ---------------------------------------------------------------
    // Stage 1: Chapter 14/16's real fusion pass. bag feeds 3 consumers
    // (consumers=3), so it must stay external -- the same real fan-out
    // rule Section 31.3's shift inputs exercised, now over a classifier
    // instead of two spatial filters.
    // ---------------------------------------------------------------
    printf("--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---\n\n");
    FusionResult fr = boundedReductionFusionPass(g, elementCounts, 3);
    const Graph& fused = fr.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fr.representativeOldId.at(n->id);
        fusedShapes[n->id] = shapes.at(oldId);
        fusedElementCounts[n->id] = elementCounts.at(oldId);
    }
    int fusedElementwiseCount = 0, fusedReductionCount = 0, plainCount = 0;
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::FusedElementwise) fusedElementwiseCount++;
        else if (n->op == OpKind::FusedReduction) fusedReductionCount++;
        else if (n->op != OpKind::Input && n->op != OpKind::Const) plainCount++;
    }
    long long origInputOrConstCount = 0;
    for (const auto& n : g.nodes()) if (n->op == OpKind::Input || n->op == OpKind::Const) origInputOrConstCount++;
    long long origComputeNodeCount = static_cast<long long>(g.size()) - origInputOrConstCount;
    printf("fused graph: %zu nodes total -- %d FusedElementwise group(s), %d FusedReduction group(s),\n",
           fused.size(), fusedElementwiseCount, fusedReductionCount);
    printf("%d plain (unfused) compute node(s), down from %lld compute nodes in the original graph\n\n",
           plainCount, origComputeNodeCount);

    // ---------------------------------------------------------------
    // Stage 2: Chapter 19's real per-node vectorized-CPU dispatcher.
    // ---------------------------------------------------------------
    printf("--- Stage 2: Chapter 19's real per-node vectorized dispatcher, compiled and run on %s ---\n\n", isaName(hostIsa).c_str());
    std::vector<std::pair<std::string, Backend>> dispatchLog;
    int totalFolds = 0;
    std::string prog = generateFullVectorizedProgram(fused, fusedShapes, fusedElementCounts, inputArrays, hostIsa, dispatchLog, totalFolds);
    printf("per-node dispatch (%zu compute nodes):\n", dispatchLog.size());
    for (const auto& entry : dispatchLog) printf("  %-10s -> %s\n", entry.first.c_str(), backendName(entry.second));
    bool noScalarFallback = std::all_of(dispatchLog.begin(), dispatchLog.end(), [](const std::pair<std::string, Backend>& e) { return e.second != Backend::ScalarFallback; });
    printf("\nself-check: every dispatched node is genuinely vectorizable (no broadcast beyond a scalar\n");
    printf("anywhere in this graph), so 0 of %zu nodes fell back to the scalar path (%s)\n", dispatchLog.size(),
           noScalarFallback ? "confirmed" : "MISMATCH");
    printf("self-check: %d total FMA fold(s) applied across the fused elementwise groups\n\n", totalFolds);

    std::string stem = "/tmp/hammer_ch32_086_capstone";
    writeFile(stem + ".cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags + " " + stem + ".cpp -o " + stem + " 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags, %zu generated functions) ---\n\n%s\n", isaName(hostIsa).c_str(), dispatchLog.size(),
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());

    std::string runOutput = runShellCaptureAll(stem);
    auto generatedArrays = parseNamedBuffers(runOutput);

    bool allNodesMatch = true;
    auto fusedOriginalResult = evaluateArrays(fused, inputArrays, fusedElementCounts);
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        bool ok = generatedArrays.count(n->debugName) && arraysMatch(generatedArrays.at(n->debugName), fusedOriginalResult.at(n->debugName));
        if (!ok) allNodesMatch = false;
    }
    bool logitsMatch = true;
    for (int c = 0; c < 3; ++c) {
        std::string ln = "logit" + std::to_string(c);
        if (!generatedArrays.count(ln) || std::fabs(generatedArrays.at(ln)[0] - originalResult.at(ln)[0]) > 1e-2f) logitsMatch = false;
    }
    printf("--- the real logits, computed 3 independent ways ---\n\n");
    for (int c = 0; c < 3; ++c) {
        std::string ln = "logit" + std::to_string(c);
        printf("logit%d: interpreted(original)=%.4f interpreted(fused)=%.4f real-vectorized(%s)=%.4f\n",
               c, static_cast<double>(originalResult.at(ln)[0]), static_cast<double>(fusedOriginalResult.at(ln)[0]),
               isaName(hostIsa).c_str(), static_cast<double>(generatedArrays.at(ln)[0]));
    }
    printf("\nself-check: all three agree for all 3 logits -- the fusion pass and the vectorized\n");
    printf("dispatcher both preserve the original graph's own answer exactly (%s)\n", logitsMatch ? "confirmed" : "MISMATCH");
    printf("self-check: every intermediate buffer in the fused graph (not just the 3 logits) matches\n");
    printf("between evaluateArrays() and the real compiled-and-run vectorized code (%s)\n\n",
           allNodesMatch ? "confirmed" : "MISMATCH");

    allOk = allOk && compileClean && logitsMatch && allNodesMatch && noScalarFallback;

    printf("=== Section 32.3 complete, Chapter 32 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 [-mavx2 -mfma added automatically on x86_64] 086_the_whole_classifier_fused_and_vectorized_for_real.cpp -o 086_the_whole_classifier_fused_and_vectorized_for_real_driver
./086_the_whole_classifier_fused_and_vectorized_for_real_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 32 (Part 7): Section 32.3 -- the whole classifier, fused and vectorized for real ===

--- Stage 0: rebuilding Section 32.2's own graph exactly ---

evaluateArrays() on the ORIGINAL (unfused) graph: logit0=-54.5000 logit1=54.5000 logit2=3.0000

--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---

fused graph: 19 nodes total -- 2 FusedElementwise group(s), 3 FusedReduction group(s),
3 plain (unfused) compute node(s), down from 16 compute nodes in the original graph

--- Stage 2: Chapter 19's real per-node vectorized dispatcher, compiled and run on AVX2/FMA ---

per-node dispatch (8 compute nodes):
  bag_partial2 -> VECTOR+FMA
  bag        -> VECTOR+FMA
  sum0       -> VECTOR-REDUCTION
  sum1       -> VECTOR-REDUCTION
  sum2       -> VECTOR-REDUCTION
  logit0     -> VECTOR+FMA
  logit1     -> VECTOR+FMA
  logit2     -> VECTOR+FMA

self-check: every dispatched node is genuinely vectorizable (no broadcast beyond a scalar
anywhere in this graph), so 0 of 8 nodes fell back to the scalar path (confirmed)
self-check: 3 total FMA fold(s) applied across the fused elementwise groups

--- g++ compile (real AVX2/FMA flags, 8 generated functions) ---

(no output -- clean compile)

--- the real logits, computed 3 independent ways ---

logit0: interpreted(original)=-54.5000 interpreted(fused)=-54.5000 real-vectorized(AVX2/FMA)=-54.5000
logit1: interpreted(original)=54.5000 interpreted(fused)=54.5000 real-vectorized(AVX2/FMA)=54.5000
logit2: interpreted(original)=3.0000 interpreted(fused)=3.0000 real-vectorized(AVX2/FMA)=3.0000

self-check: all three agree for all 3 logits -- the fusion pass and the vectorized
dispatcher both preserve the original graph's own answer exactly (confirmed)
self-check: every intermediate buffer in the fused graph (not just the 3 logits) matches
between evaluateArrays() and the real compiled-and-run vectorized code (confirmed)

=== Section 32.3 complete, Chapter 32 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 32 (Part 7): Section 32.3 -- the whole classifier, fused and vectorized for real ===

--- Stage 0: rebuilding Section 32.2's own graph exactly ---

evaluateArrays() on the ORIGINAL (unfused) graph: logit0=-54.5000 logit1=54.5000 logit2=3.0000

--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---

fused graph: 19 nodes total -- 2 FusedElementwise group(s), 3 FusedReduction group(s),
3 plain (unfused) compute node(s), down from 16 compute nodes in the original graph

--- Stage 2: Chapter 19's real per-node vectorized dispatcher, compiled and run on NEON ---

per-node dispatch (8 compute nodes):
  bag_partial2 -> VECTOR+FMA
  bag        -> VECTOR+FMA
  sum0       -> VECTOR-REDUCTION
  sum1       -> VECTOR-REDUCTION
  sum2       -> VECTOR-REDUCTION
  logit0     -> VECTOR+FMA
  logit1     -> VECTOR+FMA
  logit2     -> VECTOR+FMA

self-check: every dispatched node is genuinely vectorizable (no broadcast beyond a scalar
anywhere in this graph), so 0 of 8 nodes fell back to the scalar path (confirmed)
self-check: 3 total FMA fold(s) applied across the fused elementwise groups

--- g++ compile (real NEON flags, 8 generated functions) ---

(no output -- clean compile)

--- the real logits, computed 3 independent ways ---

logit0: interpreted(original)=-54.5000 interpreted(fused)=-54.5000 real-vectorized(NEON)=-54.5000
logit1: interpreted(original)=54.5000 interpreted(fused)=54.5000 real-vectorized(NEON)=54.5000
logit2: interpreted(original)=3.0000 interpreted(fused)=3.0000 real-vectorized(NEON)=3.0000

self-check: all three agree for all 3 logits -- the fusion pass and the vectorized
dispatcher both preserve the original graph's own answer exactly (confirmed)
self-check: every intermediate buffer in the fused graph (not just the 3 logits) matches
between evaluateArrays() and the real compiled-and-run vectorized code (confirmed)

=== Section 32.3 complete, Chapter 32 complete: all self-checks confirmed ===
```

*Identical apart from the ISA name itself (AVX2/FMA on the cloud sandbox, NEON on the device, each machine's own real vector hardware) -- all 3 logits agree exactly across all three computations (interpreted-original, interpreted-fused, real-vectorized-compiled) on both machines, because this classifier's own arithmetic never needs the extra FMA-fold rounding room Chapter 31's larger pipeline did.*


## What Chapter 32 actually shows

Nothing here is a simulation. Section 32.1's four measured means are
real numbers CUDA Hammer's own `Sum` op produced by actually running
`evaluateArrays()` four separate times, not values chosen to make an
invariant come out right -- and the constant-vector and already-zero-
mean edge cases are checked against their own real re-measured output,
not merely asserted. Section 32.2's bag vector and three logits are a
real CUDA Hammer graph's real output, checked against an independent
host-side reference that never touches CUDA Hammer at all. Section
32.3's three logits are the same real numbers three separate times:
from the original graph's interpreter, from the SAME graph after
Chapter 14/16's real fusion pass rewrote it (with `bag`'s genuine
3-way fan-out forcing it to stay a real boundary node, not silently
duplicated), and from real AVX2/FMA or NEON machine code Chapter 19's
real codegen generated, compiled, and ran.

Chapter 31 found a missing primitive (no "read my neighbor" op) and
routed around it with pure host-side staging, before the graph ever
saw a shifted pixel. Chapter 32 found a different kind of gap -- no
batched or per-row reduction -- and routed around it the same honest
way: not with a new `OpKind`, but with the host repeating a real,
already-existing two-pass graph pattern once per token. Both chapters
also hit a real boundary this IR has never crossed (no `sqrt`, no
division, no `exp`) and said so directly rather than faking the
missing half with a hardcoded stand-in. That is Part 7's own opening
promise, kept twice now in two different domains: no new `OpKind`, no
new pass, no new backend -- only real domain-shaped graph construction
and, where the op set genuinely cannot do something on its own, an
honest, citable, correctly-costed staging technique instead.

Chapter 33 moves to a third domain. Part 7's own TOC entry records the
remaining candidates -- scientific computing and quantitative finance --
still to be built.
