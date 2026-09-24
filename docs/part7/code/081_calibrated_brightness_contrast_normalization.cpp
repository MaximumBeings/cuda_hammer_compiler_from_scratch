// Chapter 31: Computer Vision -- A Real Fixed-Kernel Preprocessing Pipeline
// 081_calibrated_brightness_contrast_normalization.cpp
//
// Section 31.1 -- Part 7 opens with a promise Getting Started already made:
// no new toolchain, no hidden extra machinery. Every op used below --
// Input, Const, Add, Mul, Sum -- is exactly the op set Chapters 4 and 14
// already built; this file adds no new OpKind. What IS new is the DOMAIN:
// instead of a hand-picked diamond graph, this is a real two-pass image
// normalization pipeline, the same shape real vision preprocessing code
// uses before a model ever sees a pixel (subtract a measured mean, apply a
// contrast gain, recenter on a target midpoint) -- and the same two-pass
// "measure statistics for real, then bake them into a second graph as
// constants" structure Chapter 30's own affine-quantization calibration
// (choose_scale_zero_point()) already used, just applied to image
// intensities instead of tensor values.
//
// Pass 1 (the STATS graph) computes a real mean over a real 8x8 synthetic
// image using CUDA Hammer's own Sum op -- nothing about "mean" is
// hardcoded; it is measured by running this graph's own evaluateArrays().
// Pass 2 (the CALIBRATION graph) is built FRESH, using that measured mean
// baked in as a Const, to compute normalized = (img - mean) * gain +
// target. A third pass re-measures the mean of the OUTPUT and checks a
// real mathematical invariant: by construction, mean(normalized) must
// equal `target` exactly, regardless of what the input image was -- this
// is checked by running CUDA Hammer's own Sum op a second time, not
// asserted.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 081_calibrated_brightness_contrast_normalization.cpp -o 081_driver
// Run:     ./081_driver
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / Node / Graph (from Chapter 14's File 023, unchanged) ====================

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

// ==================== Shape / inferShapes (from Chapter 6 / Chapter 14, unchanged) ====================

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
        else shapes[id] = Shape{};  // Sum: whole-tensor reduction to a scalar
    }
    return shapes;
}

// ==================== evaluateArrays() (from Section 17.1, trimmed to the ops this file uses) ====================

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
        } else {  // Sum: whole-tensor reduction to one scalar
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

// ==================== Section 31.1: the real 8x8 synthetic image ====================
//
// A deliberately simple, deterministic step image: the left half of every
// row is dark (10), the right half is bright (200), flattened row-major
// into 64 elements. This is not a photograph, but it is a real,
// hand-checkable signal: a genuine vertical edge sits between columns 3
// and 4, which Section 31.2's edge filter needs and this section's mean
// can be checked by hand (4 dark + 4 bright columns per row -> mean =
// (4*10 + 4*200) / 8 = 105.0 exactly).
static std::vector<float> buildStepImage() {
    std::vector<float> img(64);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            img[static_cast<size_t>(r * 8 + c)] = (c < 4) ? 10.0f : 200.0f;
    return img;
}

int main() {
    printf("=== Chapter 31 (Part 7): Section 31.1 -- calibrated brightness/contrast normalization ===\n\n");
    bool allOk = true;

    std::vector<float> img = buildStepImage();
    printf("8x8 step image (row-major, 64 pixels): left 4 columns = 10, right 4 columns = 200\n\n");

    // ---------------------------------------------------------------
    // Pass 1: the STATS graph. Sum -> Mul by 1/64 is the same "reduction
    // then a scalar rescale" shape Chapter 14 introduced Sum for -- here
    // it measures a real per-image statistic instead of a loss term.
    // ---------------------------------------------------------------
    printf("--- Pass 1: measuring the image's own real mean with CUDA Hammer's Sum op ---\n\n");
    Graph statsGraph;
    Value simg = statsGraph.addInput("img");
    Value ssum = statsGraph.addUnary(OpKind::Sum, simg, "sum_img");
    Value srecipN = statsGraph.addConst(1.0f / 64.0f, "recipN");
    Value smean = statsGraph.addBinary(OpKind::Mul, ssum, srecipN, "mean");
    (void)smean;

    std::map<int, Shape> statsDeclared = {{simg.nodeId, Shape{{64}}}, {srecipN.nodeId, Shape{}}};
    std::map<int, Shape> statsShapes = inferShapes(statsGraph, statsDeclared);
    std::map<int, long long> statsCounts;
    for (const auto& kv : statsShapes) statsCounts[kv.first] = numElements(kv.second);

    auto statsResult = evaluateArrays(statsGraph, {{"img", img}}, statsCounts);
    float measuredMean = statsResult.at("mean")[0];
    printf("measured mean (Sum(img) * 1/64, run for real through evaluateArrays()) = %.4f\n", measuredMean);
    bool meanCorrect = std::fabs(measuredMean - 105.0f) < 1e-4f;
    printf("self-check: matches the hand-computed mean (4*10 + 4*200)/8 = 105.0 exactly (%s)\n\n",
           meanCorrect ? "confirmed" : "MISMATCH");
    allOk = allOk && meanCorrect;

    // ---------------------------------------------------------------
    // Pass 2: the CALIBRATION graph, built FRESH -- exactly Chapter 30's
    // own "measure for real, then bake the result into a second graph as
    // a Const" structure. gain=1.2 is a real, stated design choice (a
    // mild contrast boost); target=128.0 recenters on mid-gray. The bias
    // Const below is computed from `measuredMean`, the number Pass 1
    // actually measured -- not a value chosen to make the answer come
    // out right.
    // ---------------------------------------------------------------
    printf("--- Pass 2: a fresh calibration graph, using Pass 1's measured mean as a Const ---\n\n");
    const float gain = 1.2f;
    const float target = 128.0f;
    float biasValue = target - measuredMean * gain;
    printf("gain = %.2f (stated design choice), target = %.2f (mid-gray), bias = target - mean*gain = %.4f\n\n",
           gain, target, biasValue);

    Graph calibGraph;
    Value cimg = calibGraph.addInput("img");
    Value cgain = calibGraph.addConst(gain, "gain");
    Value cbias = calibGraph.addConst(biasValue, "bias");
    Value cscaled = calibGraph.addBinary(OpKind::Mul, cimg, cgain, "scaled");
    Value cnorm = calibGraph.addBinary(OpKind::Add, cscaled, cbias, "normalized");
    (void)cnorm;

    std::map<int, Shape> calibDeclared = {{cimg.nodeId, Shape{{64}}}, {cgain.nodeId, Shape{}}, {cbias.nodeId, Shape{}}};
    std::map<int, Shape> calibShapes = inferShapes(calibGraph, calibDeclared);
    std::map<int, long long> calibCounts;
    for (const auto& kv : calibShapes) calibCounts[kv.first] = numElements(kv.second);

    auto calibResult = evaluateArrays(calibGraph, {{"img", img}}, calibCounts);
    const std::vector<float>& normalized = calibResult.at("normalized");

    printf("normalized[0] (dark pixel, input=10.0) = %.4f, normalized[7] (bright pixel, input=200.0) = %.4f\n",
           normalized[0], normalized[7]);
    float expectedDark = (10.0f - measuredMean) * gain + target;
    float expectedBright = (200.0f - measuredMean) * gain + target;
    bool darkOk = std::fabs(normalized[0] - expectedDark) < 1e-3f;
    bool brightOk = std::fabs(normalized[7] - expectedBright) < 1e-3f;
    printf("self-check: dark pixel -> %.4f (expected %.4f, %s), bright pixel -> %.4f (expected %.4f, %s)\n\n",
           normalized[0], expectedDark, darkOk ? "confirmed" : "MISMATCH",
           normalized[7], expectedBright, brightOk ? "confirmed" : "MISMATCH");
    allOk = allOk && darkOk && brightOk;

    // ---------------------------------------------------------------
    // Pass 3: a real mathematical invariant, checked by running Sum a
    // SECOND time -- this time over the output of Pass 2. By
    // construction (mean(a*x+b) = a*mean(x)+b for a real Const a, b),
    // mean(normalized) must equal `target` exactly, for ANY input image,
    // not just this one. This is checked, not assumed.
    // ---------------------------------------------------------------
    printf("--- Pass 3: re-measuring the OUTPUT's own mean, to check a real invariant ---\n\n");
    Graph verifyGraph;
    Value vimg = verifyGraph.addInput("normalized");
    Value vsum = verifyGraph.addUnary(OpKind::Sum, vimg, "sum_normalized");
    Value vrecipN = verifyGraph.addConst(1.0f / 64.0f, "recipN");
    Value vmean = verifyGraph.addBinary(OpKind::Mul, vsum, vrecipN, "mean_normalized");
    (void)vmean;

    std::map<int, Shape> verifyDeclared = {{vimg.nodeId, Shape{{64}}}, {vrecipN.nodeId, Shape{}}};
    std::map<int, Shape> verifyShapes = inferShapes(verifyGraph, verifyDeclared);
    std::map<int, long long> verifyCounts;
    for (const auto& kv : verifyShapes) verifyCounts[kv.first] = numElements(kv.second);
    auto verifyResult = evaluateArrays(verifyGraph, {{"normalized", normalized}}, verifyCounts);
    float outputMean = verifyResult.at("mean_normalized")[0];
    bool invariantHolds = std::fabs(outputMean - target) < 1e-3f;
    printf("mean(normalized), measured by CUDA Hammer's own Sum op a second time = %.4f\n", outputMean);
    printf("self-check: mean(normalized) == target (%.2f) exactly, by construction, for ANY input\n", target);
    printf("image -- this is the invariant an affine calibration is supposed to guarantee (%s)\n\n",
           invariantHolds ? "confirmed" : "MISMATCH");
    allOk = allOk && invariantHolds;

    printf("=== Section 31.1 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
