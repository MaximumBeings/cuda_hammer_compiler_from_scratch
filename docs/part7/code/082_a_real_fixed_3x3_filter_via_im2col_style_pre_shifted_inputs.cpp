// Chapter 31: Computer Vision -- A Real Fixed-Kernel Preprocessing Pipeline
// 082_a_real_fixed_3x3_filter_via_im2col_style_pre_shifted_inputs.cpp
//
// Section 31.2 -- CUDA Hammer's IR (Input, Const, Add, Mul, ReLU, Sum) has
// no "shift a tensor by an offset" or "slice a 2D neighborhood" op --
// Chapters 4 through 30 never needed one, and this book's own Part 7
// promise (Getting Started, restated at the top of File 081) is no new
// toolchain, no hidden extra machinery added just for this Part. So a
// fixed 3x3 filter -- a box blur, and a Sobel-style edge detector -- is
// built the way real convolution libraries built it before implicit-GEMM
// kernels existed: im2col. NVIDIA's own CUTLASS documentation describes
// it plainly: "The earliest form of this algorithm constructs the
// convolution matrix explicitly via an operation conventionally referred
// to as im2col" -- and names its real cost too: "The resulting matrix
// replicates each activation element by a factor equal the filter size,
// consuming additional storage capacity and memory bandwidth."
// (https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html)
//
// This section does exactly that shift, but HONESTLY split across the
// boundary this book has drawn since Chapter 5: the 9 neighbor SHIFTS
// (data movement, zero-padded at the border) happen in host-side C++
// before CUDA Hammer ever sees the data -- exactly like im2col's own
// "construct the convolution matrix explicitly" step. The actual
// multiply-accumulate -- 9 real Mul-by-a-fixed-weight-Const steps folded
// into an explicit Add chain -- is a real CUDA Hammer graph, built,
// shape-inferred, and evaluated by this book's own evaluateArrays(),
// exactly like every other graph in this book. A direct 2D convolution,
// written as an independent nested-loop reference with NO im2col shift at
// all, checks that the graph's own answer is exactly right.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 082_a_real_fixed_3x3_filter_via_im2col_style_pre_shifted_inputs.cpp -o 082_driver
// Run:     ./082_driver
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

// ==================== Section 31.2: the image (reused from File 081, unchanged) ====================

static std::vector<float> buildStepImage() {
    std::vector<float> img(64);
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c)
            img[static_cast<size_t>(r * 8 + c)] = (c < 4) ? 10.0f : 200.0f;
    return img;
}

// ==================== im2col-style host-side shift (the "construct the convolution
//                      matrix explicitly" step CUTLASS's own docs describe) ====================
//
// For each of the 9 (dr,dc) offsets in a 3x3 neighborhood, builds ONE
// shifted copy of the whole 8x8 image, zero-padded at the border. This is
// real data movement, done in plain host C++ -- CUDA Hammer's own graph
// never sees an "offset"; it only ever sees 9 same-shaped [64] arrays and
// combines them with Mul/Add, exactly like every other elementwise graph
// in this book.
static std::vector<float> shiftImage(const std::vector<float>& img, int dr, int dc) {
    std::vector<float> out(64, 0.0f);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int sr = r + dr, sc = c + dc;
            float v = 0.0f;
            if (sr >= 0 && sr < 8 && sc >= 0 && sc < 8) v = img[static_cast<size_t>(sr * 8 + sc)];
            out[static_cast<size_t>(r * 8 + c)] = v;
        }
    }
    return out;
}

// Independent reference: a direct 2D convolution with NO im2col shift at
// all -- nested loops over rows/columns, reading neighbors straight out
// of the original 2D image. This is what the im2col-based graph is
// checked against.
static std::vector<float> directConvolve2D(const std::vector<float>& img, const std::array<float, 9>& kernel) {
    std::vector<float> out(64, 0.0f);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            float acc = 0.0f;
            int k = 0;
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    int sr = r + dr, sc = c + dc;
                    float v = (sr >= 0 && sr < 8 && sc >= 0 && sc < 8) ? img[static_cast<size_t>(sr * 8 + sc)] : 0.0f;
                    acc += v * kernel[static_cast<size_t>(k)];
                    ++k;
                }
            }
            out[static_cast<size_t>(r * 8 + c)] = acc;
        }
    }
    return out;
}

// Builds a real CUDA Hammer graph: 9 Input nodes (one per pre-shifted
// array), each multiplied by its own fixed weight Const, folded together
// through 8 real Add nodes -- an explicit, unrolled weighted sum using
// nothing but the ops Chapters 4/13/14 already built.
static Value buildConvGraph(Graph& g, const std::array<Value, 9>& shiftedInputs, const std::array<float, 9>& kernel,
                             const std::string& outName) {
    std::vector<Value> terms;
    for (int k = 0; k < 9; ++k) {
        Value w = g.addConst(kernel[static_cast<size_t>(k)], "w" + std::to_string(k));
        terms.push_back(g.addBinary(OpKind::Mul, shiftedInputs[static_cast<size_t>(k)], w, "term" + std::to_string(k)));
    }
    Value acc = terms[0];
    for (int k = 1; k < 9; ++k) {
        std::string name = (k == 8) ? outName : ("partial" + std::to_string(k));
        acc = g.addBinary(OpKind::Add, acc, terms[static_cast<size_t>(k)], name);
    }
    return acc;
}

static bool arraysMatch(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-3f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    printf("=== Chapter 31 (Part 7): Section 31.2 -- a real fixed 3x3 filter via im2col-style pre-shifted inputs ===\n\n");
    bool allOk = true;

    std::vector<float> img = buildStepImage();

    // 9 offsets in row-major order: (-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)
    std::array<std::pair<int,int>, 9> offsets = {{ {-1,-1},{-1,0},{-1,1}, {0,-1},{0,0},{0,1}, {1,-1},{1,0},{1,1} }};

    // ---------------------------------------------------------------
    // Part 1: box blur. Real im2col-style shift, real CUDA Hammer graph,
    // checked against a real direct 2D convolution.
    // ---------------------------------------------------------------
    printf("--- Part 1: 3x3 box blur (weight 1/9 on all 9 neighbors) ---\n\n");
    std::array<float, 9> boxKernel;
    boxKernel.fill(1.0f / 9.0f);

    Graph blurGraph;
    std::array<Value, 9> blurShiftedValues;
    std::map<std::string, std::vector<float>> blurInputs;
    for (int k = 0; k < 9; ++k) {
        std::string name = "shift" + std::to_string(k);
        blurShiftedValues[static_cast<size_t>(k)] = blurGraph.addInput(name);
        blurInputs[name] = shiftImage(img, offsets[static_cast<size_t>(k)].first, offsets[static_cast<size_t>(k)].second);
    }
    Value blurOut = buildConvGraph(blurGraph, blurShiftedValues, boxKernel, "blurred");
    (void)blurOut;

    std::map<int, Shape> blurDeclared;
    for (const auto& n : blurGraph.nodes())
        if (n->op == OpKind::Input || n->op == OpKind::Const) blurDeclared[n->id] = (n->op == OpKind::Input) ? Shape{{64}} : Shape{};
    std::map<int, Shape> blurShapes = inferShapes(blurGraph, blurDeclared);
    std::map<int, long long> blurCounts;
    for (const auto& kv : blurShapes) blurCounts[kv.first] = numElements(kv.second);
    auto blurResult = evaluateArrays(blurGraph, blurInputs, blurCounts);
    const std::vector<float>& blurred = blurResult.at("blurred");

    std::vector<float> blurRef = directConvolve2D(img, boxKernel);
    bool blurMatches = arraysMatch(blurred, blurRef);
    printf("im2col-graph blurred[3] (near the edge, row 0 col 3) = %.4f, direct-conv2D reference = %.4f\n",
           blurred[3], blurRef[3]);
    printf("im2col-graph blurred[27] (interior of the dark block, row 3 col 3) = %.4f, direct-conv2D reference = %.4f\n",
           blurred[27], blurRef[27]);
    printf("self-check: all 64 pixels of the im2col-style CUDA Hammer graph match the independent\n");
    printf("direct 2D convolution reference exactly (%s)\n\n", blurMatches ? "confirmed" : "MISMATCH");
    allOk = allOk && blurMatches;

    // ---------------------------------------------------------------
    // Part 2: the classic Sobel horizontal-gradient kernel, ReLU'd to
    // keep only the positive-direction (dark-to-bright, left-to-right)
    // edge response. Same im2col-style graph-building code, a different
    // kernel and a trailing ReLU node.
    // ---------------------------------------------------------------
    printf("--- Part 2: Sobel horizontal-gradient edge filter, ReLU'd to positive edges only ---\n\n");
    // Row-major 3x3 kernel matching the offset order above:
    // (-1,-1)=-1 (-1,0)=0 (-1,1)=1 / (0,-1)=-2 (0,0)=0 (0,1)=2 / (1,-1)=-1 (1,0)=0 (1,1)=1
    std::array<float, 9> sobelKernel = { -1.0f, 0.0f, 1.0f,  -2.0f, 0.0f, 2.0f,  -1.0f, 0.0f, 1.0f };

    Graph edgeGraph;
    std::array<Value, 9> edgeShiftedValues;
    std::map<std::string, std::vector<float>> edgeInputs;
    for (int k = 0; k < 9; ++k) {
        std::string name = "shift" + std::to_string(k);
        edgeShiftedValues[static_cast<size_t>(k)] = edgeGraph.addInput(name);
        edgeInputs[name] = shiftImage(img, offsets[static_cast<size_t>(k)].first, offsets[static_cast<size_t>(k)].second);
    }
    Value edgeRaw = buildConvGraph(edgeGraph, edgeShiftedValues, sobelKernel, "sobel_raw");
    Value edgePositive = edgeGraph.addUnary(OpKind::ReLU, edgeRaw, "edges_positive");
    (void)edgePositive;

    std::map<int, Shape> edgeDeclared;
    for (const auto& n : edgeGraph.nodes())
        if (n->op == OpKind::Input || n->op == OpKind::Const) edgeDeclared[n->id] = (n->op == OpKind::Input) ? Shape{{64}} : Shape{};
    std::map<int, Shape> edgeShapes = inferShapes(edgeGraph, edgeDeclared);
    std::map<int, long long> edgeCounts;
    for (const auto& kv : edgeShapes) edgeCounts[kv.first] = numElements(kv.second);
    auto edgeResult = evaluateArrays(edgeGraph, edgeInputs, edgeCounts);
    const std::vector<float>& edgesPositive = edgeResult.at("edges_positive");

    std::vector<float> sobelRawRef = directConvolve2D(img, sobelKernel);
    std::vector<float> edgesPositiveRef(64);
    for (size_t i = 0; i < 64; ++i) edgesPositiveRef[i] = std::max(0.0f, sobelRawRef[i]);
    bool edgeMatches = arraysMatch(edgesPositive, edgesPositiveRef);

    printf("row 3 (a middle row), all 8 columns of the ReLU'd Sobel response:\n  ");
    for (int c = 0; c < 8; ++c) printf("%7.2f ", edgesPositive[static_cast<size_t>(3 * 8 + c)]);
    printf("\n");
    printf("(the real vertical edge between columns 3 and 4 produces a strong positive response of\n");
    printf(" 760 at both columns 3 and 4; the small 40 at column 0 is a real zero-padding border\n");
    printf(" artifact, not an edge; every flat interior pixel is exactly 0 after ReLU -- exactly what\n");
    printf(" a horizontal-gradient detector should do to a single vertical step edge)\n\n");
    printf("self-check: all 64 pixels of the im2col-style ReLU'd Sobel graph match the independent\n");
    printf("direct 2D convolution + ReLU reference exactly (%s)\n\n", edgeMatches ? "confirmed" : "MISMATCH");
    allOk = allOk && edgeMatches;

    // ---------------------------------------------------------------
    // The honest cost this section's design pays, named directly:
    // im2col-style shifting replicates each of the 64 real pixels once
    // per filter tap (9 taps here), so this section's own host-side
    // shiftImage() calls allocate 9 * 64 = 576 floats of shifted storage
    // to filter a 64-pixel image -- a real, measured 9x blow-up, exactly
    // the cost CUTLASS's own documentation names for im2col.
    // ---------------------------------------------------------------
    long long originalPixels = 64;
    long long im2colStorage = 9 * 64;
    printf("--- the real cost this design pays (named by CUTLASS's own docs, not hidden) ---\n\n");
    printf("original image: %lld pixels. im2col-style shifted storage: 9 * %lld = %lld floats --\n",
           originalPixels, originalPixels, im2colStorage);
    printf("a real %.1fx blow-up, matching CUTLASS's own stated tradeoff for this exact technique.\n\n",
           static_cast<double>(im2colStorage) / static_cast<double>(originalPixels));

    printf("=== Section 31.2 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
