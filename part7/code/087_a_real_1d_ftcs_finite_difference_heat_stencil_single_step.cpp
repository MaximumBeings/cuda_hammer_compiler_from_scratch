// Chapter 33: Scientific Computing -- A Real Finite-Difference Heat Stencil
// 087_a_real_1d_ftcs_finite_difference_heat_stencil_single_step.cpp
//
// Section 33.1 -- Part 7's third domain, and a THIRD kind of gap in the
// same fixed op set. Chapter 31 found no shift/gather primitive (worked
// around with host-side im2col shifts). Chapter 32 found no batched/
// per-row reduction (worked around with a host-side loop repeating a
// small graph once per token). This chapter's own real technique, the
// explicit (FTCS -- Forward Time, Centered Space) finite-difference
// update for the 1D heat equation, needs exactly the SAME "read my
// neighbor" shape Chapter 31.2 already solved: each output point reads
// its own left and right neighbor. So THIS section adds no new
// machinery at all -- it reuses Chapter 31.2's own im2col-style
// pre-shifted-input technique, unchanged, now in 1D instead of 2D.
//
// The real update formula and its real stability condition are quoted
// directly from a real numerical-analysis reference (John S. Butler,
// DIT, "The Explicit Forward Time Centered Space (FTCS) Difference
// Equation for the Heat Equation," Numerical Analysis book,
// https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2008%20-%20Heat%20Equations/801_Heat%20Equation-%20FTCS.html):
//   "w_{i,j+1} = r*w_{i-1,j} + (1-2r)*w_{i,j} + r*w_{i+1,j}"  where  "r = k/h^2"
// and its real stability condition: "r <= 1/2".
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 087_a_real_1d_ftcs_finite_difference_heat_stencil_single_step.cpp -o 087_driver
// Run:     ./087_driver
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

// ==================== Section 33.1: the real 1D domain, the im2col-style neighbor shift
//                      (Chapter 31.2's own shiftImage(), specialized to 1D), and the real
//                      FTCS graph ====================

// A deliberately small, hand-checkable 1D "hot spot": zero everywhere except
// two adjacent interior cells held at 100 -- symmetric around the domain's
// own center, so every later step stays symmetric too (a real, checkable
// invariant on its own).
static std::vector<float> buildInitialCondition() {
    return {0, 0, 0, 100, 100, 0, 0, 0};
}

// shift1D(w, offset): the SAME real technique as Chapter 31.2's shiftImage(),
// specialized from a 2D image to a 1D array -- zero-padded at the domain
// boundary, exactly the way Section 31.2 zero-padded at the image border.
// offset=-1 reads each point's LEFT neighbor; offset=+1 reads its RIGHT
// neighbor. This is real host-side C++, not a CUDA Hammer graph op -- the
// graph itself never sees an offset, only two already-shifted arrays.
static std::vector<float> shift1D(const std::vector<float>& w, int offset) {
    size_t n = w.size();
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        long long src = static_cast<long long>(i) + offset;
        if (src >= 0 && src < static_cast<long long>(n)) out[i] = w[static_cast<size_t>(src)];
    }
    return out;
}

// The real CUDA Hammer graph for ONE FTCS step: w_new = r*wLeft + (1-2r)*wCenter + r*wRight,
// built from 3 real Input nodes (the center array plus its two host-shifted
// neighbor arrays) and 2 real Const nodes (r and 1-2r), combined with the
// same Mul/Add ops every chapter since Chapter 4 has had.
static Value buildFtcsStepGraph(Graph& g, Value wLeft, Value wCenter, Value wRight, float r) {
    Value rConst = g.addConst(r, "r");
    Value oneMinus2rConst = g.addConst(1.0f - 2.0f * r, "one_minus_2r");
    Value leftTerm = g.addBinary(OpKind::Mul, wLeft, rConst, "left_term");
    Value centerTerm = g.addBinary(OpKind::Mul, wCenter, oneMinus2rConst, "center_term");
    Value rightTerm = g.addBinary(OpKind::Mul, wRight, rConst, "right_term");
    Value partial = g.addBinary(OpKind::Add, leftTerm, centerTerm, "partial");
    Value wNew = g.addBinary(OpKind::Add, partial, rightTerm, "w_new");
    return wNew;
}

// An independent host-side reference -- a plain nested loop that never
// touches CUDA Hammer's graph at all, the same cross-check pattern every
// im2col-style section in this book has used since Chapter 31.2.
static std::vector<float> directFtcsStep(const std::vector<float>& w, float r) {
    size_t n = w.size();
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        float left = (i > 0) ? w[i - 1] : 0.0f;
        float right = (i + 1 < n) ? w[i + 1] : 0.0f;
        out[i] = r * left + (1.0f - 2.0f * r) * w[i] + r * right;
    }
    return out;
}

int main() {
    printf("=== Chapter 33 (Part 7): Section 33.1 -- a real 1D FTCS finite-difference heat stencil, one step ===\n\n");
    bool allOk = true;
    const int n = 8;
    const float r = 0.25f;

    printf("real update formula and stability condition (John S. Butler, DIT, \"The Explicit Forward\n");
    printf("Time Centered Space (FTCS) Difference Equation for the Heat Equation,\" Numerical Analysis\n");
    printf("book, quoted verbatim):\n\n");
    printf("  \"w_{i,j+1} = r*w_{i-1,j} + (1-2r)*w_{i,j} + r*w_{i+1,j}\"   where   \"r = k/h^2\"\n");
    printf("  stability condition: \"r <= 1/2\"\n\n");
    printf("this section's own r = %.4f, which satisfies r <= 0.5 with margin (%s)\n\n",
           static_cast<double>(r), (r <= 0.5f) ? "confirmed" : "MISMATCH");
    allOk = allOk && (r <= 0.5f);

    std::vector<float> w0 = buildInitialCondition();
    printf("initial condition w0 (a symmetric 2-cell hot spot, N=%d):", n);
    for (float v : w0) printf(" %6.1f", v);
    printf("\n\n");

    printf("--- the real im2col-style neighbor shift (Chapter 31.2's own shiftImage(), now in 1D) ---\n\n");
    std::vector<float> wLeftArr = shift1D(w0, -1);
    std::vector<float> wRightArr = shift1D(w0, +1);
    printf("wLeft  (each point's LEFT  neighbor, zero-padded at the boundary):");
    for (float v : wLeftArr) printf(" %6.1f", v);
    printf("\nwRight (each point's RIGHT neighbor, zero-padded at the boundary):");
    for (float v : wRightArr) printf(" %6.1f", v);
    printf("\n\n");

    Graph g;
    Value wLeft = g.addInput("wLeft");
    Value wCenter = g.addInput("wCenter");
    Value wRight = g.addInput("wRight");
    Value wNew = buildFtcsStepGraph(g, wLeft, wCenter, wRight, r);
    (void)wNew;

    std::map<int, Shape> declared = {
        {wLeft.nodeId, Shape{{n}}}, {wCenter.nodeId, Shape{{n}}}, {wRight.nodeId, Shape{{n}}},
    };
    for (const auto& node : g.nodes())
        if (node->op == OpKind::Const) declared[node->id] = Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    std::map<std::string, std::vector<float>> inputArrays = {
        {"wLeft", wLeftArr}, {"wCenter", w0}, {"wRight", wRightArr},
    };
    auto graphResult = evaluateArrays(g, inputArrays, elementCounts);
    std::vector<float> wNewFromGraph = graphResult.at("w_new");

    std::vector<float> wNewReference = directFtcsStep(w0, r);

    printf("--- the real CUDA Hammer graph's own w_new, vs. an independent direct FTCS reference ---\n\n");
    printf("CUDA Hammer graph : ");
    for (float v : wNewFromGraph) printf(" %7.3f", v);
    printf("\nindependent ref   : ");
    for (float v : wNewReference) printf(" %7.3f", v);
    printf("\n\n");

    bool matchesReference = true;
    for (int i = 0; i < n; ++i) if (std::fabs(wNewFromGraph[static_cast<size_t>(i)] - wNewReference[static_cast<size_t>(i)]) > 1e-4f) matchesReference = false;
    printf("self-check: the graph's own w_new matches the independent direct reference exactly (%s)\n",
           matchesReference ? "confirmed" : "MISMATCH");
    allOk = allOk && matchesReference;

    float sumBefore = 0.0f, sumAfter = 0.0f;
    for (float v : w0) sumBefore += v;
    for (float v : wNewFromGraph) sumAfter += v;
    bool sumConserved = std::fabs(sumBefore - sumAfter) < 1e-3f;
    printf("self-check: total heat is exactly conserved this step (sum before=%.4f, sum after=%.4f) (%s)\n",
           static_cast<double>(sumBefore), static_cast<double>(sumAfter), sumConserved ? "confirmed" : "MISMATCH");
    printf("            -- the disturbance has not yet reached the domain's own boundary cells, so no\n");
    printf("            heat has flowed into the zero-padded ghost region on either side yet.\n\n");
    allOk = allOk && sumConserved;

    bool symmetric = true;
    for (int i = 0; i < n; ++i) if (std::fabs(wNewFromGraph[static_cast<size_t>(i)] - wNewFromGraph[static_cast<size_t>(n - 1 - i)]) > 1e-4f) symmetric = false;
    printf("self-check: w_new is symmetric (w[i] == w[N-1-i]), matching the symmetric initial condition\n");
    printf("and the symmetric stencil and boundary treatment (%s)\n\n", symmetric ? "confirmed" : "MISMATCH");
    allOk = allOk && symmetric;

    printf("--- the real gap this section works around, named directly ---\n\n");
    printf("this stencil needs each output point to read its own left and right neighbor -- CUDA\n");
    printf("Hammer's IR has never had a shift/gather op, and Part 7's own promise rules out adding\n");
    printf("one now. This section needs no new technique at all: it reuses Section 31.2's own\n");
    printf("im2col-style pre-shifted-input pattern unchanged, in 1D instead of 2D. Section 33.2\n");
    printf("continues with the real gap THIS chapter is actually about: running more than one step.\n\n");

    printf("=== Section 33.1 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
