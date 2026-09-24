# Chapter 33: Scientific Computing -- A Real Finite-Difference Heat Stencil

Chapter 31 opened Part 7 with computer vision, finding no shift/gather
primitive and working around it with host-side im2col staging. Chapter
32 moved to NLP, finding no batched/per-row reduction and working
around it by repeating a small graph once per token. Chapter 33 moves
to a third domain, scientific computing, and finds a THIRD kind of gap:
not a missing primitive this time, but a missing control-flow
construct -- CUDA Hammer's IR has never had a loop.

## Chapter 33's own shape

```text
+------------------------------------------------------------------+
|  Chapter 33's own shape, section by section                      |
|                                                                    |
|  33.1  A real 1D FTCS finite-difference heat stencil, one step -- |
|        the update formula and its real stability condition,      |
|        quoted directly from a real numerical-analysis reference,  |
|        built with Chapter 31.2's own im2col-style neighbor-shift  |
|        technique, unchanged, now in 1D instead of 2D. No new      |
|        technique needed at all for this part.                     |
|                                                                    |
|  33.2  5 steps unrolled, a fixed-depth chain -- the real gap this |
|        chapter is actually about. A real solver needs a LOOP;     |
|        this IR has never had one. The honest workaround: a        |
|        FIXED, compile-time-known step count, unrolled directly    |
|        into the host driver as 5 small, sequentially-dependent    |
|        CUDA Hammer graphs. A real, hand-checkable consequence:    |
|        total heat stays exactly conserved until the disturbance   |
|        reaches the boundary, then genuinely leaks.                |
|                                                                    |
|  33.3  The capstone -- 33.1's own single-step graph, unchanged,   |
|        run through Chapter 14/16's real fusion pass (0 Fused-     |
|        Reduction groups this time -- no Sum node at all) and      |
|        Chapter 19's real vectorized-CPU dispatcher, compiled      |
|        with this machine's own real AVX2/FMA or NEON flags and    |
|        RUN once per real timestep across 33.2's own 5-step chain, |
|        mirroring how real production stencil codes actually run   |
|        on real GPUs: one compiled kernel, launched repeatedly.    |
+------------------------------------------------------------------+
```

## 33.1 -- A real 1D FTCS finite-difference heat stencil, one step

The explicit (FTCS -- Forward Time, Centered Space) finite-difference
method is one of the oldest, most standard ways to numerically solve
the 1D heat equation. Its own update formula and its own real stability
condition are quoted directly, verbatim, from a real numerical-analysis
reference (John S. Butler, DIT, "The Explicit Forward Time Centered
Space (FTCS) Difference Equation for the Heat Equation," Numerical
Analysis book):

> "w_{i,j+1} = r*w_{i-1,j} + (1-2r)*w_{i,j} + r*w_{i+1,j}"
> where "r = k/h^2"
> (https://john-s-butler-dit.github.io/NumericalAnalysisBook/Chapter%2008%20-%20Heat%20Equations/801_Heat%20Equation-%20FTCS.html)

and its real stability condition, also quoted directly:

> "r <= 1/2"

Read structurally, this formula needs exactly the same shape Section
31.2 already solved: each output point reads its own left neighbor,
its own right neighbor, and itself. CUDA Hammer's IR has never had a
shift/gather op, and Part 7's own promise rules out adding one now --
but this section does not need to solve that problem again. It reuses
Section 31.2's own im2col-style pre-shifted-input technique, completely
unchanged, now in 1D instead of 2D:

```text
  original 1D array w0          2 shifted copies (im2col-style,
  +--+--+--+--+--+--+--+--+      host-side, zero-padded at the
  |. |. |. |. |. |. |. |. |      domain boundary -- the SAME
  +--+--+--+--+--+--+--+--+      technique as Section 31.2's own
   (8 floats, one real 1D         shiftImage(), just 1D not 2D)
    hot-spot domain)
                                 wLeft  = shift(w0, -1)
                                 wRight = shift(w0, +1)
                                        |          |
                                 3 real CUDA Hammer Input nodes
                                 (wLeft, w0 itself, wRight),
                                 combined by Mul(r)/Mul(1-2r)/Add --
                                 no new OpKind, no shift op
```

This section's own domain is a deliberately small, symmetric,
hand-checkable "hot spot": zero everywhere except two adjacent interior
cells held at 100, `r = 0.25` (well inside the cited `r <= 1/2` bound).
An independent direct nested-loop reference -- real host-side C++ that
never touches CUDA Hammer's graph at all -- checks that the graph's own
output matches exactly, and that total heat (the array's own sum) is
exactly conserved this step, since the disturbance has not yet reached
either boundary cell.


```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 087_a_real_1d_ftcs_finite_difference_heat_stencil_single_step.cpp -o 087_a_real_1d_ftcs_finite_difference_heat_stencil_single_step_driver
./087_a_real_1d_ftcs_finite_difference_heat_stencil_single_step_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 33 (Part 7): Section 33.1 -- a real 1D FTCS finite-difference heat stencil, one step ===

real update formula and stability condition (John S. Butler, DIT, "The Explicit Forward
Time Centered Space (FTCS) Difference Equation for the Heat Equation," Numerical Analysis
book, quoted verbatim):

  "w_{i,j+1} = r*w_{i-1,j} + (1-2r)*w_{i,j} + r*w_{i+1,j}"   where   "r = k/h^2"
  stability condition: "r <= 1/2"

this section's own r = 0.2500, which satisfies r <= 0.5 with margin (confirmed)

initial condition w0 (a symmetric 2-cell hot spot, N=8):    0.0    0.0    0.0  100.0  100.0    0.0    0.0    0.0

--- the real im2col-style neighbor shift (Chapter 31.2's own shiftImage(), now in 1D) ---

wLeft  (each point's LEFT  neighbor, zero-padded at the boundary):    0.0    0.0    0.0    0.0  100.0  100.0    0.0    0.0
wRight (each point's RIGHT neighbor, zero-padded at the boundary):    0.0    0.0  100.0  100.0    0.0    0.0    0.0    0.0

--- the real CUDA Hammer graph's own w_new, vs. an independent direct FTCS reference ---

CUDA Hammer graph :    0.000   0.000  25.000  75.000  75.000  25.000   0.000   0.000
independent ref   :    0.000   0.000  25.000  75.000  75.000  25.000   0.000   0.000

self-check: the graph's own w_new matches the independent direct reference exactly (confirmed)
self-check: total heat is exactly conserved this step (sum before=200.0000, sum after=200.0000) (confirmed)
            -- the disturbance has not yet reached the domain's own boundary cells, so no
            heat has flowed into the zero-padded ghost region on either side yet.

self-check: w_new is symmetric (w[i] == w[N-1-i]), matching the symmetric initial condition
and the symmetric stencil and boundary treatment (confirmed)

--- the real gap this section works around, named directly ---

this stencil needs each output point to read its own left and right neighbor -- CUDA
Hammer's IR has never had a shift/gather op, and Part 7's own promise rules out adding
one now. This section needs no new technique at all: it reuses Section 31.2's own
im2col-style pre-shifted-input pattern unchanged, in 1D instead of 2D. Section 33.2
continues with the real gap THIS chapter is actually about: running more than one step.

=== Section 33.1 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 33 (Part 7): Section 33.1 -- a real 1D FTCS finite-difference heat stencil, one step ===

real update formula and stability condition (John S. Butler, DIT, "The Explicit Forward
Time Centered Space (FTCS) Difference Equation for the Heat Equation," Numerical Analysis
book, quoted verbatim):

  "w_{i,j+1} = r*w_{i-1,j} + (1-2r)*w_{i,j} + r*w_{i+1,j}"   where   "r = k/h^2"
  stability condition: "r <= 1/2"

this section's own r = 0.2500, which satisfies r <= 0.5 with margin (confirmed)

initial condition w0 (a symmetric 2-cell hot spot, N=8):    0.0    0.0    0.0  100.0  100.0    0.0    0.0    0.0

--- the real im2col-style neighbor shift (Chapter 31.2's own shiftImage(), now in 1D) ---

wLeft  (each point's LEFT  neighbor, zero-padded at the boundary):    0.0    0.0    0.0    0.0  100.0  100.0    0.0    0.0
wRight (each point's RIGHT neighbor, zero-padded at the boundary):    0.0    0.0  100.0  100.0    0.0    0.0    0.0    0.0

--- the real CUDA Hammer graph's own w_new, vs. an independent direct FTCS reference ---

CUDA Hammer graph :    0.000   0.000  25.000  75.000  75.000  25.000   0.000   0.000
independent ref   :    0.000   0.000  25.000  75.000  75.000  25.000   0.000   0.000

self-check: the graph's own w_new matches the independent direct reference exactly (confirmed)
self-check: total heat is exactly conserved this step (sum before=200.0000, sum after=200.0000) (confirmed)
            -- the disturbance has not yet reached the domain's own boundary cells, so no
            heat has flowed into the zero-padded ghost region on either side yet.

self-check: w_new is symmetric (w[i] == w[N-1-i]), matching the symmetric initial condition
and the symmetric stencil and boundary treatment (confirmed)

--- the real gap this section works around, named directly ---

this stencil needs each output point to read its own left and right neighbor -- CUDA
Hammer's IR has never had a shift/gather op, and Part 7's own promise rules out adding
one now. This section needs no new technique at all: it reuses Section 31.2's own
im2col-style pre-shifted-input pattern unchanged, in 1D instead of 2D. Section 33.2
continues with the real gap THIS chapter is actually about: running more than one step.

=== Section 33.1 complete: all self-checks confirmed ===
```

*Byte-identical on both machines -- this file has no CUDA/NCCL/MPI linkage and no vector intrinsics, so it is cross-verified on the cloud sandbox and the device, the same standing rule every plain C++ file in this book follows.*


## 33.2 -- 5 steps unrolled, a fixed-depth chain (no loop construct)

Section 33.1 proved ONE real step correct. A real PDE solver needs many
steps -- it repeats the update until some stopping point, which every
real implementation expresses as a loop in the host driver. This is the
real gap this chapter is actually about, and it is a different KIND of
gap than Chapters 31 and 32 found: not a missing primitive that host-
side staging can route around before the graph ever runs, but a
missing CONTROL-FLOW construct. CUDA Hammer's IR has never had a loop:
Chapters 4 through 32 built a pass manager, a fusion engine, four code
generators, and an autotuner, and none of them ever needed one, because
every graph so far has been a fixed, finite DAG known completely at
build time. Adding a loop `OpKind` now, just to make repeated
timestepping convenient, would be exactly the new hidden machinery
Part 7's own opening promise rules out.

The honest workaround: a FIXED, compile-time-known number of steps (5)
unrolled directly into the host-side driver, each one a fresh call to
Section 33.1's own step-building function -- 5 small, independent CUDA
Hammer graphs run in sequence, not one big graph. This is the same
"repeat a small graph in a host-side loop" shape Section 32.1 already
used for its 4 tokens, but with a real structural difference: Section
32.1's 4 token graphs were independent of each other, while here step
t's own real INPUT is step (t-1)'s own real OUTPUT -- a genuine
sequential dependency this IR's own lack of a loop construct forces
onto the host driver instead of the graph itself:

```text
  step 0        step 1         step 2               step 5
  (initial) --> (fresh graph,  (fresh graph,   ...   (fresh graph,
                 host-shifts    host-shifts            host-shifts
                 step 0's own   step 1's own           step 4's own
                 real output)   real output)           real output)
                    |              |                      |
                 CUDA Hammer   CUDA Hammer            CUDA Hammer
                 graph #1      graph #2                graph #5
                    |              |                      |
                 w1 (real)  -> w2 (real)  -> ... -> w5 (real, FINAL)

  5 independent graphs, each one Section 33.1's OWN unchanged structure
  -- the sequential dependency lives entirely in the HOST driver, since
  this IR has no loop construct to express "repeat until step 5" itself
```

Running this chain for real produces a genuine, hand-checkable
physical result: total heat stays EXACTLY conserved through step 3,
while the disturbance is still confined to the domain's interior, then
by step 4 real heat genuinely leaks into the zero-padded boundary --
the same zero-padded approximation Section 31.2 used for its image
border, here producing a real, physically-sensible consequence rather
than an artifact to explain away. Every one of the 5 steps is checked
against an independent host-side reference that never touches CUDA
Hammer's graph at all.

This section is honest about where it stops, too: it can only run a
FIXED, predetermined number of steps decided at build time. It cannot
express a real adaptive "iterate until converged" solver loop, since
deciding to stop early based on a runtime condition needs control flow
this IR has never had -- an honest scope boundary, not a hidden gap,
stated the same direct way Chapters 31 and 32 stated their own.


```cpp
// Chapter 33: Scientific Computing -- A Real Finite-Difference Heat Stencil
// 088_five_steps_unrolled_a_fixed_depth_chain_no_loop_construct.cpp
//
// Section 33.2 -- the real gap this chapter is actually about. A real
// PDE solver does not take one step -- it repeats Section 33.1's own
// update until some stopping point, which in every real implementation
// means a LOOP in the host driver. CUDA Hammer's IR has never had a
// loop or any other control-flow construct: Chapters 4 through 32 built
// a pass manager, a fusion engine, four code generators, and an
// autotuner, and none of them ever needed one, because every graph so
// far was a fixed, finite DAG known completely at build time. Adding a
// loop `OpKind` now, just to make repeated timestepping convenient,
// would be exactly the new hidden machinery Part 7's own opening
// promise rules out.
//
// The honest workaround: a FIXED, compile-time-known number of steps
// (5) unrolled directly into the host-side driver, each one a fresh
// call to Section 33.1's own `buildFtcsStepGraph()` -- five small,
// independent CUDA Hammer graphs run in sequence, not one big graph,
// the same "repeat a small graph in a host-side loop" shape Section
// 32.1 already used for its 4 independent tokens. The real difference
// from Section 32.1: there, the 4 token graphs were independent of each
// other; here, step t's own real INPUT is step (t-1)'s own real OUTPUT
// -- a genuine sequential dependency this IR's own lack of a loop
// construct forces onto the host driver instead of the graph itself.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 088_five_steps_unrolled_a_fixed_depth_chain_no_loop_construct.cpp -o 088_driver
// Run:     ./088_driver
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / Node / Graph (from File 087, unchanged) ====================

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

// ==================== Topological sort / Shape / inferShapes / evaluateArrays (from File 087, unchanged) ====================

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

// ==================== Section 33.2: shift1D() / buildFtcsStepGraph() / directFtcsStep()
//                      (from File 087, unchanged) ====================

static std::vector<float> shift1D(const std::vector<float>& w, int offset) {
    size_t n = w.size();
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        long long src = static_cast<long long>(i) + offset;
        if (src >= 0 && src < static_cast<long long>(n)) out[i] = w[static_cast<size_t>(src)];
    }
    return out;
}
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

// One real CUDA Hammer step, built and evaluated fresh: host-shifts the
// PREVIOUS step's own real output, builds Section 33.1's own graph on
// those shifted arrays, evaluates it, and returns the new real state --
// the unit this section repeats 5 times in a plain host-side C++ loop.
static std::vector<float> runOneHammerStep(const std::vector<float>& wPrev, float r, int n) {
    std::vector<float> wLeftArr = shift1D(wPrev, -1);
    std::vector<float> wRightArr = shift1D(wPrev, +1);
    Graph g;
    Value wLeft = g.addInput("wLeft");
    Value wCenter = g.addInput("wCenter");
    Value wRight = g.addInput("wRight");
    buildFtcsStepGraph(g, wLeft, wCenter, wRight, r);
    std::map<int, Shape> declared = {
        {wLeft.nodeId, Shape{{n}}}, {wCenter.nodeId, Shape{{n}}}, {wRight.nodeId, Shape{{n}}},
    };
    for (const auto& node : g.nodes())
        if (node->op == OpKind::Const) declared[node->id] = Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);
    std::map<std::string, std::vector<float>> inputArrays = {
        {"wLeft", wLeftArr}, {"wCenter", wPrev}, {"wRight", wRightArr},
    };
    auto result = evaluateArrays(g, inputArrays, elementCounts);
    return result.at("w_new");
}

int main() {
    printf("=== Chapter 33 (Part 7): Section 33.2 -- 5 steps unrolled, a fixed-depth chain (no loop construct) ===\n\n");
    bool allOk = true;
    const int n = 8;
    const float r = 0.25f;
    const int numSteps = 5;

    printf("real gap named directly: a real PDE solver repeats the update until some stopping point,\n");
    printf("which every real implementation expresses as a LOOP in the host driver. CUDA Hammer's IR\n");
    printf("has never had a loop or any other control-flow construct (Chapters 4-32 never added one) --\n");
    printf("every graph so far has been a fixed, finite DAG known completely at build time. This\n");
    printf("section's own workaround: a FIXED, compile-time-known step count (%d) unrolled directly\n", numSteps);
    printf("into the HOST driver -- %d small, independent CUDA Hammer graphs run in sequence, not one\n", numSteps);
    printf("big graph, where step t's own real INPUT is step (t-1)'s own real OUTPUT.\n\n");

    std::vector<float> wHammer = {0, 0, 0, 100, 100, 0, 0, 0};
    std::vector<float> wReference = wHammer;
    printf("step 0 (initial): ");
    for (float v : wHammer) printf(" %8.4f", v);
    printf("   sum=%.4f\n", [&]{ float s=0; for (float v : wHammer) s+=v; return s; }());

    bool allStepsMatch = true;
    for (int t = 1; t <= numSteps; ++t) {
        wHammer = runOneHammerStep(wHammer, r, n);
        wReference = directFtcsStep(wReference, r);
        bool stepMatches = true;
        for (int i = 0; i < n; ++i) if (std::fabs(wHammer[static_cast<size_t>(i)] - wReference[static_cast<size_t>(i)]) > 1e-3f) stepMatches = false;
        allStepsMatch = allStepsMatch && stepMatches;
        float sum = 0.0f;
        for (float v : wHammer) sum += v;
        printf("step %d (Hammer): ", t);
        for (float v : wHammer) printf(" %8.4f", v);
        printf("   sum=%.4f  (matches independent reference: %s)\n", static_cast<double>(sum), stepMatches ? "yes" : "NO");
    }
    printf("\nself-check: all %d steps, run through %d fresh CUDA Hammer graphs, match the independent\n", numSteps, numSteps);
    printf("host-side reference exactly at every step (%s)\n\n", allStepsMatch ? "confirmed" : "MISMATCH");
    allOk = allOk && allStepsMatch;

    float sumStep0 = 200.0f;
    float sumStep3, sumStep4;
    {
        std::vector<float> tmp = {0, 0, 0, 100, 100, 0, 0, 0};
        for (int t = 1; t <= 3; ++t) tmp = directFtcsStep(tmp, r);
        sumStep3 = 0.0f; for (float v : tmp) sumStep3 += v;
        tmp = directFtcsStep(tmp, r);
        sumStep4 = 0.0f; for (float v : tmp) sumStep4 += v;
    }
    bool conservedThroughStep3 = std::fabs(sumStep3 - sumStep0) < 1e-2f;
    bool leaksByStep4 = (sumStep0 - sumStep4) > 0.1f;
    printf("--- a real, hand-checkable consequence of this section's own boundary treatment ---\n\n");
    printf("total heat stays EXACTLY conserved (sum=%.4f, matching step 0's sum=%.4f) through step 3,\n",
           static_cast<double>(sumStep3), static_cast<double>(sumStep0));
    printf("while the disturbance is still confined to the domain's interior (%s). By step 4 it has\n",
           conservedThroughStep3 ? "confirmed" : "MISMATCH");
    printf("reached the boundary cells and real heat genuinely leaks into the zero-padded ghost region\n");
    printf("(sum=%.4f, a real loss of %.4f, not a bug) -- the same zero-padded boundary APPROXIMATION\n",
           static_cast<double>(sumStep4), static_cast<double>(sumStep0 - sumStep4));
    printf("Section 31.2 used for its image border, here producing a real, physically-sensible\n");
    printf("consequence (%s) rather than an artifact to explain away.\n\n", leaksByStep4 ? "confirmed" : "MISMATCH");
    allOk = allOk && conservedThroughStep3 && leaksByStep4;

    printf("--- the honest scope boundary this section stops at ---\n\n");
    printf("this section can only run a FIXED, predetermined number of steps decided at build time. It\n");
    printf("cannot express a real adaptive \"iterate until converged\" solver loop, since deciding to\n");
    printf("stop early based on a runtime condition needs control flow this IR has never had. Section\n");
    printf("33.3 continues with this SAME fixed-depth chain, now run through Chapter 14/16's real\n");
    printf("fusion pass and Chapter 19's real vectorized-CPU codegen at every step.\n\n");

    printf("=== Section 33.2 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 088_five_steps_unrolled_a_fixed_depth_chain_no_loop_construct.cpp -o 088_five_steps_unrolled_a_fixed_depth_chain_no_loop_construct_driver
./088_five_steps_unrolled_a_fixed_depth_chain_no_loop_construct_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 33 (Part 7): Section 33.2 -- 5 steps unrolled, a fixed-depth chain (no loop construct) ===

real gap named directly: a real PDE solver repeats the update until some stopping point,
which every real implementation expresses as a LOOP in the host driver. CUDA Hammer's IR
has never had a loop or any other control-flow construct (Chapters 4-32 never added one) --
every graph so far has been a fixed, finite DAG known completely at build time. This
section's own workaround: a FIXED, compile-time-known step count (5) unrolled directly
into the HOST driver -- 5 small, independent CUDA Hammer graphs run in sequence, not one
big graph, where step t's own real INPUT is step (t-1)'s own real OUTPUT.

step 0 (initial):    0.0000   0.0000   0.0000 100.0000 100.0000   0.0000   0.0000   0.0000   sum=200.0000
step 1 (Hammer):    0.0000   0.0000  25.0000  75.0000  75.0000  25.0000   0.0000   0.0000   sum=200.0000  (matches independent reference: yes)
step 2 (Hammer):    0.0000   6.2500  31.2500  62.5000  62.5000  31.2500   6.2500   0.0000   sum=200.0000  (matches independent reference: yes)
step 3 (Hammer):    1.5625  10.9375  32.8125  54.6875  54.6875  32.8125  10.9375   1.5625   sum=200.0000  (matches independent reference: yes)
step 4 (Hammer):    3.5156  14.0625  32.8125  49.2188  49.2188  32.8125  14.0625   3.5156   sum=199.2188  (matches independent reference: yes)
step 5 (Hammer):    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734   sum=197.4609  (matches independent reference: yes)

self-check: all 5 steps, run through 5 fresh CUDA Hammer graphs, match the independent
host-side reference exactly at every step (confirmed)

--- a real, hand-checkable consequence of this section's own boundary treatment ---

total heat stays EXACTLY conserved (sum=200.0000, matching step 0's sum=200.0000) through step 3,
while the disturbance is still confined to the domain's interior (confirmed). By step 4 it has
reached the boundary cells and real heat genuinely leaks into the zero-padded ghost region
(sum=199.2188, a real loss of 0.7812, not a bug) -- the same zero-padded boundary APPROXIMATION
Section 31.2 used for its image border, here producing a real, physically-sensible
consequence (confirmed) rather than an artifact to explain away.

--- the honest scope boundary this section stops at ---

this section can only run a FIXED, predetermined number of steps decided at build time. It
cannot express a real adaptive "iterate until converged" solver loop, since deciding to
stop early based on a runtime condition needs control flow this IR has never had. Section
33.3 continues with this SAME fixed-depth chain, now run through Chapter 14/16's real
fusion pass and Chapter 19's real vectorized-CPU codegen at every step.

=== Section 33.2 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 33 (Part 7): Section 33.2 -- 5 steps unrolled, a fixed-depth chain (no loop construct) ===

real gap named directly: a real PDE solver repeats the update until some stopping point,
which every real implementation expresses as a LOOP in the host driver. CUDA Hammer's IR
has never had a loop or any other control-flow construct (Chapters 4-32 never added one) --
every graph so far has been a fixed, finite DAG known completely at build time. This
section's own workaround: a FIXED, compile-time-known step count (5) unrolled directly
into the HOST driver -- 5 small, independent CUDA Hammer graphs run in sequence, not one
big graph, where step t's own real INPUT is step (t-1)'s own real OUTPUT.

step 0 (initial):    0.0000   0.0000   0.0000 100.0000 100.0000   0.0000   0.0000   0.0000   sum=200.0000
step 1 (Hammer):    0.0000   0.0000  25.0000  75.0000  75.0000  25.0000   0.0000   0.0000   sum=200.0000  (matches independent reference: yes)
step 2 (Hammer):    0.0000   6.2500  31.2500  62.5000  62.5000  31.2500   6.2500   0.0000   sum=200.0000  (matches independent reference: yes)
step 3 (Hammer):    1.5625  10.9375  32.8125  54.6875  54.6875  32.8125  10.9375   1.5625   sum=200.0000  (matches independent reference: yes)
step 4 (Hammer):    3.5156  14.0625  32.8125  49.2188  49.2188  32.8125  14.0625   3.5156   sum=199.2188  (matches independent reference: yes)
step 5 (Hammer):    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734   sum=197.4609  (matches independent reference: yes)

self-check: all 5 steps, run through 5 fresh CUDA Hammer graphs, match the independent
host-side reference exactly at every step (confirmed)

--- a real, hand-checkable consequence of this section's own boundary treatment ---

total heat stays EXACTLY conserved (sum=200.0000, matching step 0's sum=200.0000) through step 3,
while the disturbance is still confined to the domain's interior (confirmed). By step 4 it has
reached the boundary cells and real heat genuinely leaks into the zero-padded ghost region
(sum=199.2188, a real loss of 0.7812, not a bug) -- the same zero-padded boundary APPROXIMATION
Section 31.2 used for its image border, here producing a real, physically-sensible
consequence (confirmed) rather than an artifact to explain away.

--- the honest scope boundary this section stops at ---

this section can only run a FIXED, predetermined number of steps decided at build time. It
cannot express a real adaptive "iterate until converged" solver loop, since deciding to
stop early based on a runtime condition needs control flow this IR has never had. Section
33.3 continues with this SAME fixed-depth chain, now run through Chapter 14/16's real
fusion pass and Chapter 19's real vectorized-CPU codegen at every step.

=== Section 33.2 complete: all self-checks confirmed ===
```

*Byte-identical on both machines at every one of the 5 steps, including the exact point (step 4) where total heat begins to leak into the zero-padded boundary -- plain C++, no vector intrinsics, cross-verified the same way as File 087.*


## 33.3 -- The whole stencil chain, fused and vectorized for real

Section 33.1's own single-step graph -- unchanged, not reimplemented --
is run through Chapter 14/16's real `boundedReductionFusionPass()` and
Chapter 19's real per-node vectorized-CPU dispatcher, exactly Section
31.3's and 32.3's own capstone structure. This graph is different from
either of those in one real, structural way: it has NO `Sum` node at
all -- it is pure elementwise (3 `Mul`, 2 `Add`). The SAME unmodified
pass this book has called "Chapter 14's real reduction fusion pass"
since Chapter 14 produces only `FusedElementwise` groups here, zero
`FusedReduction` groups -- a real, direct demonstration that Chapter
16's own bounding logic genuinely generalized the pass to handle both
shapes, not just reductions, rather than merely happening to work on
graphs that always had a `Sum` in them.

A second real, honestly-named design boundary shows up here too.
Chapter 18's own `generateFullVectorizedProgram()` has always baked its
input arrays in as compile-time literals -- unchanged since it was
built, it was never designed to read runtime data. So running the SAME
fused, vectorized kernel across Section 33.2's own 5 real timesteps
means calling this one unchanged generator 5 separate times, once per
step, each with that step's own real host-shifted data baked in -- not
because the compiled KERNEL's own logic changes step to step (it never
does: identical fused steps, every single step), but because of that
real, stated design choice this generator has had since Chapter 18:

```text
  Section 33.1's own single-step graph (UNCHANGED)
  wLeft, wCenter, wRight -> Mul(r)/Mul(1-2r)/Add/Add -> w_new
                |
        boundedReductionFusionPass()          (Chapter 14/16, unmodified --
        (Chapter 14/16, unmodified)             0 FusedReduction groups: no
                |                                Sum node exists in this graph)
        fused graph (1 FusedElementwise group)
                |
        generateFullVectorizedProgram()   -- called ONCE PER STEP, each
        (Chapter 18/19, unmodified)            call's own literal data
                |                               freshly host-shifted from
        compiled with real AVX2/FMA or          the PREVIOUS step's real
        NEON flags, and RUN                     compiled output
                |
        step 1 -> step 2 -> step 3 -> step 4 -> step 5 (all REAL, compiled,
        executed machine code, chained exactly like Section 33.2's own
        interpreted chain -- same final state, 3 independent ways)
```

This mirrors, in miniature, how real production stencil codes actually
run on real GPUs: one compiled kernel, launched repeatedly from a host
driver, once per timestep -- the loop lives in the host, never in the
kernel itself, which is also exactly why this IR's own lack of a loop
construct was never a barrier to real vectorized execution, only to
expressing the repetition INSIDE a single graph.


```cpp
// Chapter 33: Scientific Computing -- A Real Finite-Difference Heat Stencil
// 089_the_whole_stencil_chain_fused_and_vectorized_for_real.cpp
//
// Section 33.3 -- the capstone. Sections 33.1 and 33.2 proved the real
// per-step graph correct, then chained 5 of them across a real fixed-
// depth, host-orchestrated timestepping loop. This file takes Section
// 33.1's own single-step graph -- unchanged, not reimplemented -- and
// runs it through Chapter 14/16's real boundedReductionFusionPass() and
// Chapter 19's real per-node vectorized-CPU dispatcher, exactly Section
// 31.3's and 32.3's own capstone structure. The real difference this
// time: this graph has NO Sum node at all (it is pure elementwise --
// Mul/Mul/Mul/Add/Add), so the SAME unmodified pass this book has
// called "the real reduction fusion pass" since Chapter 14 produces
// only FusedElementwise groups here, zero FusedReduction groups -- a
// real, direct demonstration that Chapter 16's own bounding logic
// generalized the pass to handle both shapes, not just reductions.
//
// A second real, honestly-named boundary: Chapter 18's own
// generateFullVectorizedProgram() has always baked its input arrays in
// as compile-time literals (unchanged since Chapter 18) -- it was never
// built to read runtime data. So running the SAME fused, vectorized
// kernel across Section 33.2's own 5 real timesteps means calling this
// unchanged generator 5 separate times, once per step, each with that
// step's own real host-shifted data baked in -- not because the
// KERNEL's own logic changes step to step (it never does: identical
// fused steps every time), but because of that real, stated design
// choice this generator has had since it was built. This mirrors, in
// miniature, how real production stencil codes actually run on real
// GPUs: one compiled kernel, launched repeatedly from a host driver,
// once per timestep.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 [-mavx2 -mfma on x86_64] 089_the_whole_stencil_chain_fused_and_vectorized_for_real.cpp -o 089_driver
// Run:     ./089_driver
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

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14 / File 086, unchanged) ====================

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

// ==================== boundedReductionFusionPass() (from Chapter 16 / Section 17.3 / File 086, unchanged) ====================

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

// ==================== evaluateArrays() (from Section 17.1 / File 086, unchanged) ====================

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

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2 / File 086, unchanged) ====================

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

// ==================== emitSteps() / generateCpuElementwiseFunction() (from Section 18.1 / File 086, unchanged) ====================

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
//                      (from Sections 19.1-19.3 / File 086, unchanged) ====================

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

// ==================== Section 33.3: shift1D() (from Files 087/088, unchanged) ====================

static std::vector<float> shift1D(const std::vector<float>& w, int offset) {
    size_t n = w.size();
    std::vector<float> out(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        long long src = static_cast<long long>(i) + offset;
        if (src >= 0 && src < static_cast<long long>(n)) out[i] = w[static_cast<size_t>(src)];
    }
    return out;
}

int main() {
    printf("=== Chapter 33 (Part 7): Section 33.3 -- the whole stencil chain, fused and vectorized for real ===\n\n");
    bool allOk = true;
    const int n = 8;
    const float r = 0.25f;
    const int numSteps = 5;

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
    // Stage 0: Section 33.1's own single-step graph, rebuilt exactly.
    // ---------------------------------------------------------------
    printf("--- Stage 0: rebuilding Section 33.1's own single-step graph exactly ---\n\n");
    Graph g;
    Value wLeft = g.addInput("wLeft");
    Value wCenter = g.addInput("wCenter");
    Value wRight = g.addInput("wRight");
    Value rConst = g.addConst(r, "r");
    Value oneMinus2rConst = g.addConst(1.0f - 2.0f * r, "one_minus_2r");
    Value leftTerm = g.addBinary(OpKind::Mul, wLeft, rConst, "left_term");
    Value centerTerm = g.addBinary(OpKind::Mul, wCenter, oneMinus2rConst, "center_term");
    Value rightTerm = g.addBinary(OpKind::Mul, wRight, rConst, "right_term");
    Value partial = g.addBinary(OpKind::Add, leftTerm, centerTerm, "partial");
    Value wNew = g.addBinary(OpKind::Add, partial, rightTerm, "w_new");
    (void)wNew;

    std::map<int, Shape> declared = {
        {wLeft.nodeId, Shape{{n}}}, {wCenter.nodeId, Shape{{n}}}, {wRight.nodeId, Shape{{n}}},
    };
    for (const auto& node : g.nodes())
        if (node->op == OpKind::Const) declared[node->id] = Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    long long origInputOrConstCount = 0;
    for (const auto& node : g.nodes()) if (node->op == OpKind::Input || node->op == OpKind::Const) origInputOrConstCount++;
    long long origComputeNodeCount = static_cast<long long>(g.size()) - origInputOrConstCount;
    printf("single-step graph: %zu nodes total, %lld of them real compute nodes (3 Mul, 2 Add) -- NO\n", g.size(), origComputeNodeCount);
    printf("Sum node at all, unlike Chapter 31.3's and 32.3's own capstone graphs.\n\n");

    // ---------------------------------------------------------------
    // Stage 1: Chapter 14/16's real fusion pass, unmodified.
    // ---------------------------------------------------------------
    printf("--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---\n\n");
    FusionResult fr = boundedReductionFusionPass(g, elementCounts, 3);
    const Graph& fused = fr.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& node : fused.nodes()) {
        int oldId = fr.representativeOldId.at(node->id);
        fusedShapes[node->id] = shapes.at(oldId);
        fusedElementCounts[node->id] = elementCounts.at(oldId);
    }
    int fusedElementwiseCount = 0, fusedReductionCount = 0, plainCount = 0;
    for (const auto& node : fused.nodes()) {
        if (node->op == OpKind::FusedElementwise) fusedElementwiseCount++;
        else if (node->op == OpKind::FusedReduction) fusedReductionCount++;
        else if (node->op != OpKind::Input && node->op != OpKind::Const) plainCount++;
    }
    printf("fused graph: %zu nodes total -- %d FusedElementwise group(s), %d FusedReduction group(s),\n",
           fused.size(), fusedElementwiseCount, fusedReductionCount);
    printf("%d plain (unfused) compute node(s), down from %lld compute nodes in the original graph.\n", plainCount, origComputeNodeCount);
    bool zeroReductionGroups = (fusedReductionCount == 0);
    printf("self-check: 0 FusedReduction groups, since this graph has no Sum node at all -- the SAME\n");
    printf("unmodified pass this book has called Chapter 14's reduction fusion pass since Chapter 14\n");
    printf("genuinely generalizes to a pure-elementwise graph (%s)\n\n", zeroReductionGroups ? "confirmed" : "MISMATCH");
    allOk = allOk && zeroReductionGroups;

    // ---------------------------------------------------------------
    // Stage 2: run the SAME fused/vectorized kernel 5 times, once per
    // real timestep, host-shifting between calls -- Section 33.2's own
    // fixed-depth chain, now compiled and RUN instead of interpreted.
    // ---------------------------------------------------------------
    printf("--- Stage 2: Chapter 19's real vectorized dispatcher, called once per real timestep on %s ---\n\n", isaName(hostIsa).c_str());
    std::vector<float> wInterpreted = {0, 0, 0, 100, 100, 0, 0, 0};
    std::vector<float> wFusedInterpreted = wInterpreted;
    std::vector<float> wVectorized = wInterpreted;
    bool allStepsMatch = true;
    bool noScalarFallbackAnywhere = true;
    int totalDispatchedNodes = 0, totalFoldsAllSteps = 0;

    for (int t = 1; t <= numSteps; ++t) {
        // interpreted, original (unfused) graph -- re-derives Section 33.2's own chain
        std::vector<float> lArr = shift1D(wInterpreted, -1), rArr = shift1D(wInterpreted, +1);
        auto interpResult = evaluateArrays(g, {{"wLeft", lArr}, {"wCenter", wInterpreted}, {"wRight", rArr}}, elementCounts);
        wInterpreted = interpResult.at("w_new");

        // interpreted, FUSED graph -- same fusion-preserves-the-answer check as 31.3/32.3
        std::vector<float> lArrF = shift1D(wFusedInterpreted, -1), rArrF = shift1D(wFusedInterpreted, +1);
        auto fusedInterpResult = evaluateArrays(fused, {{"wLeft", lArrF}, {"wCenter", wFusedInterpreted}, {"wRight", rArrF}}, fusedElementCounts);
        wFusedInterpreted = fusedInterpResult.at("w_new");

        // real vectorized, compiled, RUN -- generateFullVectorizedProgram() called fresh
        // this step, this step's own real shifted data baked in as literals (unchanged
        // design since Chapter 18: this generator has never read runtime input)
        std::vector<float> lArrV = shift1D(wVectorized, -1), rArrV = shift1D(wVectorized, +1);
        std::map<std::string, std::vector<float>> stepInputs = {{"wLeft", lArrV}, {"wCenter", wVectorized}, {"wRight", rArrV}};
        std::vector<std::pair<std::string, Backend>> dispatchLog;
        int totalFolds = 0;
        std::string prog = generateFullVectorizedProgram(fused, fusedShapes, fusedElementCounts, stepInputs, hostIsa, dispatchLog, totalFolds);
        std::string stem = "/tmp/hammer_ch33_089_step" + std::to_string(t);
        writeFile(stem + ".cpp", prog);
        std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags + " " + stem + ".cpp -o " + stem + " 2>&1");
        bool compileClean = compileLog.empty();
        if (!compileClean) { printf("step %d compile FAILED:\n%s\n", t, compileLog.c_str()); allOk = false; }
        std::string runOutput = runShellCaptureAll(stem);
        auto generatedArrays = parseNamedBuffers(runOutput);
        wVectorized = generatedArrays.at("w_new");
        totalDispatchedNodes += static_cast<int>(dispatchLog.size());
        totalFoldsAllSteps += totalFolds;
        for (const auto& entry : dispatchLog) if (entry.second == Backend::ScalarFallback) noScalarFallbackAnywhere = false;
        if (t == 1) {
            printf("per-node dispatch, identical every step since it is the SAME fused graph structure\n");
            printf("each time (only the literal data baked into main() changes step to step):\n");
            for (const auto& entry : dispatchLog) printf("  %-12s -> %s\n", entry.first.c_str(), backendName(entry.second));
            printf("\n");
        }

        bool stepMatches = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(wInterpreted[static_cast<size_t>(i)] - wFusedInterpreted[static_cast<size_t>(i)]) > 1e-3f) stepMatches = false;
            if (std::fabs(wInterpreted[static_cast<size_t>(i)] - wVectorized[static_cast<size_t>(i)]) > 1e-2f) stepMatches = false;
        }
        allStepsMatch = allStepsMatch && stepMatches;
        printf("step %d: interpreted(original) vs interpreted(fused) vs real-vectorized(%s) -- %s\n",
               t, isaName(hostIsa).c_str(), stepMatches ? "all 3 agree" : "MISMATCH");
    }

    printf("\nfinal state after %d real timesteps, computed 3 independent ways:\n", numSteps);
    printf("interpreted(original) : "); for (float v : wInterpreted) printf(" %8.4f", v); printf("\n");
    printf("interpreted(fused)    : "); for (float v : wFusedInterpreted) printf(" %8.4f", v); printf("\n");
    printf("real-vectorized(%-6s): ", isaName(hostIsa).c_str()); for (float v : wVectorized) printf(" %8.4f", v); printf("\n\n");

    printf("self-check: all %d steps agree across all 3 computations -- the fusion pass and the real\n", numSteps);
    printf("vectorized dispatcher both preserve the original graph's own answer exactly, chained\n");
    printf("across %d real, sequentially-dependent timesteps (%s)\n", numSteps, allStepsMatch ? "confirmed" : "MISMATCH");
    printf("self-check: 0 of %d total dispatched nodes fell back to the scalar path across all %d steps\n",
           totalDispatchedNodes, numSteps);
    printf("(%s), %d total FMA fold(s) applied\n\n", noScalarFallbackAnywhere ? "confirmed" : "MISMATCH", totalFoldsAllSteps);
    allOk = allOk && allStepsMatch && noScalarFallbackAnywhere;

    printf("--- what this capstone actually shows ---\n\n");
    printf("Chapter 18's own generateFullVectorizedProgram() bakes its inputs in as compile-time\n");
    printf("literals, unchanged since it was built -- it was never designed to read runtime data. So\n");
    printf("running the SAME fused, vectorized kernel across %d real timesteps here means calling this\n", numSteps);
    printf("one unchanged generator %d separate times, once per step, each with that step's own real\n", numSteps);
    printf("host-shifted data baked in -- not because the compiled kernel's own logic changes (it never\n");
    printf("does: identical fused steps, every single step), but because of that real, stated design\n");
    printf("choice. This mirrors, in miniature, how real production stencil codes actually run on real\n");
    printf("GPUs: one compiled kernel, launched repeatedly from a host driver, once per timestep.\n\n");

    printf("=== Section 33.3 complete, Chapter 33 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 [-mavx2 -mfma added automatically on x86_64] 089_the_whole_stencil_chain_fused_and_vectorized_for_real.cpp -o 089_the_whole_stencil_chain_fused_and_vectorized_for_real_driver
./089_the_whole_stencil_chain_fused_and_vectorized_for_real_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 33 (Part 7): Section 33.3 -- the whole stencil chain, fused and vectorized for real ===

--- Stage 0: rebuilding Section 33.1's own single-step graph exactly ---

single-step graph: 10 nodes total, 5 of them real compute nodes (3 Mul, 2 Add) -- NO
Sum node at all, unlike Chapter 31.3's and 32.3's own capstone graphs.

--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---

fused graph: 6 nodes total -- 1 FusedElementwise group(s), 0 FusedReduction group(s),
0 plain (unfused) compute node(s), down from 5 compute nodes in the original graph.
self-check: 0 FusedReduction groups, since this graph has no Sum node at all -- the SAME
unmodified pass this book has called Chapter 14's reduction fusion pass since Chapter 14
genuinely generalizes to a pure-elementwise graph (confirmed)

--- Stage 2: Chapter 19's real vectorized dispatcher, called once per real timestep on AVX2/FMA ---

per-node dispatch, identical every step since it is the SAME fused graph structure
each time (only the literal data baked into main() changes step to step):
  w_new        -> VECTOR+FMA

step 1: interpreted(original) vs interpreted(fused) vs real-vectorized(AVX2/FMA) -- all 3 agree
step 2: interpreted(original) vs interpreted(fused) vs real-vectorized(AVX2/FMA) -- all 3 agree
step 3: interpreted(original) vs interpreted(fused) vs real-vectorized(AVX2/FMA) -- all 3 agree
step 4: interpreted(original) vs interpreted(fused) vs real-vectorized(AVX2/FMA) -- all 3 agree
step 5: interpreted(original) vs interpreted(fused) vs real-vectorized(AVX2/FMA) -- all 3 agree

final state after 5 real timesteps, computed 3 independent ways:
interpreted(original) :    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734
interpreted(fused)    :    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734
real-vectorized(AVX2/FMA):    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734

self-check: all 5 steps agree across all 3 computations -- the fusion pass and the real
vectorized dispatcher both preserve the original graph's own answer exactly, chained
across 5 real, sequentially-dependent timesteps (confirmed)
self-check: 0 of 5 total dispatched nodes fell back to the scalar path across all 5 steps
(confirmed), 10 total FMA fold(s) applied

--- what this capstone actually shows ---

Chapter 18's own generateFullVectorizedProgram() bakes its inputs in as compile-time
literals, unchanged since it was built -- it was never designed to read runtime data. So
running the SAME fused, vectorized kernel across 5 real timesteps here means calling this
one unchanged generator 5 separate times, once per step, each with that step's own real
host-shifted data baked in -- not because the compiled kernel's own logic changes (it never
does: identical fused steps, every single step), but because of that real, stated design
choice. This mirrors, in miniature, how real production stencil codes actually run on real
GPUs: one compiled kernel, launched repeatedly from a host driver, once per timestep.

=== Section 33.3 complete, Chapter 33 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 33 (Part 7): Section 33.3 -- the whole stencil chain, fused and vectorized for real ===

--- Stage 0: rebuilding Section 33.1's own single-step graph exactly ---

single-step graph: 10 nodes total, 5 of them real compute nodes (3 Mul, 2 Add) -- NO
Sum node at all, unlike Chapter 31.3's and 32.3's own capstone graphs.

--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---

fused graph: 6 nodes total -- 1 FusedElementwise group(s), 0 FusedReduction group(s),
0 plain (unfused) compute node(s), down from 5 compute nodes in the original graph.
self-check: 0 FusedReduction groups, since this graph has no Sum node at all -- the SAME
unmodified pass this book has called Chapter 14's reduction fusion pass since Chapter 14
genuinely generalizes to a pure-elementwise graph (confirmed)

--- Stage 2: Chapter 19's real vectorized dispatcher, called once per real timestep on NEON ---

per-node dispatch, identical every step since it is the SAME fused graph structure
each time (only the literal data baked into main() changes step to step):
  w_new        -> VECTOR+FMA

step 1: interpreted(original) vs interpreted(fused) vs real-vectorized(NEON) -- all 3 agree
step 2: interpreted(original) vs interpreted(fused) vs real-vectorized(NEON) -- all 3 agree
step 3: interpreted(original) vs interpreted(fused) vs real-vectorized(NEON) -- all 3 agree
step 4: interpreted(original) vs interpreted(fused) vs real-vectorized(NEON) -- all 3 agree
step 5: interpreted(original) vs interpreted(fused) vs real-vectorized(NEON) -- all 3 agree

final state after 5 real timesteps, computed 3 independent ways:
interpreted(original) :    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734
interpreted(fused)    :    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734
real-vectorized(NEON  ):    5.2734  16.1133  32.2266  45.1172  45.1172  32.2266  16.1133   5.2734

self-check: all 5 steps agree across all 3 computations -- the fusion pass and the real
vectorized dispatcher both preserve the original graph's own answer exactly, chained
across 5 real, sequentially-dependent timesteps (confirmed)
self-check: 0 of 5 total dispatched nodes fell back to the scalar path across all 5 steps
(confirmed), 10 total FMA fold(s) applied

--- what this capstone actually shows ---

Chapter 18's own generateFullVectorizedProgram() bakes its inputs in as compile-time
literals, unchanged since it was built -- it was never designed to read runtime data. So
running the SAME fused, vectorized kernel across 5 real timesteps here means calling this
one unchanged generator 5 separate times, once per step, each with that step's own real
host-shifted data baked in -- not because the compiled kernel's own logic changes (it never
does: identical fused steps, every single step), but because of that real, stated design
choice. This mirrors, in miniature, how real production stencil codes actually run on real
GPUs: one compiled kernel, launched repeatedly from a host driver, once per timestep.

=== Section 33.3 complete, Chapter 33 complete: all self-checks confirmed ===
```

*Identical apart from the ISA name itself (AVX2/FMA on the cloud sandbox, NEON on the device, each machine's own real vector hardware) -- the final state after all 5 real timesteps agrees exactly across all three computations (interpreted-original, interpreted-fused, real-vectorized-compiled) on both machines, matching File 088's own already-verified 5-step reference exactly.*


## What Chapter 33 actually shows

Nothing here is a simulation. Section 33.1's single step is a real
CUDA Hammer graph's real output, built from real host-shifted neighbor
arrays exactly the way Section 31.2's im2col technique already
established, checked against an independent direct finite-difference
reference and against a real, verbatim-quoted formula and stability
condition from a real numerical-analysis source. Section 33.2's 5-step
chain is 5 real, independently built and evaluated CUDA Hammer graphs,
each one's real input taken directly from the previous one's real
output -- and the exact step (step 4) where total heat begins to leak
into the zero-padded boundary was computed for real, not asserted, and
matches a plain hand derivation of the same stencil. Section 33.3's
final state after 5 real timesteps is the same real numbers three
separate times: from the original graph's interpreter, from the SAME
graph after Chapter 14/16's real fusion pass rewrote it, and from real
AVX2/FMA or NEON machine code Chapter 19's real codegen generated,
compiled, and ran -- called fresh once per timestep, the same way a
real production stencil kernel is launched repeatedly from a host
driver.

Chapter 31 found a missing primitive (no shift/gather) and routed
around it with host-side staging. Chapter 32 found a missing reduction
mode (no batched/per-row `Sum`) and routed around it with a host-side
loop repeating a small graph. Chapter 33 found a missing CONTROL-FLOW
construct (no loop at all) and routed around it the same honest way:
not with a new `OpKind`, but with the host unrolling a fixed,
predetermined number of steps, each one Section 33.1's own unchanged
graph. All three chapters also named a real boundary this IR has never
crossed -- no shift op, no batched reduction, no loop -- and said so
directly rather than quietly working around it with something that
would have broken Part 7's own opening promise. No new `OpKind`, no
new pass, no new backend, kept for a third domain in a third different
way.

Chapter 34 moves to a fourth and final domain. Part 7's own TOC entry
records the remaining candidate -- quantitative finance -- still to be
built.
