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
