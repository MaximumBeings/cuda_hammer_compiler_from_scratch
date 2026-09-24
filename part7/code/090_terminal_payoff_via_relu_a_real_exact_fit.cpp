// Chapter 34: Quantitative Finance -- A Real Binomial Option Pricer
// 090_terminal_payoff_via_relu_a_real_exact_fit.cpp
//
// Section 34.1 -- Part 7's fourth and final domain, and the first
// section in this whole Part that needs NO workaround at all for the
// piece it builds. A European option's own payoff at maturity is,
// by definition, `max(S_T - K, 0)` for a call and `max(K - S_T, 0)`
// for a put -- and `max(x, 0)` is exactly what `ReLU` has computed
// since Chapter 4. This is not a coincidence to route around; it is a
// real, exact fit between an op this IR built for neural-net
// activations and a real financial payoff function.
//
// The Cox-Ross-Rubinstein (CRR) binomial tree (Cox, Ross, and
// Rubinstein, 1979, "Option Pricing: A Simplified Approach," Journal
// of Financial Economics) needs one real input this IR has never had:
// its own up/down factors and risk-neutral probability are built from
// `exp` and `sqrt` (quoted verbatim from AnalystPrep's FRM study notes,
// https://analystprep.com/study-notes/frm/part-1/valuation-and-risk-management/binomial-trees/):
//   "u=size of the up move factor=e^(sigma*sqrt(t)), and
//    d=size of the down move factor=e^(-sigma*sqrt(t))=1/(e^(sigma*sqrt(t)))=1/u"
//   "pi=probability of an up move=(e^(r*t)-d)/(u-d)"
// This IR has never had `exp` or `sqrt` (the same real gap Section
// 32.1 already named for LayerNorm's variance/sqrt/divide half). The
// honest workaround, unchanged from Section 31.1's own pattern: compute
// u, d, and the risk-neutral probability on the HOST with plain
// `std::exp`/`std::sqrt`, then feed the real results into the graph as
// real `Const` values.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 090_terminal_payoff_via_relu_a_real_exact_fit.cpp -o 090_driver
// Run:     ./090_driver
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

// ==================== Section 34.1: real CRR tree parameters (host-side, exp/sqrt --
//                      this IR has neither) and the real terminal payoff graphs ====================

struct CrrParams { float u, d, p, disc; };
// Host-side only -- exp/sqrt are NOT CUDA Hammer graph ops. This IR has never had
// either (the same real gap Section 32.1 already named for LayerNorm's own
// variance/sqrt/divide half), so these are computed here, in plain C++, and fed
// into the graph as real Const values -- unchanged from Section 31.1's own
// "measure/compute host-side, bake in as Const" pattern.
static CrrParams computeCrrParams(float sigma, float r, float dt) {
    float u = std::exp(sigma * std::sqrt(dt));
    float d = 1.0f / u;
    float p = (std::exp(r * dt) - d) / (u - d);
    float disc = std::exp(-r * dt);
    return CrrParams{u, d, p, disc};
}

// Terminal stock prices S_T[j] = S0 * u^j * d^(N-j), j=0..N -- real host-side
// arithmetic (powers of two real host-computed constants), not a graph op.
static std::vector<float> buildTerminalPrices(float s0, float u, float d, int n) {
    std::vector<float> prices(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) prices[static_cast<size_t>(j)] = s0 * std::pow(u, j) * std::pow(d, n - j);
    return prices;
}

int main() {
    printf("=== Chapter 34 (Part 7): Section 34.1 -- terminal payoff via ReLU, a real exact fit ===\n\n");
    bool allOk = true;

    const float s0 = 100.0f, k = 100.0f, r = 0.05f, sigma = 0.2f, T = 1.0f;
    const int n = 4;
    const float dt = T / static_cast<float>(n);

    printf("real CRR (Cox, Ross, Rubinstein 1979) tree parameters, quoted verbatim (AnalystPrep FRM\n");
    printf("study notes):\n\n");
    printf("  \"u=size of the up move factor=e^(sigma*sqrt(t)), and\n");
    printf("   d=size of the down move factor=e^(-sigma*sqrt(t))=1/(e^(sigma*sqrt(t)))=1/u\"\n");
    printf("  \"pi=probability of an up move=(e^(r*t)-d)/(u-d)\"\n\n");
    printf("this IR has no exp and no sqrt (the same real gap Section 32.1 already named for\n");
    printf("LayerNorm's own variance/sqrt/divide half), so u, d, and the risk-neutral probability are\n");
    printf("computed HOST-SIDE with plain std::exp/std::sqrt, then fed into the graph as real Const\n");
    printf("values -- unchanged from Section 31.1's own \"measure/compute host-side, bake in as Const\"\n");
    printf("pattern.\n\n");

    CrrParams crr = computeCrrParams(sigma, r, dt);
    printf("S0=%.1f K=%.1f r=%.2f sigma=%.2f T=%.1f N=%d steps, dt=%.4f\n", static_cast<double>(s0),
           static_cast<double>(k), static_cast<double>(r), static_cast<double>(sigma), static_cast<double>(T), n, static_cast<double>(dt));
    printf("real computed: u=%.6f d=%.6f p=%.6f disc=%.6f (1-p=%.6f)\n\n",
           static_cast<double>(crr.u), static_cast<double>(crr.d), static_cast<double>(crr.p),
           static_cast<double>(crr.disc), static_cast<double>(1.0f - crr.p));
    bool pInUnitInterval = (crr.p > 0.0f && crr.p < 1.0f);
    printf("self-check: the real risk-neutral probability p is strictly between 0 and 1, a required\n");
    printf("no-arbitrage condition for this real CRR tree (%s)\n\n", pInUnitInterval ? "confirmed" : "MISMATCH");
    allOk = allOk && pInUnitInterval;

    std::vector<float> terminalPrices = buildTerminalPrices(s0, crr.u, crr.d, n);
    printf("real terminal stock prices S_T[j] = S0 * u^j * d^(N-j), j=0..%d:\n", n);
    for (float v : terminalPrices) printf(" %9.4f", v);
    printf("\n\n");

    // Real CUDA Hammer graph: PUT payoff = ReLU(K - S_T) = ReLU(-S_T + K).
    Graph putGraph;
    Value putST = putGraph.addInput("S_T");
    Value putNegOne = putGraph.addConst(-1.0f, "neg_one");
    Value putK = putGraph.addConst(k, "K");
    Value putNegS = putGraph.addBinary(OpKind::Mul, putST, putNegOne, "neg_S_T");
    Value putDiff = putGraph.addBinary(OpKind::Add, putNegS, putK, "K_minus_S_T");
    Value putPayoff = putGraph.addUnary(OpKind::ReLU, putDiff, "put_payoff");
    (void)putPayoff;

    // Real CUDA Hammer graph: CALL payoff = ReLU(S_T - K) = ReLU(S_T + (-K)).
    Graph callGraph;
    Value callST = callGraph.addInput("S_T");
    Value callNegK = callGraph.addConst(-k, "neg_K");
    Value callDiff = callGraph.addBinary(OpKind::Add, callST, callNegK, "S_T_minus_K");
    Value callPayoff = callGraph.addUnary(OpKind::ReLU, callDiff, "call_payoff");
    (void)callPayoff;

    std::map<int, Shape> putDeclared = {{putST.nodeId, Shape{{n + 1}}}};
    for (const auto& node : putGraph.nodes()) if (node->op == OpKind::Const) putDeclared[node->id] = Shape{};
    std::map<int, Shape> putShapes = inferShapes(putGraph, putDeclared);
    std::map<int, long long> putCounts;
    for (const auto& kv : putShapes) putCounts[kv.first] = numElements(kv.second);
    auto putResult = evaluateArrays(putGraph, {{"S_T", terminalPrices}}, putCounts);
    std::vector<float> putPayoffs = putResult.at("put_payoff");

    std::map<int, Shape> callDeclared = {{callST.nodeId, Shape{{n + 1}}}};
    for (const auto& node : callGraph.nodes()) if (node->op == OpKind::Const) callDeclared[node->id] = Shape{};
    std::map<int, Shape> callShapes = inferShapes(callGraph, callDeclared);
    std::map<int, long long> callCounts;
    for (const auto& kv : callShapes) callCounts[kv.first] = numElements(kv.second);
    auto callResult = evaluateArrays(callGraph, {{"S_T", terminalPrices}}, callCounts);
    std::vector<float> callPayoffs = callResult.at("call_payoff");

    printf("real CUDA Hammer graph, PUT payoff  = ReLU(K - S_T)  :");
    for (float v : putPayoffs) printf(" %9.4f", v);
    printf("\nreal CUDA Hammer graph, CALL payoff = ReLU(S_T - K)  :");
    for (float v : callPayoffs) printf(" %9.4f", v);
    printf("\n\n");

    bool putMatchesRef = true, callMatchesRef = true;
    for (int j = 0; j <= n; ++j) {
        float s = terminalPrices[static_cast<size_t>(j)];
        float refPut = std::max(k - s, 0.0f);
        float refCall = std::max(s - k, 0.0f);
        if (std::fabs(putPayoffs[static_cast<size_t>(j)] - refPut) > 1e-3f) putMatchesRef = false;
        if (std::fabs(callPayoffs[static_cast<size_t>(j)] - refCall) > 1e-3f) callMatchesRef = false;
    }
    printf("self-check: the graph's own PUT payoff matches an independent std::max(K-S,0) reference\n");
    printf("exactly at every node (%s)\n", putMatchesRef ? "confirmed" : "MISMATCH");
    printf("self-check: the graph's own CALL payoff matches an independent std::max(S-K,0) reference\n");
    printf("exactly at every node (%s)\n\n", callMatchesRef ? "confirmed" : "MISMATCH");
    allOk = allOk && putMatchesRef && callMatchesRef;

    printf("--- what this section actually shows ---\n\n");
    printf("no workaround was needed for this part at all. `max(x, 0)` IS `ReLU(x)`, exactly -- an op\n");
    printf("this book built in Chapter 4 for neural-net activations turns out to be a real, exact fit\n");
    printf("for a real option's own payoff function, with nothing approximated and nothing left out.\n");
    printf("Section 34.2 continues with the harder part: the tree's own backward induction.\n\n");

    printf("=== Section 34.1 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
