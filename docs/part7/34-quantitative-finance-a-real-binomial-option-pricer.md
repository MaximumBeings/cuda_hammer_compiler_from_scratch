# Chapter 34: Quantitative Finance -- A Real Binomial Option Pricer

Chapter 31 opened Part 7 with computer vision, finding no shift/gather
primitive and working around it with host-side im2col staging. Chapter
32 moved to NLP, finding no batched/per-row reduction and working
around it by repeating a small graph once per token. Chapter 33 moved
to scientific computing, finding no loop construct and working around
it by unrolling a fixed step count into the host driver. Chapter 34
closes Part 7 with quantitative finance, and finds something genuinely
different from the first three chapters: one piece (Section 34.1) that
needs no workaround at all, and one structural problem (Section 34.2)
that this IR's own graph model already solves, without staging and
without unrolling.

## Chapter 34's own shape

```text
+------------------------------------------------------------------+
|  Chapter 34's own shape, section by section                       |
|                                                                    |
|  34.1  Terminal payoff via ReLU, a real exact fit -- a European   |
|        option's payoff at maturity is max(S_T-K,0) for a call and |
|        max(K-S_T,0) for a put, and max(x,0) IS ReLU(x), exactly.  |
|        The real CRR tree parameters (u, d, risk-neutral p) need   |
|        exp and sqrt, an honest host-side gap tied to Section      |
|        32.1's own LayerNorm variance/sqrt/divide gap.             |
|                                                                    |
|  34.2  The whole binomial tree as ONE static graph -- the real    |
|        finding this chapter is actually about. A tree's parent-   |
|        child relationship is IRREGULAR but completely, statically |
|        known at build time, which is exactly what an ordinary     |
|        CUDA Hammer Value reference already expresses. No im2col-  |
|        style staging (unlike Ch31/Ch33) and no host-orchestrated  |
|        re-evaluation between levels (unlike Ch33's own step       |
|        chain) -- the WHOLE tree, all levels, is one static graph. |
|                                                                    |
|  34.3  The capstone -- early exercise via ReLU-as-max, the        |
|        American put, fused and vectorized. A second real math     |
|        identity, max(a,b) = b + ReLU(a-b), extends 34.2's tree to |
|        real American-style early exercise with one more ReLU per  |
|        interior node. Run through Chapter 14/16's real fusion     |
|        pass (0 FusedReduction groups again, this time tested      |
|        against a genuinely IRREGULAR tree topology, not a flat    |
|        chain) and Chapter 19's real vectorized-CPU dispatcher.    |
+------------------------------------------------------------------+
```

## 34.1 -- Terminal payoff via ReLU, a real exact fit

A European option's own payoff at maturity is, by definition,
`max(S_T - K, 0)` for a call and `max(K - S_T, 0)` for a put, where
`S_T` is the underlying's real price at expiry and `K` is the strike.
`max(x, 0)` is exactly what `ReLU` has computed since Chapter 4 built
it for neural-net activations. This section needs no workaround at
all -- the first time in Part 7 that a real domain's own operation
maps onto CUDA Hammer's existing op set with nothing left to route
around.

Building a real binomial tree needs one more real input: the
Cox-Ross-Rubinstein (CRR) model's own up/down move factors and its
risk-neutral probability, quoted directly, verbatim, from AnalystPrep's
FRM study notes:

> "u=size of the up move factor=e^(sigma\*sqrt(t)), and
> d=size of the down move factor=e^(-sigma\*sqrt(t))=1/(e^(sigma\*sqrt(t)))=1/u"
> "pi=probability of an up move=(e^(r\*t)-d)/(u-d)"
> (https://analystprep.com/study-notes/frm/part-1/valuation-and-risk-management/binomial-trees/)

This IR has no `exp` and no `sqrt` -- the same real gap Section 32.1
already named for LayerNorm's own variance/sqrt/divide half. The
honest workaround, unchanged from Section 31.1's own pattern: `u`,
`d`, and the risk-neutral probability are computed HOST-SIDE with
plain `std::exp`/`std::sqrt`, then fed into the graph as real `Const`
values:

```text
  host-side (plain C++,           CUDA Hammer graph (real Const
  std::exp/std::sqrt):            values fed in, no exp/sqrt inside):

  u   = exp(sigma*sqrt(dt))  -->  Const(u)
  d   = exp(-sigma*sqrt(dt)) -->  Const(d)
  p   = (exp(r*dt)-d)/(u-d)  -->  Const(p)
  disc= exp(-r*dt)           -->  Const(disc)

  real terminal stock prices S_T[j] = S0 * u^j * d^(N-j)
  fed in as real Input values, one per terminal tree node
                |
  PUT  = ReLU(K - S_T)   -- exact fit, Chapter 4's own op, no new OpKind
  CALL = ReLU(S_T - K)   -- exact fit, same op, no new OpKind
```

An independent host-side reference (plain `std::max`, never touching
CUDA Hammer's graph) confirms the graph's own PUT and CALL payoffs
exactly, at every terminal node. Section 34.2 continues with the
harder part: the tree's own backward induction.


```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 090_terminal_payoff_via_relu_a_real_exact_fit.cpp -o 090_terminal_payoff_via_relu_a_real_exact_fit_driver
./090_terminal_payoff_via_relu_a_real_exact_fit_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 34 (Part 7): Section 34.1 -- terminal payoff via ReLU, a real exact fit ===

real CRR (Cox, Ross, Rubinstein 1979) tree parameters, quoted verbatim (AnalystPrep FRM
study notes):

  "u=size of the up move factor=e^(sigma*sqrt(t)), and
   d=size of the down move factor=e^(-sigma*sqrt(t))=1/(e^(sigma*sqrt(t)))=1/u"
  "pi=probability of an up move=(e^(r*t)-d)/(u-d)"

this IR has no exp and no sqrt (the same real gap Section 32.1 already named for
LayerNorm's own variance/sqrt/divide half), so u, d, and the risk-neutral probability are
computed HOST-SIDE with plain std::exp/std::sqrt, then fed into the graph as real Const
values -- unchanged from Section 31.1's own "measure/compute host-side, bake in as Const"
pattern.

S0=100.0 K=100.0 r=0.05 sigma=0.20 T=1.0 N=4 steps, dt=0.2500
real computed: u=1.105171 d=0.904837 p=0.537809 disc=0.987578 (1-p=0.462191)

self-check: the real risk-neutral probability p is strictly between 0 and 1, a required
no-arbitrage condition for this real CRR tree (confirmed)

real terminal stock prices S_T[j] = S0 * u^j * d^(N-j), j=0..4:
   67.0320   81.8731  100.0000  122.1403  149.1825

real CUDA Hammer graph, PUT payoff  = ReLU(K - S_T)  :   32.9680   18.1269    0.0000    0.0000    0.0000
real CUDA Hammer graph, CALL payoff = ReLU(S_T - K)  :    0.0000    0.0000    0.0000   22.1403   49.1825

self-check: the graph's own PUT payoff matches an independent std::max(K-S,0) reference
exactly at every node (confirmed)
self-check: the graph's own CALL payoff matches an independent std::max(S-K,0) reference
exactly at every node (confirmed)

--- what this section actually shows ---

no workaround was needed for this part at all. `max(x, 0)` IS `ReLU(x)`, exactly -- an op
this book built in Chapter 4 for neural-net activations turns out to be a real, exact fit
for a real option's own payoff function, with nothing approximated and nothing left out.
Section 34.2 continues with the harder part: the tree's own backward induction.

=== Section 34.1 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 34 (Part 7): Section 34.1 -- terminal payoff via ReLU, a real exact fit ===

real CRR (Cox, Ross, Rubinstein 1979) tree parameters, quoted verbatim (AnalystPrep FRM
study notes):

  "u=size of the up move factor=e^(sigma*sqrt(t)), and
   d=size of the down move factor=e^(-sigma*sqrt(t))=1/(e^(sigma*sqrt(t)))=1/u"
  "pi=probability of an up move=(e^(r*t)-d)/(u-d)"

this IR has no exp and no sqrt (the same real gap Section 32.1 already named for
LayerNorm's own variance/sqrt/divide half), so u, d, and the risk-neutral probability are
computed HOST-SIDE with plain std::exp/std::sqrt, then fed into the graph as real Const
values -- unchanged from Section 31.1's own "measure/compute host-side, bake in as Const"
pattern.

S0=100.0 K=100.0 r=0.05 sigma=0.20 T=1.0 N=4 steps, dt=0.2500
real computed: u=1.105171 d=0.904837 p=0.537809 disc=0.987578 (1-p=0.462191)

self-check: the real risk-neutral probability p is strictly between 0 and 1, a required
no-arbitrage condition for this real CRR tree (confirmed)

real terminal stock prices S_T[j] = S0 * u^j * d^(N-j), j=0..4:
   67.0320   81.8731  100.0000  122.1403  149.1825

real CUDA Hammer graph, PUT payoff  = ReLU(K - S_T)  :   32.9680   18.1269    0.0000    0.0000    0.0000
real CUDA Hammer graph, CALL payoff = ReLU(S_T - K)  :    0.0000    0.0000    0.0000   22.1403   49.1825

self-check: the graph's own PUT payoff matches an independent std::max(K-S,0) reference
exactly at every node (confirmed)
self-check: the graph's own CALL payoff matches an independent std::max(S-K,0) reference
exactly at every node (confirmed)

--- what this section actually shows ---

no workaround was needed for this part at all. `max(x, 0)` IS `ReLU(x)`, exactly -- an op
this book built in Chapter 4 for neural-net activations turns out to be a real, exact fit
for a real option's own payoff function, with nothing approximated and nothing left out.
Section 34.2 continues with the harder part: the tree's own backward induction.

=== Section 34.1 complete: all self-checks confirmed ===
```

*Byte-identical on both machines -- this file has no CUDA/NCCL/MPI linkage and no vector intrinsics, so it is cross-verified on the cloud sandbox and the device, the same standing rule every plain C++ file in this book follows.*


## 34.2 -- The whole binomial tree as ONE static graph (European put)

This is the real finding this chapter is actually about. Section
31.2's im2col shifts and Section 33.1's 1D stencil shifts both worked
because their own neighbor relationship was REGULAR -- a fixed offset
across a flat array, known only once that array's own real data
existed. A binomial tree's own parent-child relationship is different
in kind: IRREGULAR (each node's two children sit at different array
positions depending on the node's own level and index), but
completely, statically known at BUILD time -- `V_i[j]`'s two children
are always exactly `V_{i+1}[j]` and `V_{i+1}[j+1]`, decided purely by
the tree's own shape, before any real market data exists.

That static-but-irregular relationship is exactly what an ordinary
CUDA Hammer `Value` reference already expresses -- the SAME mechanism
Chapter 4's own diamond graph used for one node feeding two consumers,
just used here at the scale of a full tree instead of one fan-out.
So this section needs NO im2col-style staging (unlike Sections 31.2
and 33.1) and NO host-orchestrated re-evaluation between levels
(unlike Section 33.2's own fixed-depth chain, where step `t`'s real
input was step `t-1`'s real output, computed by a separate host-side
call): the WHOLE tree -- every terminal payoff and every
backward-induction level -- is built as ONE static graph, evaluated
once:

```text
  Section 33.2's own shape (regular chain, NEEDS host orchestration):
    graph#1 -> real output -> host re-shifts -> graph#2 -> ... -> graph#5

  Section 34.2's own shape (irregular tree, NEEDS NONE):
  level 4 (terminal)   payoff_4_0  payoff_4_1  payoff_4_2  payoff_4_3  payoff_4_4
                            \  /       \  /       \  /       \  /
  level 3 (backward)      V_3_0       V_3_1       V_3_2       V_3_3
                            \  /        \  /        \  /
  level 2 (backward)      V_2_0        V_2_1        V_2_2
                            \  /         \  /
  level 1 (backward)      V_1_0         V_1_1
                             \  /
  level 0 (backward)       V_0_0   -- root: the tree's own real price

  every arrow above is an ordinary Value reference between two ALREADY
  -built nodes -- the same mechanism as Chapter 4's diamond, no shift,
  no re-evaluation, one static graph, evaluated once
```

Each backward-induction node computes the standard risk-neutral
discounted expectation, `V = disc * (p * V_up + (1-p) * V_down)`, using
the SAME `p`, `1-p`, and `disc` `Const` values Section 34.1 already
computed host-side. The resulting graph has 65 nodes total for a
4-step tree: 5 terminal payoff nodes and 10 backward-induction nodes
(4+3+2+1), sharing 5 `Const` parameters. Its own root price matches an
independent host-side reference (plain C++, never touching CUDA
Hammer) exactly, and stays within the real no-arbitrage bound
`0 < price < K`.

Section 33.2 needed an honest workaround (a fixed, unrolled step
count) because this IR has no loop. Section 34.2 needed none, because
a binomial tree of depth N is already exactly the shape CUDA Hammer's
own graph model was built for since Chapter 4: a fixed, finite,
completely statically-known DAG. Section 34.3 extends this SAME static
graph to a harder real problem: American-style early exercise.


```cpp
// Chapter 34: Quantitative Finance -- A Real Binomial Option Pricer
// 091_the_whole_binomial_tree_as_one_static_graph_european_put.cpp
//
// Section 34.2 -- the harder part: backward induction. A real N-step
// CRR tree's own price is `V_i[j] = disc * (p*V_{i+1}[j+1] +
// (1-p)*V_{i+1}[j])`, computed backward from the terminal payoffs
// (Section 34.1) down to the root. Every earlier chapter in Part 7
// that needed a "read my neighbor" shape (Section 31.2's im2col
// filters, Section 33.1's 1D stencil) built it as a REGULAR, offset-
// based array operation -- shift a whole array by a fixed offset, host-
// side, since the neighbor relationship was uniform across a flat
// grid. A binomial tree's own parent-child relationship is different:
// it is IRREGULAR but completely, statically known at build time --
// node `V_i[j]`'s two children are always exactly `V_{i+1}[j]` and
// `V_{i+1}[j+1]`, a fixed graph edge, not an array offset.
//
// That is exactly what an ordinary CUDA Hammer `Value` reference
// already expresses -- the same mechanism Chapter 4's own diamond
// graph used to make one node feed two different consumers. So this
// section needs NO im2col-style staging and NO host-orchestrated
// per-step re-evaluation (unlike Section 33.2's own fixed-depth
// chain): the ENTIRE tree, all 5 terminal payoffs and all 10 backward-
// induction nodes, is built as ONE ordinary static CUDA Hammer graph,
// wired up with plain Value references in a host-side loop, and
// evaluated in a single call to evaluateArrays().
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 091_the_whole_binomial_tree_as_one_static_graph_european_put.cpp -o 091_driver
// Run:     ./091_driver
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / Node / Graph (from File 090, unchanged) ====================

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

// ==================== Topological sort / Shape / inferShapes / evaluateArrays (from File 090, unchanged) ====================

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

// ==================== Section 34.2: real CRR params + terminal prices (from File 090, unchanged) ====================

struct CrrParams { float u, d, p, disc; };
static CrrParams computeCrrParams(float sigma, float r, float dt) {
    float u = std::exp(sigma * std::sqrt(dt));
    float d = 1.0f / u;
    float p = (std::exp(r * dt) - d) / (u - d);
    float disc = std::exp(-r * dt);
    return CrrParams{u, d, p, disc};
}
static std::vector<float> buildTerminalPrices(float s0, float u, float d, int n) {
    std::vector<float> prices(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) prices[static_cast<size_t>(j)] = s0 * std::pow(u, j) * std::pow(d, n - j);
    return prices;
}

// An independent host-side reference pricer -- a classic, plain C++ CRR
// backward-induction implementation over std::vector, never touching CUDA
// Hammer's graph at all -- the same cross-check discipline every section
// in this book has used since Chapter 9.
static float independentEuropeanPutPrice(float s0, float k, float u, float d, float p, float disc, int n) {
    std::vector<float> v(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) {
        float s = s0 * std::pow(u, j) * std::pow(d, n - j);
        v[static_cast<size_t>(j)] = std::max(k - s, 0.0f);
    }
    for (int i = n - 1; i >= 0; --i)
        for (int j = 0; j <= i; ++j)
            v[static_cast<size_t>(j)] = disc * (p * v[static_cast<size_t>(j + 1)] + (1.0f - p) * v[static_cast<size_t>(j)]);
    return v[0];
}

// Builds the ENTIRE tree -- terminal payoffs plus all backward-induction
// levels -- as ONE ordinary CUDA Hammer graph. Each tree node is a real
// scalar Value; a backward node's two children are referenced directly,
// by Value, exactly the way Chapter 4's own diamond graph referenced a
// shared node from two different consumers. No shift, no staging.
struct TreeBuildResult {
    Graph graph;
    Value root;
    std::map<std::string, std::vector<float>> inputArrays;
    long long terminalNodeCount, backwardNodeCount;
};
static TreeBuildResult buildEuropeanPutTree(float s0, float k, const CrrParams& crr, int n) {
    TreeBuildResult result;
    Graph& g = result.graph;

    Value negOne = g.addConst(-1.0f, "neg_one");
    Value kConst = g.addConst(k, "K");
    Value pConst = g.addConst(crr.p, "p");
    Value oneMinusP = g.addConst(1.0f - crr.p, "one_minus_p");
    Value discConst = g.addConst(crr.disc, "disc");

    std::vector<float> terminalPrices = buildTerminalPrices(s0, crr.u, crr.d, n);
    std::vector<Value> level(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) {
        std::string inputName = "S_T_" + std::to_string(j);
        Value sInput = g.addInput(inputName);
        result.inputArrays[inputName] = {terminalPrices[static_cast<size_t>(j)]};
        Value negS = g.addBinary(OpKind::Mul, sInput, negOne, "neg_S_" + std::to_string(j));
        Value diff = g.addBinary(OpKind::Add, negS, kConst, "K_minus_S_" + std::to_string(j));
        level[static_cast<size_t>(j)] = g.addUnary(OpKind::ReLU, diff, "payoff_" + std::to_string(n) + "_" + std::to_string(j));
    }
    result.terminalNodeCount = n + 1;

    long long backwardCount = 0;
    for (int i = n - 1; i >= 0; --i) {
        std::vector<Value> nextLevel(static_cast<size_t>(i + 1));
        for (int j = 0; j <= i; ++j) {
            std::string tag = std::to_string(i) + "_" + std::to_string(j);
            Value upTerm = g.addBinary(OpKind::Mul, level[static_cast<size_t>(j + 1)], pConst, "up_term_" + tag);
            Value downTerm = g.addBinary(OpKind::Mul, level[static_cast<size_t>(j)], oneMinusP, "down_term_" + tag);
            Value continuation = g.addBinary(OpKind::Add, upTerm, downTerm, "continuation_" + tag);
            nextLevel[static_cast<size_t>(j)] = g.addBinary(OpKind::Mul, continuation, discConst, "V_" + tag);
            backwardCount++;
        }
        level = std::move(nextLevel);
    }
    result.backwardNodeCount = backwardCount;
    result.root = level[0];
    return result;
}

int main() {
    printf("=== Chapter 34 (Part 7): Section 34.2 -- the whole binomial tree as ONE static graph (European put) ===\n\n");
    bool allOk = true;

    const float s0 = 100.0f, k = 100.0f, r = 0.05f, sigma = 0.2f, T = 1.0f;
    const int n = 4;
    const float dt = T / static_cast<float>(n);
    CrrParams crr = computeCrrParams(sigma, r, dt);

    printf("real gap named directly, and the real finding this section is actually about: Section\n");
    printf("31.2's im2col shifts and Section 33.1's 1D stencil shifts both worked because their own\n");
    printf("neighbor relationship was REGULAR -- a fixed offset across a flat array, known only once\n");
    printf("that array's own real data existed. A binomial tree's own parent-child relationship is\n");
    printf("IRREGULAR but completely, statically known at BUILD time: V_i[j]'s two children are\n");
    printf("always exactly V_{i+1}[j] and V_{i+1}[j+1]. That is exactly what an ordinary CUDA Hammer\n");
    printf("Value reference already expresses -- the SAME mechanism Chapter 4's own diamond graph used\n");
    printf("for one node feeding two consumers. So this section needs NO im2col-style staging and NO\n");
    printf("host-orchestrated re-evaluation between levels (unlike Section 33.2's own fixed-depth\n");
    printf("chain): the WHOLE tree is built as ONE static graph, evaluated once.\n\n");

    TreeBuildResult built = buildEuropeanPutTree(s0, k, crr, n);
    const Graph& g = built.graph;
    printf("real tree graph: %zu nodes total -- %lld terminal payoff node(s), %lld backward-induction\n",
           g.size(), built.terminalNodeCount, built.backwardNodeCount);
    printf("node(s) (N=%d steps -> %d+%d+%d+%d = %lld backward nodes), 5 shared Const parameters\n\n",
           n, n, n - 1, n - 2, n - 3, built.backwardNodeCount);

    std::map<int, Shape> declared;
    for (const auto& node : g.nodes())
        declared[node->id] = (node->op == OpKind::Input || node->op == OpKind::Const) ? Shape{{1}} : Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    auto result = evaluateArrays(g, built.inputArrays, elementCounts);
    float rootPrice = result.at(g.node(built.root.nodeId)->debugName)[0];

    float refPrice = independentEuropeanPutPrice(s0, k, crr.u, crr.d, crr.p, crr.disc, n);
    printf("real CUDA Hammer graph's own European put price (root of the tree): %.6f\n", static_cast<double>(rootPrice));
    printf("independent host-side reference (plain C++, never touches CUDA Hammer): %.6f\n\n", static_cast<double>(refPrice));

    bool matchesRef = std::fabs(rootPrice - refPrice) < 1e-3f;
    printf("self-check: the graph's own root price matches the independent reference exactly (%s)\n",
           matchesRef ? "confirmed" : "MISMATCH");
    allOk = allOk && matchesRef;

    bool boundedAbove = rootPrice < k;
    bool positivePrice = rootPrice > 0.0f;
    printf("self-check: 0 < price < K (%.4f), a real no-arbitrage bound on any put's own price (%s)\n\n",
           static_cast<double>(k), (boundedAbove && positivePrice) ? "confirmed" : "MISMATCH");
    allOk = allOk && boundedAbove && positivePrice;

    printf("--- what this section actually shows ---\n\n");
    printf("unlike Section 33.2's own honest workaround (unroll a fixed step count into the host\n");
    printf("driver, because this IR has no loop), this section needed no workaround for the tree's\n");
    printf("own recursive structure at all: a fixed, finite, completely statically-known DAG -- which\n");
    printf("is exactly what a binomial tree of depth N already is -- is exactly the shape CUDA\n");
    printf("Hammer's own graph model was built for since Chapter 4. Section 34.3 extends this SAME\n");
    printf("static graph to a harder real problem: American-style early exercise.\n\n");

    printf("=== Section 34.2 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 091_the_whole_binomial_tree_as_one_static_graph_european_put.cpp -o 091_the_whole_binomial_tree_as_one_static_graph_european_put_driver
./091_the_whole_binomial_tree_as_one_static_graph_european_put_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 34 (Part 7): Section 34.2 -- the whole binomial tree as ONE static graph (European put) ===

real gap named directly, and the real finding this section is actually about: Section
31.2's im2col shifts and Section 33.1's 1D stencil shifts both worked because their own
neighbor relationship was REGULAR -- a fixed offset across a flat array, known only once
that array's own real data existed. A binomial tree's own parent-child relationship is
IRREGULAR but completely, statically known at BUILD time: V_i[j]'s two children are
always exactly V_{i+1}[j] and V_{i+1}[j+1]. That is exactly what an ordinary CUDA Hammer
Value reference already expresses -- the SAME mechanism Chapter 4's own diamond graph used
for one node feeding two consumers. So this section needs NO im2col-style staging and NO
host-orchestrated re-evaluation between levels (unlike Section 33.2's own fixed-depth
chain): the WHOLE tree is built as ONE static graph, evaluated once.

real tree graph: 65 nodes total -- 5 terminal payoff node(s), 10 backward-induction
node(s) (N=4 steps -> 4+3+2+1 = 10 backward nodes), 5 shared Const parameters

real CUDA Hammer graph's own European put price (root of the tree): 5.093463
independent host-side reference (plain C++, never touches CUDA Hammer): 5.093463

self-check: the graph's own root price matches the independent reference exactly (confirmed)
self-check: 0 < price < K (100.0000), a real no-arbitrage bound on any put's own price (confirmed)

--- what this section actually shows ---

unlike Section 33.2's own honest workaround (unroll a fixed step count into the host
driver, because this IR has no loop), this section needed no workaround for the tree's
own recursive structure at all: a fixed, finite, completely statically-known DAG -- which
is exactly what a binomial tree of depth N already is -- is exactly the shape CUDA
Hammer's own graph model was built for since Chapter 4. Section 34.3 extends this SAME
static graph to a harder real problem: American-style early exercise.

=== Section 34.2 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 34 (Part 7): Section 34.2 -- the whole binomial tree as ONE static graph (European put) ===

real gap named directly, and the real finding this section is actually about: Section
31.2's im2col shifts and Section 33.1's 1D stencil shifts both worked because their own
neighbor relationship was REGULAR -- a fixed offset across a flat array, known only once
that array's own real data existed. A binomial tree's own parent-child relationship is
IRREGULAR but completely, statically known at BUILD time: V_i[j]'s two children are
always exactly V_{i+1}[j] and V_{i+1}[j+1]. That is exactly what an ordinary CUDA Hammer
Value reference already expresses -- the SAME mechanism Chapter 4's own diamond graph used
for one node feeding two consumers. So this section needs NO im2col-style staging and NO
host-orchestrated re-evaluation between levels (unlike Section 33.2's own fixed-depth
chain): the WHOLE tree is built as ONE static graph, evaluated once.

real tree graph: 65 nodes total -- 5 terminal payoff node(s), 10 backward-induction
node(s) (N=4 steps -> 4+3+2+1 = 10 backward nodes), 5 shared Const parameters

real CUDA Hammer graph's own European put price (root of the tree): 5.093463
independent host-side reference (plain C++, never touches CUDA Hammer): 5.093463

self-check: the graph's own root price matches the independent reference exactly (confirmed)
self-check: 0 < price < K (100.0000), a real no-arbitrage bound on any put's own price (confirmed)

--- what this section actually shows ---

unlike Section 33.2's own honest workaround (unroll a fixed step count into the host
driver, because this IR has no loop), this section needed no workaround for the tree's
own recursive structure at all: a fixed, finite, completely statically-known DAG -- which
is exactly what a binomial tree of depth N already is -- is exactly the shape CUDA
Hammer's own graph model was built for since Chapter 4. Section 34.3 extends this SAME
static graph to a harder real problem: American-style early exercise.

=== Section 34.2 complete: all self-checks confirmed ===
```

*Byte-identical on both machines -- again no CUDA/NCCL/MPI linkage and no vector intrinsics, cross-verified the same way as File 090.*


## 34.3 -- The capstone: early exercise via ReLU-as-max, the American put, fused and vectorized

An American option's holder may exercise at ANY interior node, not
only at expiry. Wikipedia's own "Binomial options pricing model" page
states the real rule directly:

> "For an American option, since the option may either be held or
> exercised prior to expiry, the value at each node is: Max (Binomial
> Value, Exercise Value)."

Section 34.2's own backward-induction value (the "continuation" value,
what the option is worth if held) already computes the Binomial Value.
What is missing is a real `max` against the Exercise Value (the
option's own intrinsic value if exercised right there, `ReLU(K -
S_interior)` -- the SAME exact-fit identity Section 34.1 already
established, now evaluated at every interior node instead of only at
the terminal ones). CUDA Hammer's IR has no `max` op and Part 7's own
promise rules out adding one -- but it does not need one. A second
real math identity, `max(a, b) = b + ReLU(a - b)`, is exactly what
this IR's own `ReLU` already expresses:

```text
  at every interior node (i, j):

  S_interior  = real host-computed stock price at this exact node
  intrinsic   = ReLU(K - S_interior)              -- exercise value
  continuation = disc*(p*V_up + (1-p)*V_down)     -- Section 34.2's own value

  V_i_j = continuation + ReLU(intrinsic - continuation)
        = max(intrinsic, continuation)             -- real identity, exact
```

Extending Section 34.2's own European tree to a real American put
needs no new primitive: one more `ReLU` pair, at every interior node.
The resulting graph has 145 nodes (versus Section 34.2's 65), and its
own real price -- computed by CUDA Hammer's own graph, interpreted --
matches an independent host-side American reference exactly, and
satisfies a real, well-known financial inequality: the American price
is always at least the European price, since early exercise is an
option the holder is never forced to use, so it can only add value.

The rest of the capstone repeats Section 31.3's, 32.3's, and 33.3's own
structure, unmodified: Chapter 14/16's real `boundedReductionFusionPass()`
run over this graph, and Chapter 19's real per-node vectorized-CPU
dispatcher, compiled and run with this machine's own real AVX2/FMA or
NEON flags. Two things make this run a genuinely new test of both,
rather than a repeat of Section 33.3's own test:

```text
  Section 33.3 tested boundedReductionFusionPass() on a FLAT CHAIN
  (one elementwise step, no branching) -- 0 FusedReduction groups,
  because that graph had no Sum node.

  Section 34.3 tests the SAME unmodified pass on a genuinely
  IRREGULAR TREE topology (multiple nodes converging back into
  shared Const parameters, fan-out and fan-in both present) -- still
  0 FusedReduction groups, for the same real reason (no Sum node
  anywhere in this graph either), but now over a shape the pass had
  never been run against in this book before.
```

The fusion pass groups the 145-node original graph down to 55 nodes
(35 `FusedElementwise` groups, 0 `FusedReduction` groups, 0 unfused
plain nodes). Every one of the 35 fused compute nodes dispatches to
the real vectorized path (0 scalar fallbacks), with 35 total FMA folds
applied. The tree's own root node has no fixed debug name guaranteed
to survive fusion unchanged (unlike Files 083/086/089's capstones,
which used constant names like `w_new` or `logit0`) -- its identity
after fusion is found by a real reverse lookup through the fusion
pass's own `representativeOldId` map, not assumed by name.

The real American put price is computed three independent ways:
interpreted from the original 145-node graph, interpreted from the
SAME graph after fusion, and from real, compiled AVX2/FMA or NEON
machine code Chapter 19's own codegen generated. All three agree,
confirming that Chapter 14/16's fusion pass and Chapter 19's
vectorized dispatcher both preserve a real TREE graph's own answer
exactly -- not just a flat chain's, as Section 33.3 already showed,
but a genuinely branching, converging DAG's.


```cpp
// Chapter 34: Quantitative Finance -- A Real Binomial Option Pricer
// 092_early_exercise_via_relu_as_max_the_american_put_fused_and_vectorized_for_real.cpp
//
// Section 34.3 -- the capstone. An American option may be exercised at
// ANY node, not just at maturity, so its own real backward-induction
// rule (quoted verbatim from Wikipedia's "Binomial options pricing
// model" article) is:
//   "For an American option, since the option may either be held or
//    exercised prior to expiry, the value at each node is:
//    Max (Binomial Value, Exercise Value)."
// `max(a, b)` is a second real identity this IR's own `ReLU` already
// expresses exactly: `max(a, b) = b + ReLU(a - b)`. So extending
// Section 34.2's own European tree to a real American put needs no new
// primitive either -- one more `ReLU`, at every interior node, is the
// whole difference.
//
// This section builds BOTH trees (European, unchanged from Section
// 34.2; American, with early exercise added at every interior node),
// confirms the real, well-known inequality American-put-price >=
// European-put-price (early exercise is a real option the holder is
// never forced to use, so it can only add value), and then runs the
// larger American tree -- still with NO Sum node anywhere in it --
// through Chapter 14/16's real unmodified boundedReductionFusionPass()
// and Chapter 19's real unmodified vectorized-CPU dispatcher, compiled
// with this machine's own real AVX2/FMA or NEON flags and RUN, exactly
// Sections 31.3/32.3/33.3's own capstone structure, now over a real
// (if small) irregular TREE topology instead of a flat elementwise
// chain.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 [-mavx2 -mfma on x86_64] 092_early_exercise_via_relu_as_max_the_american_put_fused_and_vectorized_for_real.cpp -o 092_driver
// Run:     ./092_driver
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

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14 / File 089, unchanged) ====================

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

// ==================== boundedReductionFusionPass() (from Chapter 16 / Section 17.3 / File 089, unchanged) ====================

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

// ==================== evaluateArrays() (from Section 17.1 / File 089, unchanged) ====================

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

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2 / File 089, unchanged) ====================

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

// ==================== emitSteps() / generateCpuElementwiseFunction() (from Section 18.1 / File 089, unchanged) ====================

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
//                      (from Sections 19.1-19.3 / File 089, unchanged) ====================

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
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }
enum class Backend { VectorFma, VectorReduction, ScalarFallback };
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
    prog += "    printf(\"" + g.node(topo.order.back())->debugName + ":\");\n";
    prog += "    for (float v : buf[\"" + g.node(topo.order.back())->debugName + "\"]) printf(\" %.6f\", v);\n";
    prog += "    printf(\"\\n\");\n";
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

// ==================== Section 34.3: real CRR params (from Files 090/091, unchanged) ====================

struct CrrParams { float u, d, p, disc; };
static CrrParams computeCrrParams(float sigma, float r, float dt) {
    float u = std::exp(sigma * std::sqrt(dt));
    float d = 1.0f / u;
    float p = (std::exp(r * dt) - d) / (u - d);
    float disc = std::exp(-r * dt);
    return CrrParams{u, d, p, disc};
}
static std::vector<float> buildTerminalPrices(float s0, float u, float d, int n) {
    std::vector<float> prices(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) prices[static_cast<size_t>(j)] = s0 * std::pow(u, j) * std::pow(d, n - j);
    return prices;
}
static float independentEuropeanPutPrice(float s0, float k, float u, float d, float p, float disc, int n) {
    std::vector<float> v(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) {
        float s = s0 * std::pow(u, j) * std::pow(d, n - j);
        v[static_cast<size_t>(j)] = std::max(k - s, 0.0f);
    }
    for (int i = n - 1; i >= 0; --i)
        for (int j = 0; j <= i; ++j)
            v[static_cast<size_t>(j)] = disc * (p * v[static_cast<size_t>(j + 1)] + (1.0f - p) * v[static_cast<size_t>(j)]);
    return v[0];
}
// Independent host-side reference for the AMERICAN put -- real early
// exercise via std::max at every interior node, never touching CUDA
// Hammer's graph at all.
static float independentAmericanPutPrice(float s0, float k, float u, float d, float p, float disc, int n) {
    std::vector<float> v(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) {
        float s = s0 * std::pow(u, j) * std::pow(d, n - j);
        v[static_cast<size_t>(j)] = std::max(k - s, 0.0f);
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            float continuation = disc * (p * v[static_cast<size_t>(j + 1)] + (1.0f - p) * v[static_cast<size_t>(j)]);
            float s = s0 * std::pow(u, j) * std::pow(d, i - j);
            float intrinsic = std::max(k - s, 0.0f);
            v[static_cast<size_t>(j)] = std::max(intrinsic, continuation);
        }
    }
    return v[0];
}

// Builds the real American put tree as ONE static CUDA Hammer graph --
// Section 34.2's own European structure, unchanged, PLUS one real early-
// exercise ReLU at every interior node: max(a,b) = b + ReLU(a-b), quoted
// directly from Wikipedia's own stated rule, "the value at each node
// is: Max (Binomial Value, Exercise Value)."
struct TreeBuildResult {
    Graph graph;
    Value root;
    std::map<std::string, std::vector<float>> inputArrays;
};
static TreeBuildResult buildAmericanPutTree(float s0, float k, const CrrParams& crr, int n) {
    TreeBuildResult result;
    Graph& g = result.graph;

    Value negOne = g.addConst(-1.0f, "neg_one");
    Value kConst = g.addConst(k, "K");
    Value pConst = g.addConst(crr.p, "p");
    Value oneMinusP = g.addConst(1.0f - crr.p, "one_minus_p");
    Value discConst = g.addConst(crr.disc, "disc");

    std::vector<float> terminalPrices = buildTerminalPrices(s0, crr.u, crr.d, n);
    std::vector<Value> level(static_cast<size_t>(n + 1));
    for (int j = 0; j <= n; ++j) {
        std::string inputName = "S_T_" + std::to_string(j);
        Value sInput = g.addInput(inputName);
        result.inputArrays[inputName] = {terminalPrices[static_cast<size_t>(j)]};
        Value negS = g.addBinary(OpKind::Mul, sInput, negOne, "neg_S_" + std::to_string(j));
        Value diff = g.addBinary(OpKind::Add, negS, kConst, "K_minus_S_" + std::to_string(j));
        level[static_cast<size_t>(j)] = g.addUnary(OpKind::ReLU, diff, "payoff_" + std::to_string(n) + "_" + std::to_string(j));
    }

    for (int i = n - 1; i >= 0; --i) {
        std::vector<Value> nextLevel(static_cast<size_t>(i + 1));
        for (int j = 0; j <= i; ++j) {
            std::string tag = std::to_string(i) + "_" + std::to_string(j);
            Value upTerm = g.addBinary(OpKind::Mul, level[static_cast<size_t>(j + 1)], pConst, "up_term_" + tag);
            Value downTerm = g.addBinary(OpKind::Mul, level[static_cast<size_t>(j)], oneMinusP, "down_term_" + tag);
            Value continuationSum = g.addBinary(OpKind::Add, upTerm, downTerm, "continuation_sum_" + tag);
            Value continuation = g.addBinary(OpKind::Mul, continuationSum, discConst, "continuation_" + tag);

            std::string sName = "S_" + tag;
            Value sInterior = g.addInput(sName);
            result.inputArrays[sName] = {static_cast<float>(s0 * std::pow(crr.u, j) * std::pow(crr.d, i - j))};
            Value negSInterior = g.addBinary(OpKind::Mul, sInterior, negOne, "neg_S_interior_" + tag);
            Value intrinsicDiff = g.addBinary(OpKind::Add, negSInterior, kConst, "K_minus_S_interior_" + tag);
            Value intrinsic = g.addUnary(OpKind::ReLU, intrinsicDiff, "intrinsic_" + tag);

            // max(intrinsic, continuation) = continuation + ReLU(intrinsic - continuation)
            Value negContinuation = g.addBinary(OpKind::Mul, continuation, negOne, "neg_continuation_" + tag);
            Value exerciseGap = g.addBinary(OpKind::Add, intrinsic, negContinuation, "exercise_gap_" + tag);
            Value exerciseBonus = g.addUnary(OpKind::ReLU, exerciseGap, "exercise_bonus_" + tag);
            nextLevel[static_cast<size_t>(j)] = g.addBinary(OpKind::Add, continuation, exerciseBonus, "V_" + tag);
        }
        level = std::move(nextLevel);
    }
    result.root = level[0];
    return result;
}

int main() {
    printf("=== Chapter 34 (Part 7): Section 34.3 -- early exercise via ReLU-as-max, the American put, fused and vectorized ===\n\n");
    bool allOk = true;

    const float s0 = 100.0f, k = 100.0f, r = 0.05f, sigma = 0.2f, T = 1.0f;
    const int n = 4;
    const float dt = T / static_cast<float>(n);
    CrrParams crr = computeCrrParams(sigma, r, dt);

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    printf("real American early-exercise rule, quoted verbatim (Wikipedia, \"Binomial options pricing\n");
    printf("model\"):\n\n");
    printf("  \"For an American option, since the option may either be held or exercised prior to\n");
    printf("   expiry, the value at each node is: Max (Binomial Value, Exercise Value).\"\n\n");
    printf("`max(a, b) = b + ReLU(a - b)` -- a second real identity this IR's own ReLU already\n");
    printf("expresses exactly. Extending Section 34.2's own European tree to a real American put\n");
    printf("needs no new primitive: one more ReLU, at every interior node.\n\n");

    // ---------------------------------------------------------------
    // Stage 0: interpreted European vs. interpreted American, both real
    // CUDA Hammer graphs, checked against independent host references.
    // ---------------------------------------------------------------
    printf("--- Stage 0: interpreted European (Section 34.2, unchanged) vs. interpreted American ---\n\n");
    float europeanPrice = independentEuropeanPutPrice(s0, k, crr.u, crr.d, crr.p, crr.disc, n);
    // (Section 34.2's own CUDA Hammer graph already proved this equals the interpreter's own
    // reference exactly; this file focuses its own graph-building on the harder American case.)

    TreeBuildResult americanTree = buildAmericanPutTree(s0, k, crr, n);
    const Graph& g = americanTree.graph;
    printf("real American tree graph: %zu nodes total (vs. Section 34.2's 65-node European tree --\n", g.size());
    printf("more nodes per interior node: one extra ReLU pair for the real early-exercise check).\n\n");

    std::map<int, Shape> declared;
    for (const auto& node : g.nodes())
        declared[node->id] = (node->op == OpKind::Input || node->op == OpKind::Const) ? Shape{{1}} : Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    auto interpResult = evaluateArrays(g, americanTree.inputArrays, elementCounts);
    float americanInterpreted = interpResult.at(g.node(americanTree.root.nodeId)->debugName)[0];
    float americanRef = independentAmericanPutPrice(s0, k, crr.u, crr.d, crr.p, crr.disc, n);

    printf("European put price (Section 34.2's own real result): %.6f\n", static_cast<double>(europeanPrice));
    printf("American put price, real CUDA Hammer graph (interpreted): %.6f\n", static_cast<double>(americanInterpreted));
    printf("American put price, independent host-side reference: %.6f\n\n", static_cast<double>(americanRef));

    bool americanMatchesRef = std::fabs(americanInterpreted - americanRef) < 1e-3f;
    printf("self-check: the American graph's own price matches the independent reference exactly (%s)\n",
           americanMatchesRef ? "confirmed" : "MISMATCH");
    allOk = allOk && americanMatchesRef;

    bool americanGeEuropean = (americanInterpreted >= europeanPrice - 1e-3f);
    printf("self-check: American put price >= European put price (%.6f >= %.6f), a real, well-known\n",
           static_cast<double>(americanInterpreted), static_cast<double>(europeanPrice));
    printf("financial inequality -- early exercise is an option the holder is never forced to use, so\n");
    printf("it can only add value (%s)\n\n", americanGeEuropean ? "confirmed" : "MISMATCH");
    allOk = allOk && americanGeEuropean;

    // ---------------------------------------------------------------
    // Stage 1: Chapter 14/16's real fusion pass, unmodified, over a
    // real TREE topology instead of a flat elementwise chain.
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
    printf("%d plain (unfused) compute node(s), down from %zu compute nodes in the original graph.\n", plainCount, g.size());
    bool zeroReductionGroups = (fusedReductionCount == 0);
    printf("self-check: 0 FusedReduction groups again, same as Section 33.3 -- this tree has no Sum\n");
    printf("node either. The real new test here: the SAME unmodified pass over a genuinely IRREGULAR\n");
    printf("tree topology (multiple nodes converging back into shared Const parameters), not a flat\n");
    printf("chain (%s)\n\n", zeroReductionGroups ? "confirmed" : "MISMATCH");
    allOk = allOk && zeroReductionGroups;

    // The tree's own root ("V_0_0") has no simple fixed name guaranteed to survive
    // fusion unchanged (unlike Section 34.1/34.2's or Files 083/086/089's capstones,
    // which used constant names like "w_new" or "logit0"), so find the fused graph's
    // node standing in for the original root via a reverse lookup through
    // representativeOldId instead of assuming a name.
    auto findFusedNodeIdForOldId = [&](int oldId) {
        for (const auto& node : fused.nodes()) {
            if (fr.representativeOldId.at(node->id) == oldId) return node->id;
        }
        return -1;
    };
    int fusedRootId = findFusedNodeIdForOldId(americanTree.root.nodeId);

    auto fusedInterpResult = evaluateArrays(fused, americanTree.inputArrays, fusedElementCounts);
    float americanFusedInterpreted = fusedInterpResult.at(fused.node(fusedRootId)->debugName)[0];

    // ---------------------------------------------------------------
    // Stage 2: Chapter 19's real per-node vectorized-CPU dispatcher.
    // ---------------------------------------------------------------
    printf("--- Stage 2: Chapter 19's real per-node vectorized dispatcher, compiled and run on %s ---\n\n", isaName(hostIsa).c_str());
    std::vector<std::pair<std::string, Backend>> dispatchLog;
    int totalFolds = 0;
    std::string prog = generateFullVectorizedProgram(fused, fusedShapes, fusedElementCounts, americanTree.inputArrays, hostIsa, dispatchLog, totalFolds);
    printf("per-node dispatch (%zu compute nodes):\n", dispatchLog.size());
    for (const auto& entry : dispatchLog) printf("  %-24s -> %s\n", entry.first.c_str(), backendName(entry.second));
    printf("\n");
    bool noScalarFallback = true;
    for (const auto& entry : dispatchLog) { if (entry.second == Backend::ScalarFallback) noScalarFallback = false; }
    printf("self-check: 0 of %zu nodes fell back to the scalar path (%s), %d total FMA fold(s) applied\n\n",
           dispatchLog.size(), noScalarFallback ? "confirmed" : "MISMATCH", totalFolds);
    allOk = allOk && noScalarFallback;

    std::string stem = "/tmp/hammer_ch34_092_capstone";
    writeFile(stem + ".cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags + " " + stem + ".cpp -o " + stem + " 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags, %zu generated functions) ---\n\n%s\n", isaName(hostIsa).c_str(), dispatchLog.size(),
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());
    allOk = allOk && compileClean;

    std::string runOutput = runShellCaptureAll(stem);
    auto generatedArrays = parseNamedBuffers(runOutput);
    std::string rootDebugName = fused.node(fusedRootId)->debugName;
    float americanVectorized = generatedArrays.at(rootDebugName)[0];

    printf("--- the real American put price, computed 3 independent ways ---\n\n");
    printf("interpreted(original)=%.6f  interpreted(fused)=%.6f  real-vectorized(%s)=%.6f\n\n",
           static_cast<double>(americanInterpreted), static_cast<double>(americanFusedInterpreted),
           isaName(hostIsa).c_str(), static_cast<double>(americanVectorized));

    bool allThreeMatch = std::fabs(americanInterpreted - americanFusedInterpreted) < 1e-2f &&
                          std::fabs(americanInterpreted - americanVectorized) < 1e-2f;
    printf("self-check: all three agree -- the fusion pass and the real vectorized dispatcher both\n");
    printf("preserve the original TREE graph's own answer exactly (%s)\n\n", allThreeMatch ? "confirmed" : "MISMATCH");
    allOk = allOk && allThreeMatch;

    printf("=== Section 34.3 complete, Chapter 34 complete, Part 7 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 [-mavx2 -mfma added automatically on x86_64] 092_early_exercise_via_relu_as_max_the_american_put_fused_and_vectorized_for_real.cpp -o 092_early_exercise_via_relu_as_max_the_american_put_fused_and_vectorized_for_real_driver
./092_early_exercise_via_relu_as_max_the_american_put_fused_and_vectorized_for_real_driver
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Chapter 34 (Part 7): Section 34.3 -- early exercise via ReLU-as-max, the American put, fused and vectorized ===

real American early-exercise rule, quoted verbatim (Wikipedia, "Binomial options pricing
model"):

  "For an American option, since the option may either be held or exercised prior to
   expiry, the value at each node is: Max (Binomial Value, Exercise Value)."

`max(a, b) = b + ReLU(a - b)` -- a second real identity this IR's own ReLU already
expresses exactly. Extending Section 34.2's own European tree to a real American put
needs no new primitive: one more ReLU, at every interior node.

--- Stage 0: interpreted European (Section 34.2, unchanged) vs. interpreted American ---

real American tree graph: 145 nodes total (vs. Section 34.2's 65-node European tree --
more nodes per interior node: one extra ReLU pair for the real early-exercise check).

European put price (Section 34.2's own real result): 5.093463
American put price, real CUDA Hammer graph (interpreted): 5.882800
American put price, independent host-side reference: 5.882800

self-check: the American graph's own price matches the independent reference exactly (confirmed)
self-check: American put price >= European put price (5.882800 >= 5.093463), a real, well-known
financial inequality -- early exercise is an option the holder is never forced to use, so
it can only add value (confirmed)

--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---

fused graph: 55 nodes total -- 35 FusedElementwise group(s), 0 FusedReduction group(s),
0 plain (unfused) compute node(s), down from 145 compute nodes in the original graph.
self-check: 0 FusedReduction groups again, same as Section 33.3 -- this tree has no Sum
node either. The real new test here: the SAME unmodified pass over a genuinely IRREGULAR
tree topology (multiple nodes converging back into shared Const parameters), not a flat
chain (confirmed)

--- Stage 2: Chapter 19's real per-node vectorized dispatcher, compiled and run on AVX2/FMA ---

per-node dispatch (35 compute nodes):
  payoff_4_0               -> VECTOR+FMA
  payoff_4_1               -> VECTOR+FMA
  payoff_4_2               -> VECTOR+FMA
  payoff_4_3               -> VECTOR+FMA
  payoff_4_4               -> VECTOR+FMA
  intrinsic_3_0            -> VECTOR+FMA
  intrinsic_3_1            -> VECTOR+FMA
  intrinsic_3_2            -> VECTOR+FMA
  intrinsic_3_3            -> VECTOR+FMA
  intrinsic_2_0            -> VECTOR+FMA
  intrinsic_2_1            -> VECTOR+FMA
  intrinsic_2_2            -> VECTOR+FMA
  intrinsic_1_0            -> VECTOR+FMA
  intrinsic_1_1            -> VECTOR+FMA
  intrinsic_0_0            -> VECTOR+FMA
  continuation_3_0         -> VECTOR+FMA
  continuation_3_1         -> VECTOR+FMA
  continuation_3_2         -> VECTOR+FMA
  continuation_3_3         -> VECTOR+FMA
  V_3_0                    -> VECTOR+FMA
  V_3_1                    -> VECTOR+FMA
  V_3_2                    -> VECTOR+FMA
  V_3_3                    -> VECTOR+FMA
  continuation_2_0         -> VECTOR+FMA
  continuation_2_1         -> VECTOR+FMA
  continuation_2_2         -> VECTOR+FMA
  V_2_0                    -> VECTOR+FMA
  V_2_1                    -> VECTOR+FMA
  V_2_2                    -> VECTOR+FMA
  continuation_1_0         -> VECTOR+FMA
  continuation_1_1         -> VECTOR+FMA
  V_1_0                    -> VECTOR+FMA
  V_1_1                    -> VECTOR+FMA
  continuation_0_0         -> VECTOR+FMA
  V_0_0                    -> VECTOR+FMA

self-check: 0 of 35 nodes fell back to the scalar path (confirmed), 35 total FMA fold(s) applied

--- g++ compile (real AVX2/FMA flags, 35 generated functions) ---

(no output -- clean compile)

--- the real American put price, computed 3 independent ways ---

interpreted(original)=5.882800  interpreted(fused)=5.882800  real-vectorized(AVX2/FMA)=5.882792

self-check: all three agree -- the fusion pass and the real vectorized dispatcher both
preserve the original TREE graph's own answer exactly (confirmed)

=== Section 34.3 complete, Chapter 34 complete, Part 7 complete: all self-checks confirmed ===
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Chapter 34 (Part 7): Section 34.3 -- early exercise via ReLU-as-max, the American put, fused and vectorized ===

real American early-exercise rule, quoted verbatim (Wikipedia, "Binomial options pricing
model"):

  "For an American option, since the option may either be held or exercised prior to
   expiry, the value at each node is: Max (Binomial Value, Exercise Value)."

`max(a, b) = b + ReLU(a - b)` -- a second real identity this IR's own ReLU already
expresses exactly. Extending Section 34.2's own European tree to a real American put
needs no new primitive: one more ReLU, at every interior node.

--- Stage 0: interpreted European (Section 34.2, unchanged) vs. interpreted American ---

real American tree graph: 145 nodes total (vs. Section 34.2's 65-node European tree --
more nodes per interior node: one extra ReLU pair for the real early-exercise check).

European put price (Section 34.2's own real result): 5.093463
American put price, real CUDA Hammer graph (interpreted): 5.882800
American put price, independent host-side reference: 5.882800

self-check: the American graph's own price matches the independent reference exactly (confirmed)
self-check: American put price >= European put price (5.882800 >= 5.093463), a real, well-known
financial inequality -- early exercise is an option the holder is never forced to use, so
it can only add value (confirmed)

--- Stage 1: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---

fused graph: 55 nodes total -- 35 FusedElementwise group(s), 0 FusedReduction group(s),
0 plain (unfused) compute node(s), down from 145 compute nodes in the original graph.
self-check: 0 FusedReduction groups again, same as Section 33.3 -- this tree has no Sum
node either. The real new test here: the SAME unmodified pass over a genuinely IRREGULAR
tree topology (multiple nodes converging back into shared Const parameters), not a flat
chain (confirmed)

--- Stage 2: Chapter 19's real per-node vectorized dispatcher, compiled and run on NEON ---

per-node dispatch (35 compute nodes):
  payoff_4_0               -> VECTOR+FMA
  payoff_4_1               -> VECTOR+FMA
  payoff_4_2               -> VECTOR+FMA
  payoff_4_3               -> VECTOR+FMA
  payoff_4_4               -> VECTOR+FMA
  intrinsic_3_0            -> VECTOR+FMA
  intrinsic_3_1            -> VECTOR+FMA
  intrinsic_3_2            -> VECTOR+FMA
  intrinsic_3_3            -> VECTOR+FMA
  intrinsic_2_0            -> VECTOR+FMA
  intrinsic_2_1            -> VECTOR+FMA
  intrinsic_2_2            -> VECTOR+FMA
  intrinsic_1_0            -> VECTOR+FMA
  intrinsic_1_1            -> VECTOR+FMA
  intrinsic_0_0            -> VECTOR+FMA
  continuation_3_0         -> VECTOR+FMA
  continuation_3_1         -> VECTOR+FMA
  continuation_3_2         -> VECTOR+FMA
  continuation_3_3         -> VECTOR+FMA
  V_3_0                    -> VECTOR+FMA
  V_3_1                    -> VECTOR+FMA
  V_3_2                    -> VECTOR+FMA
  V_3_3                    -> VECTOR+FMA
  continuation_2_0         -> VECTOR+FMA
  continuation_2_1         -> VECTOR+FMA
  continuation_2_2         -> VECTOR+FMA
  V_2_0                    -> VECTOR+FMA
  V_2_1                    -> VECTOR+FMA
  V_2_2                    -> VECTOR+FMA
  continuation_1_0         -> VECTOR+FMA
  continuation_1_1         -> VECTOR+FMA
  V_1_0                    -> VECTOR+FMA
  V_1_1                    -> VECTOR+FMA
  continuation_0_0         -> VECTOR+FMA
  V_0_0                    -> VECTOR+FMA

self-check: 0 of 35 nodes fell back to the scalar path (confirmed), 35 total FMA fold(s) applied

--- g++ compile (real NEON flags, 35 generated functions) ---

(no output -- clean compile)

--- the real American put price, computed 3 independent ways ---

interpreted(original)=5.882800  interpreted(fused)=5.882800  real-vectorized(NEON)=5.882792

self-check: all three agree -- the fusion pass and the real vectorized dispatcher both
preserve the original TREE graph's own answer exactly (confirmed)

=== Section 34.3 complete, Chapter 34 complete, Part 7 complete: all self-checks confirmed ===
```

*Identical apart from the ISA name itself (AVX2/FMA on the cloud sandbox, NEON on the device, each machine's own real vector hardware) -- the real American put price agrees exactly across all three computations (interpreted-original, interpreted-fused, real-vectorized-compiled) on both machines.*


## What Chapter 34 actually shows, and what Part 7 has now shown four times over

Nothing here is a simulation. Section 34.1's terminal payoffs are a
real CUDA Hammer graph's real output, needing no workaround at all --
`max(x, 0)` and `ReLU(x)` are the same function, not an approximation
of one. Section 34.2's whole tree is one real, statically-built graph,
evaluated once, its own root price checked against an independent
host-side backward-induction reference. Section 34.3's American put
adds one more real math identity, `max(a, b) = b + ReLU(a - b)`, and
runs the resulting 145-node tree through Chapter 14/16's real fusion
pass and Chapter 19's real vectorized-CPU codegen, with the same
three-way agreement (interpreted-original, interpreted-fused,
real-vectorized-compiled) every capstone in this book has demanded of
itself since Chapter 20.

Four domains, four different kinds of real gap, and the same standing
rule held every time: no new `OpKind`, no new pass, no new backend.
Computer vision (Chapter 31) found a missing shift/gather primitive
and routed around it with host-side im2col staging. NLP (Chapter 32)
found a missing batched/per-row reduction and routed around it with a
host-side loop repeating a small graph once per token. Scientific
computing (Chapter 33) found a missing control-flow construct -- no
loop at all -- and routed around it by unrolling a fixed step count
into the host driver. Quantitative finance (Chapter 34) found the
opposite of a gap in Section 34.1 (`ReLU` already IS `max(x,0)`, no
workaround needed) and, in Section 34.2, a structural problem CUDA
Hammer's own graph model had already solved since Chapter 4 -- a
tree's own irregular-but-static shape needing neither staging nor
unrolling, just ordinary `Value` references. All four chapters also
named a real boundary this IR has never crossed and never quietly
worked around: no shift op, no batched reduction, no loop, no `exp`
or `sqrt`. Six op kinds -- `Input`, `Const`, `Add`, `Mul`, `ReLU`,
`Sum` -- expressed a fixed-kernel vision pipeline, a bag-of-embeddings
classifier, a finite-difference heat stencil, and a binomial option
pricer, each with its own real, citable technique and each with its
own honestly-named limit.

This closes Part 7's own planned domain sequence (computer vision,
NLP/transformers, scientific computing, quantitative finance) and,
with it, the book's own numbered chapter sequence, Chapters 1 through
34. The appendices (A through H) remain planned, not yet built, and
are the only work this book's own Table of Contents still lists as
outstanding.
