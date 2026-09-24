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
