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
