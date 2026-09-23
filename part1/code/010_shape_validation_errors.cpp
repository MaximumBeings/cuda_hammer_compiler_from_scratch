// Chapter 6: Graph Validation and Shape Inference
// 010_shape_validation_errors.cpp
//
// Section 6.3 -- catching a real shape mismatch, and a missing declared
// leaf shape, before a graph is ever handed to a later pass or codegen
// stage that would trust its shapes blindly.
//
// Reuses File 009's own Shape/broadcastShapes()/Graph/topologicalSort()/
// inferShapes() completely unchanged. The only new code here is the test
// harness: two genuinely shape-incompatible graphs, one graph missing a
// declared leaf shape, and one valid control graph (File 009's own
// diamond) confirming none of these checks misfire on correct input --
// the same four-invalid-cases-plus-one-control-case discipline Chapter
// 5's own File 008 used for parse errors.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 010_shape_validation_errors.cpp -o 010_shape_validation_errors
// Run:     ./010_shape_validation_errors
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include <functional>

// ==================== Value / Node / Graph / TopoResult / Shape (from File 009) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU };

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;
    float constValue = 0.0f;
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
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) {
        return addNode(op, {lhs, rhs}, name);
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

struct TopoResult {
    std::vector<int> order;
    bool ok = true;
};

static TopoResult topologicalSort(const Graph& g) {
    std::map<int, int> inDegree;
    for (const auto& n : g.nodes()) inDegree[n->id] = static_cast<int>(n->inputs.size());
    std::deque<int> ready;
    for (const auto& n : g.nodes()) {
        if (inDegree[n->id] == 0) ready.push_back(n->id);
    }
    TopoResult result;
    while (!ready.empty()) {
        int id = ready.front();
        ready.pop_front();
        result.order.push_back(id);
        for (const auto& n : g.nodes()) {
            for (const Value& in : n->inputs) {
                if (in.nodeId == id) {
                    if (--inDegree[n->id] == 0) ready.push_back(n->id);
                }
            }
        }
    }
    result.ok = (result.order.size() == g.size());
    return result;
}

struct Shape {
    std::vector<int> dims;
};

static std::string shapeStr(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s.dims[i]);
    }
    out += "]";
    return out;
}

static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        int outDim;
        if (da == db) {
            outDim = da;
        } else if (da == 1) {
            outDim = db;
        } else if (db == 1) {
            outDim = da;
        } else {
            throw std::runtime_error(context + ": shapes " + shapeStr(a) + " and " + shapeStr(b) +
                                      " are not broadcast-compatible (" + std::to_string(da) + " vs " +
                                      std::to_string(db) + " at trailing position " + std::to_string(i) + ")");
        }
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

        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            auto it = declaredShapes.find(id);
            if (it == declaredShapes.end()) {
                throw std::runtime_error(context + ": no declared shape was provided for this leaf node");
            }
            shapes[id] = it->second;
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            const Shape& lhs = shapes.at(n->inputs[0].nodeId);
            const Shape& rhs = shapes.at(n->inputs[1].nodeId);
            shapes[id] = broadcastShapes(lhs, rhs, context);
        } else {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

// ============================== Test harness (new) ==============================

struct TestCase {
    std::string name;
    std::function<void()> run;  // throws on failure, returns normally on success
    bool expectError;
};

int main() {
    printf("=== Section 6.3: rejecting shape-invalid graphs instead of trusting them blindly ===\n\n");

    std::vector<TestCase> tests;

    tests.push_back({
        "valid graph (control case: File 009's own diamond)",
        [] {
            Graph g;
            Value a  = g.addInput("a");
            Value b  = g.addInput("b");
            Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
            Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
            Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
            g.addBinary(OpKind::Add, t2, t3, "out");
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
            inferShapes(g, declared);
        },
        false
    });

    tests.push_back({
        "incompatible shapes, neither dimension matches or is 1",
        [] {
            Graph g;
            Value a = g.addInput("a");
            Value b = g.addInput("b");
            g.addBinary(OpKind::Add, a, b, "bad");
            // [3, 4] vs [5, 6] -- at the trailing position, 4 vs 6: not
            // equal, and neither is 1. Genuinely incompatible.
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{5, 6}}}};
            inferShapes(g, declared);
        },
        true
    });

    tests.push_back({
        "incompatible shapes, only the leading dimension differs (still rejected)",
        [] {
            Graph g;
            Value a = g.addInput("a");
            Value b = g.addInput("b");
            g.addBinary(OpKind::Mul, a, b, "bad");
            // [3, 4] vs [3, 5] -- trailing position 4 vs 5: not equal,
            // neither is 1, so this is STILL incompatible even though
            // the OTHER dimension (3 vs 3) matches perfectly -- every
            // aligned pair has to be individually compatible, matching
            // is not "good enough on average."
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{3, 5}}}};
            inferShapes(g, declared);
        },
        true
    });

    tests.push_back({
        "missing declared shape for a leaf node",
        [] {
            Graph g;
            Value a = g.addInput("a");
            Value b = g.addInput("b");  // deliberately never given a declared shape below
            g.addBinary(OpKind::Add, a, b, "bad");
            std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}};  // "b" missing on purpose
            inferShapes(g, declared);
        },
        true
    });

    bool allBehavedAsExpected = true;
    for (const auto& tc : tests) {
        printf("--- %s ---\n", tc.name.c_str());
        bool threw = false;
        std::string errorMessage;
        try {
            tc.run();
        } catch (const std::exception& e) {
            threw = true;
            errorMessage = e.what();
        }
        bool behavedAsExpected = (threw == tc.expectError);
        allBehavedAsExpected = allBehavedAsExpected && behavedAsExpected;

        if (threw) {
            printf("  threw: \"%s\"\n", errorMessage.c_str());
        } else {
            printf("  inferred successfully, no exception thrown\n");
        }
        printf("  expected %s, got %s -- %s\n\n",
               tc.expectError ? "an exception" : "success",
               threw ? "an exception" : "success",
               behavedAsExpected ? "confirmed" : "MISMATCH");
    }

    printf("self-check: every test case behaved exactly as expected (%s)\n",
           allBehavedAsExpected ? "confirmed" : "MISMATCH");

    return allBehavedAsExpected ? 0 : 1;
}
