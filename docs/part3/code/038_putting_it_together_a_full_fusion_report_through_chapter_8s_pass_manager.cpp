// Chapter 16: Fusion Boundaries
// 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager.cpp
//
// Section 16.3 -- the capstone for Part 3. A single graph, built to exercise
// every boundary this Part has ever introduced at once: a SHARED value
// (Chapter 13), a REDUCTION (Chapter 14), a long single-consumer CHAIN long
// enough to trigger Section 16.2's own size cap, and the graph's own
// GraphRoot (Section 16.1). boundedReductionFusionPass() combines Chapter
// 14's own Sum-handling with Section 16.2's own size-cap mechanism into one
// pass, plugged into Chapter 8's still-completely-unmodified runPasses()
// exactly the way every fusion pass in this book has been -- and a fusion
// REPORT, built from Section 16.1's own classifyNode() extended with one
// more reason, explains every boundary in the final structure by name.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager.cpp -o 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager
// Run:     ./038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <functional>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input:            return "Input";
        case OpKind::Const:            return "Const";
        case OpKind::Add:              return "Add";
        case OpKind::Mul:              return "Mul";
        case OpKind::ReLU:             return "ReLU";
        case OpKind::Sum:              return "Sum";
        case OpKind::FusedElementwise: return "FusedElementwise";
        default:                       return "FusedReduction";
    }
}

enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand {
    OperandKind kind;
    int index;
};

struct FusedStep {
    OpKind op;
    std::vector<FusedOperand> operands;
    long long reduceElementCount = 1;
};

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
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) {
        return addNode(op, {lhs, rhs}, name);
    }
    Value addFusedElementwise(std::vector<Value> externalInputs, std::vector<FusedStep> steps,
                               const std::string& name) {
        Value out = addNode(OpKind::FusedElementwise, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
    }
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps,
                             const std::string& name) {
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

// ==================== Shape / broadcastShapes / inferShapes (from Chapters 6 and 14, unchanged) ====================

struct Shape { std::vector<int> dims; };

static std::string shapeStr(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) { if (i) out += ", "; out += std::to_string(s.dims[i]); }
    out += "]";
    return out;
}
static long long numElements(const Shape& s) {
    long long n = 1;
    for (int d : s.dims) n *= d;
    return n;
}
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
        else throw std::runtime_error(context + ": shapes " + shapeStr(a) + " and " + shapeStr(b) + " incompatible");
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
            shapes[id] = declaredShapes.at(id);
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        } else if (n->op == OpKind::ReLU) {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        } else {  // Sum
            shapes[id] = Shape{};
        }
    }
    return shapes;
}

// ==================== Debug printer / evaluate() (from Chapter 14, unchanged) ====================

static std::string formatGraph(const Graph& g) {
    std::string out;
    char buf[256];
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise && n->op != OpKind::FusedReduction) {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            out += buf;
            if (n->op == OpKind::Const) {
                snprintf(buf, sizeof(buf), "%g", n->constValue);
                out += buf;
            } else {
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    if (i) out += ", ";
                    snprintf(buf, sizeof(buf), "%%%d", n->inputs[i].nodeId);
                    out += buf;
                }
            }
            out += ")\n";
        } else {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(external inputs: ", n->id, n->debugName.c_str(),
                     opKindStr(n->op).c_str());
            out += buf;
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                snprintf(buf, sizeof(buf), "ext%zu=%%%d", i, n->inputs[i].nodeId);
                out += buf;
            }
            out += ")\n";
            for (size_t s = 0; s < n->fusedSteps.size(); ++s) {
                const FusedStep& step = n->fusedSteps[s];
                snprintf(buf, sizeof(buf), "      step%zu = %s(", s, opKindStr(step.op).c_str());
                out += buf;
                for (size_t i = 0; i < step.operands.size(); ++i) {
                    if (i) out += ", ";
                    const FusedOperand& o = step.operands[i];
                    snprintf(buf, sizeof(buf), "%s%d", o.kind == OperandKind::ExternalInput ? "ext" : "step", o.index);
                    out += buf;
                }
                out += ")";
                if (step.op == OpKind::Sum) {
                    snprintf(buf, sizeof(buf), "  [reduces %lld elements]", step.reduceElementCount);
                    out += buf;
                }
                out += (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output\n" : "\n";
            }
        }
    }
    return out;
}
static void printGraph(const Graph& g) { printf("%s", formatGraph(g).c_str()); }

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName,
                                              const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");
    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) {
            v = inputValuesByName.at(n->debugName);
        } else if (n->op == OpKind::Const) {
            v = n->constValue;
        } else if (n->op == OpKind::Add) {
            v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        } else if (n->op == OpKind::Mul) {
            v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        } else if (n->op == OpKind::ReLU) {
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        } else if (n->op == OpKind::Sum) {
            v = valuesById.at(n->inputs[0].nodeId) * static_cast<float>(elementCounts.at(n->inputs[0].nodeId));
        } else {  // FusedElementwise or FusedReduction
            std::vector<float> externalVals;
            for (const Value& in : n->inputs) externalVals.push_back(valuesById.at(in.nodeId));
            std::vector<float> stepVals;
            for (const FusedStep& step : n->fusedSteps) {
                auto read = [&](const FusedOperand& o) {
                    return (o.kind == OperandKind::ExternalInput) ? externalVals[o.index] : stepVals[o.index];
                };
                float sv;
                if (step.op == OpKind::ReLU) sv = std::max(0.0f, read(step.operands[0]));
                else if (step.op == OpKind::Add) sv = read(step.operands[0]) + read(step.operands[1]);
                else if (step.op == OpKind::Mul) sv = read(step.operands[0]) * read(step.operands[1]);
                else sv = read(step.operands[0]) * static_cast<float>(step.reduceElementCount);  // Sum
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== boundedReductionFusionPass(): Chapter 14's Sum rule + Section 16.2's size cap ====================

struct FusionResult {
    Graph graph;
    std::map<int, int> representativeOldId;
};

static std::map<int, long long> computeChainDepths(const Graph& g, const std::map<int, int>& consumers,
                                                     const TopoResult& topo) {
    std::map<int, long long> depth;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        bool isChainCandidate = (n->op == OpKind::Add || n->op == OpKind::Mul || n->op == OpKind::ReLU) &&
                                 consumers.at(id) == 1;
        if (!isChainCandidate) {
            depth[id] = 0;
            continue;
        }
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
static std::set<int> computeSizeCapBoundaries(const Graph& g, const std::map<int, int>& consumers,
                                               long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("computeSizeCapBoundaries: graph is not acyclic");
    std::map<int, long long> depth = computeChainDepths(g, consumers, topo);
    std::set<int> boundaries;
    for (const auto& kv : depth) {
        if (kv.second > 0 && kv.second % maxChainLength == 0) boundaries.insert(kv.first);
    }
    return boundaries;
}

// resolveIntoGroup(), carrying BOTH of this book's own two independent
// extensions to Chapter 13's original three-condition mustBeExternal: Sum
// (Chapter 14, a correctness boundary) and sizeCapBoundaries (Section 16.2,
// a cost boundary). Neither condition knows the other exists -- they are
// simply two more entries in the same OR chain, exactly as independent in
// code as they are in the reasons behind them.
static FusedOperand resolveIntoGroupBounded(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                             const std::set<int>& sizeCapBoundaries,
                                             const std::map<int, Value>& materialized,
                                             std::vector<Value>& externalInputs,
                                             std::map<int, int>& externalIndexByOldId,
                                             std::vector<FusedStep>& steps,
                                             std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) {
        return FusedOperand{OperandKind::PriorStep, stepIt->second};
    }
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
                           (consumers.at(oldId) != 1) ||
                           (sizeCapBoundaries.count(oldId) > 0);
    if (mustBeExternal) {
        auto extIt = externalIndexByOldId.find(oldId);
        if (extIt != externalIndexByOldId.end()) {
            return FusedOperand{OperandKind::ExternalInput, extIt->second};
        }
        int idx = static_cast<int>(externalInputs.size());
        externalInputs.push_back(materialized.at(oldId));
        externalIndexByOldId[oldId] = idx;
        return FusedOperand{OperandKind::ExternalInput, idx};
    }
    std::vector<FusedOperand> operands;
    for (const Value& in : p->inputs) {
        operands.push_back(resolveIntoGroupBounded(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                     externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
    }
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}

static FusionResult boundedReductionFusionPass(const Graph& g, const std::map<int, long long>& elementCounts,
                                                long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("boundedReductionFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;

    std::set<int> sizeCapBoundaries = computeSizeCapBoundaries(g, consumers, maxChainLength);

    FusionResult result;
    Graph& out = result.graph;
    std::map<int, Value> materialized;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        if (n->op == OpKind::Input) {
            Value v = out.addInput(n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Const) {
            Value v = out.addConst(n->constValue, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (n->op == OpKind::Sum) {
            std::vector<Value> externalInputs;
            std::map<int, int> externalIndexByOldId;
            std::vector<FusedStep> steps;
            std::map<int, int> stepIndexByOldId;
            FusedOperand summedOperand = resolveIntoGroupBounded(n->inputs[0].nodeId, g, consumers, sizeCapBoundaries,
                                                                   materialized, externalInputs, externalIndexByOldId,
                                                                   steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroupBounded(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                         externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
        }
        steps.push_back(FusedStep{n->op, operands});
        Value v;
        if (steps.size() == 1) {
            if (n->op == OpKind::ReLU) v = out.addUnary(OpKind::ReLU, externalInputs[0], n->debugName);
            else v = out.addBinary(n->op, externalInputs[0], externalInputs[1], n->debugName);
        } else {
            v = out.addFusedElementwise(externalInputs, steps, n->debugName);
        }
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

// ==================== Section 16.1's own BoundaryReason, extended with ONE more reason ====================

enum class BoundaryReason { NotABoundary, Leaf, GraphRoot, Shared, Reduction, SizeCap };

static std::string boundaryReasonStr(BoundaryReason r) {
    switch (r) {
        case BoundaryReason::NotABoundary: return "NotABoundary";
        case BoundaryReason::Leaf:          return "Leaf";
        case BoundaryReason::GraphRoot:     return "GraphRoot";
        case BoundaryReason::Shared:        return "Shared";
        case BoundaryReason::Reduction:     return "Reduction";
        default:                            return "SizeCap";
    }
}
static std::vector<BoundaryReason> classifyNode(const Node* n, const std::map<int, int>& consumers,
                                                 const std::set<int>& sizeCapBoundaries) {
    std::vector<BoundaryReason> reasons;
    if (n->op == OpKind::Input || n->op == OpKind::Const) {
        reasons.push_back(BoundaryReason::Leaf);
        return reasons;
    }
    if (n->op == OpKind::Sum) reasons.push_back(BoundaryReason::Reduction);
    int c = consumers.at(n->id);
    if (c == 0) reasons.push_back(BoundaryReason::GraphRoot);
    else if (c >= 2) reasons.push_back(BoundaryReason::Shared);
    if (sizeCapBoundaries.count(n->id) > 0) reasons.push_back(BoundaryReason::SizeCap);
    if (reasons.empty()) reasons.push_back(BoundaryReason::NotABoundary);
    return reasons;
}
static std::string reasonsStr(const std::vector<BoundaryReason>& reasons) {
    std::string out;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i) out += "+";
        out += boundaryReasonStr(reasons[i]);
    }
    return out;
}

// ==================== Chapter 8's TransformPass / runPasses() (unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;
struct NamedPass { std::string name; TransformPass pass; };

static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass (%zu nodes) ---\n", initial.size());
    printGraph(initial);
    Graph current = passes[0].pass(initial);
    {
        TopoResult topo = topologicalSort(current);
        if (!topo.ok) throw std::runtime_error("runPasses: pass '" + passes[0].name + "' produced a cyclic graph");
    }
    printf("\n--- after pass '%s' (%zu nodes) ---\n", passes[0].name.c_str(), current.size());
    printGraph(current);
    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& np = passes[i];
        Graph next = np.pass(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) throw std::runtime_error("runPasses: pass '" + np.name + "' produced a cyclic graph");
        printf("\n--- after pass '%s' (%zu nodes) ---\n", np.name.c_str(), next.size());
        printGraph(next);
        current = std::move(next);
    }
    return current;
}

// ==================== bytesMoved() (from Chapter 13/14, unchanged) ====================

static constexpr double kBytesPerElement = 4.0;
static long long bytesMoved(const Graph& g, const std::map<int, long long>& elementCounts) {
    long long totalElements = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        for (const Value& in : n->inputs) totalElements += elementCounts.at(in.nodeId);
        totalElements += elementCounts.at(n->id);
    }
    return static_cast<long long>(static_cast<double>(totalElements) * kBytesPerElement);
}

int main() {
    printf("=== Section 16.3: one graph, every boundary this Part has ever built, at once ===\n\n");

    // a=[8], b=scalar. t1..t4: a single-consumer elementwise chain long
    // enough (depth 4) to trigger a size cap of 3. t5: SHARED (2 consumers:
    // s and y2). s: Sum(t5) -- REDUCTION. y2: Mul(t5,b), single consumer.
    // y: Add(s,y2) -- the graph's own designated output, GraphRoot (0
    // consumers). Hand-derivation for maxChainLength=3: chainDepth(t1..t4)
    // = 1,2,3,4 -- only t3 (depth 3) is a multiple of 3, so t3 is the ONE
    // new SizeCap boundary; t4 is not (depth 4, not a multiple of 3), so it
    // inlines into t5's own group instead.
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addUnary(OpKind::ReLU, a, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, b, "t2");
    Value t3 = g.addBinary(OpKind::Add, t2, b, "t3");
    Value t4 = g.addBinary(OpKind::Mul, t3, b, "t4");
    Value t5 = g.addBinary(OpKind::Add, t4, b, "t5");
    Value s  = g.addUnary(OpKind::Sum, t5, "s");
    Value y2 = g.addBinary(OpKind::Mul, t5, b, "y2");
    Value y  = g.addBinary(OpKind::Add, s, y2, "y");
    (void)t3; (void)y;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{8}}}, {b.nodeId, Shape{}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    // ---- The fusion report: classifyNode() on the ORIGINAL graph ----
    printf("--- Fusion report (Section 16.1's classifyNode(), extended with SizeCap) ---\n\n");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    std::set<int> sizeCapBoundaries = computeSizeCapBoundaries(g, consumers, 3);
    for (const auto& n : g.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumers, sizeCapBoundaries);
        printf("  %%%d %-4s consumers=%d  reasons=[%s]\n", n->id, n->debugName.c_str(),
               consumers.at(n->id), reasonsStr(reasons).c_str());
    }
    bool t3IsSizeCap = false, t4IsNotSizeCap = true;
    for (const auto& n : g.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumers, sizeCapBoundaries);
        if (n->debugName == "t3")
            t3IsSizeCap = std::find(reasons.begin(), reasons.end(), BoundaryReason::SizeCap) != reasons.end();
        if (n->debugName == "t4" && std::find(reasons.begin(), reasons.end(), BoundaryReason::SizeCap) != reasons.end())
            t4IsNotSizeCap = false;
    }
    printf("\nself-check: t3 alone carries SizeCap (chainDepth 3, a multiple of maxChainLength=3);\n");
    printf("t4 (chainDepth 4) does not (%s)\n", (t3IsSizeCap && t4IsNotSizeCap) ? "confirmed" : "MISMATCH");

    // ---- UNFUSED bytes ----
    long long unfusedBytes = bytesMoved(g, elementCounts);

    // ---- UNCAPPED fusion (maxChainLength effectively disabled: pass a huge cap) ----
    printf("\n--- UNCAPPED (boundedReductionFusionPass with no effective size limit) ---\n\n");
    TransformPass uncappedTransform = [elementCounts](const Graph& gr) {
        return boundedReductionFusionPass(gr, elementCounts, 1000000).graph;
    };
    Graph uncapped = runPasses(g, {{"boundedReductionFusionPass(uncapped)", uncappedTransform}});
    bool uncappedStructureOk = (uncapped.size() == 5);
    printf("\nself-check: uncapped fusion reaches the same structure Chapter 14's own logic would\n");
    printf("produce -- t1..t5 all fuse into ONE FusedElementwise (5 steps), s is its own\n");
    printf("FusedReduction, y2/y fuse into one more FusedElementwise -- 10 -> 5 nodes (%s)\n",
           uncappedStructureOk ? "confirmed" : "MISMATCH");

    // ---- CAPPED fusion (maxChainLength=3) ----
    printf("\n--- CAPPED (maxChainLength=3) ---\n\n");
    TransformPass cappedTransform = [elementCounts](const Graph& gr) {
        return boundedReductionFusionPass(gr, elementCounts, 3).graph;
    };
    Graph capped = runPasses(g, {{"boundedReductionFusionPass(cap=3)", cappedTransform}});
    bool cappedStructureOk = (capped.size() == 6);
    printf("\nself-check: capped fusion introduces exactly ONE extra materialization point (t3),\n");
    printf("10 -> 6 nodes instead of 10 -> 5 (%s)\n", cappedStructureOk ? "confirmed" : "MISMATCH");

    // ---- Bytes moved: all three, side by side ----
    printf("\n=== Bytes moved: unfused vs. uncapped fusion vs. capped fusion ===\n\n");

    FusionResult uncappedResult = boundedReductionFusionPass(g, elementCounts, 1000000);
    FusionResult cappedResult = boundedReductionFusionPass(g, elementCounts, 3);
    std::map<int, long long> uncappedElementCounts, cappedElementCounts;
    for (const auto& n : uncappedResult.graph.nodes())
        uncappedElementCounts[n->id] = elementCounts.at(uncappedResult.representativeOldId.at(n->id));
    for (const auto& n : cappedResult.graph.nodes())
        cappedElementCounts[n->id] = elementCounts.at(cappedResult.representativeOldId.at(n->id));

    long long uncappedBytes = bytesMoved(uncappedResult.graph, uncappedElementCounts);
    long long cappedBytes = bytesMoved(cappedResult.graph, cappedElementCounts);

    printf("Bytes moved, UNFUSED (8 separate kernels):        %lld\n", unfusedBytes);
    printf("Bytes moved, UNCAPPED fusion (3 fused kernels):   %lld\n", uncappedBytes);
    printf("Bytes moved, CAPPED fusion, maxChainLength=3\n");
    printf("             (4 fused kernels):                  %lld\n", cappedBytes);

    bool bytesOrderingOk = (uncappedBytes < cappedBytes) && (cappedBytes < unfusedBytes);
    double uncappedReduction = 1.0 - static_cast<double>(uncappedBytes) / static_cast<double>(unfusedBytes);
    double cappedReduction = 1.0 - static_cast<double>(cappedBytes) / static_cast<double>(unfusedBytes);
    printf("\nuncapped fusion: %.1f%% fewer bytes than unfused. capped fusion: %.1f%% fewer bytes\n",
           uncappedReduction * 100.0, cappedReduction * 100.0);
    printf("than unfused -- still a large real saving, just not the maximum this graph's own\n");
    printf("structure would otherwise allow.\n");
    printf("\nself-check: uncapped < capped < unfused, strictly, on real measured bytes (%s)\n",
           bytesOrderingOk ? "confirmed" : "MISMATCH");

    // ---- Correctness: evaluate() agreement across all three ----
    printf("\n=== Correctness: evaluate() agreement across original, uncapped, and capped ===\n\n");
    std::map<std::string, float> in = {{"a", 2.0f}, {"b", 3.0f}};
    float origOut = evaluate(g, in, elementCounts).at("y");
    float uncappedOut = evaluate(uncapped, in, {}).at("y");
    float cappedOut = evaluate(capped, in, {}).at("y");
    bool sameAnswer = (origOut == uncappedOut) && (origOut == cappedOut);
    printf("evaluate(original, a=2,b=3).y = %g\n", origOut);
    printf("evaluate(uncapped, ...).y     = %g\n", uncappedOut);
    printf("evaluate(capped, ...).y       = %g\n", cappedOut);
    printf("self-check: all three agree (%s)\n", sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = t3IsSizeCap && t4IsNotSizeCap && uncappedStructureOk && cappedStructureOk &&
                 bytesOrderingOk && sameAnswer;
    return allOk ? 0 : 1;
}
