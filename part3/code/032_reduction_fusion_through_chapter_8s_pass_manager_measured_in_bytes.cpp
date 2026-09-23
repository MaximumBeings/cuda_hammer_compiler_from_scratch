// Chapter 14: Reduction Fusion
// 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes.cpp
//
// Section 14.3 -- closes the chapter by plugging reductionFusionPass()
// into Chapter 8's own unmodified runPasses(), and measuring REAL bytes
// moved, before and after, on a clean chain that ends in a reduction:
// a=Input([3,4]), b=Input([4]), t1=Add(a,b), t2=ReLU(t1), s=Sum(t2) --
// no sharing anywhere in this graph, so nothing stops it from fusing all
// the way down to one kernel.
//
// One real wrinkle Chapter 13's own elementwiseFusionTransform() never
// had to face: reductionFusionPass() needs elementCounts (to bake
// reduceElementCount into each Sum step, Section 14.1), but Chapter 8's
// TransformPass contract is exactly Graph(const Graph&) -- no room for a
// second argument. This section's own wrapper captures a specific
// graph's precomputed element counts in a closure instead of changing
// that contract, a real, concrete instance of a pass gaining a genuine
// new external dependency Chapter 8's interface was never designed to
// carry.
//
// This section also has to fix Chapter 12's own totalFlops() convention
// ("one FLOP per output element"), which quietly assumed every op's
// element-for-element correspondence -- exactly true for Add/Mul/ReLU,
// but wrong for Sum: reducing N elements to 1 takes N-1 additions, not 1.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes.cpp -o 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes
// Run:     ./032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from File 030/031, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };

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
struct FusedOperand { OperandKind kind; int index; };
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
                if (in.nodeId == id)
                    if (--inDegree[n->id] == 0) ready.push_back(n->id);
    }
    result.ok = (result.order.size() == g.size());
    return result;
}

// ==================== Shape / broadcastShapes / inferShapes (from File 030/031, unchanged) ====================

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
        if (n->op == OpKind::Input || n->op == OpKind::Const) shapes[id] = declaredShapes.at(id);
        else if (n->op == OpKind::Add || n->op == OpKind::Mul)
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        else if (n->op == OpKind::ReLU) shapes[id] = shapes.at(n->inputs[0].nodeId);
        else shapes[id] = Shape{};  // Sum
    }
    return shapes;
}

// ==================== resolveIntoGroup() / reductionFusionPass() (from File 030/031, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized,
                                      std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId,
                                      std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
                           (consumers.at(oldId) != 1);
    if (mustBeExternal) {
        auto extIt = externalIndexByOldId.find(oldId);
        if (extIt != externalIndexByOldId.end()) return FusedOperand{OperandKind::ExternalInput, extIt->second};
        int idx = static_cast<int>(externalInputs.size());
        externalInputs.push_back(materialized.at(oldId));
        externalIndexByOldId[oldId] = idx;
        return FusedOperand{OperandKind::ExternalInput, idx};
    }
    std::vector<FusedOperand> operands;
    for (const Value& in : p->inputs) {
        operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                             externalIndexByOldId, steps, stepIndexByOldId));
    }
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}

static FusionResult reductionFusionPass(const Graph& g, const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("reductionFusionPass: graph is not acyclic");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;

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
            FusedOperand summedOperand = resolveIntoGroup(n->inputs[0].nodeId, g, consumers, materialized,
                                                            externalInputs, externalIndexByOldId,
                                                            steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
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

// ==================== printGraph() / evaluate() (from File 030/031, unchanged) ====================

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
        if (n->op == OpKind::Input) v = inputValuesByName.at(n->debugName);
        else if (n->op == OpKind::Const) v = n->constValue;
        else if (n->op == OpKind::Add) v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::Mul) v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::ReLU) v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        else if (n->op == OpKind::Sum) v = valuesById.at(n->inputs[0].nodeId) * static_cast<float>(elementCounts.at(n->inputs[0].nodeId));
        else {
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
                else sv = read(step.operands[0]) * static_cast<float>(step.reduceElementCount);
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
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
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + passes[0].name + "' produced a graph that is not acyclic");
        }
    }
    printf("\n--- after pass '%s' (%zu nodes) ---\n", passes[0].name.c_str(), current.size());
    printGraph(current);

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& np = passes[i];
        Graph next = np.pass(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + np.name + "' produced a graph that is not acyclic");
        }
        printf("\n--- after pass '%s' (%zu nodes) ---\n", np.name.c_str(), next.size());
        printGraph(next);
        current = std::move(next);
    }
    return current;
}

// ==================== Section 14.3: bytes moved (from Chapter 13's File 029, unchanged) ====================
//
// Works uniformly for ANY node this chapter's IR can build -- including
// FusedReduction -- with ZERO changes needed: a leaf moves nothing; every
// other node reads its own `inputs` (external reads) and writes its own
// single output. A FusedReduction's own output element count is exactly
// 1 (Section 14.1's inferShapes() extension: Sum -> Shape{}), which this
// function picks up automatically through the SAME elementCounts/
// representativeOldId side channel Chapter 13 already built -- Section
// 14.1's one-line shape extension is the ENTIRE reason Chapter 13's own
// byte-counting apparatus needs no changes at all to handle reduction
// fusion too.
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

// ==================== Section 14.3: FLOPs, corrected for a reduction ====================
//
// Chapter 12's own totalFlops() used "one FLOP per output element" -- an
// exact fit for Add/Mul/ReLU, where output element count IS operation
// count. Sum breaks that: reducing N elements to 1 takes N-1 additions,
// not 1. Using the old convention unmodified would claim summing a
// million elements costs a single FLOP -- silently, not even an error.
// This function fixes it with one new branch, on the ORIGINAL (unfused)
// graph only -- total FLOPs is identical whether fused or not, the same
// claim Chapters 12 and 13 already made; fusion changes WHERE and HOW,
// never HOW MUCH.
static long long totalFlops(const Graph& g, const std::map<int, Shape>& shapes) {
    long long total = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        if (n->op == OpKind::Sum) {
            total += numElements(shapes.at(n->inputs[0].nodeId)) - 1;  // N elements -> N-1 additions
        } else {
            total += numElements(shapes.at(n->id));  // Add/Mul/ReLU: one FLOP per output element
        }
    }
    return total;
}

int main() {
    printf("=== Section 14.3: reductionFusionPass() through Chapter 8's own runPasses() ===\n\n");

    // a=[3,4], b=[4] (broadcast); t1=Add(a,b); t2=ReLU(t1); s=Sum(t2).
    // NO sharing anywhere -- every intermediate node has exactly one
    // consumer, so nothing stops this chain from fusing all the way down.
    Graph chain;
    Value a = chain.addInput("a");
    Value b = chain.addInput("b");
    Value t1 = chain.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = chain.addUnary(OpKind::ReLU, t1, "t2");
    Value s = chain.addUnary(OpKind::Sum, t2, "s");
    (void)s;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> originalShapes = inferShapes(chain, declared);
    std::map<int, long long> originalElementCounts;
    for (const auto& kv : originalShapes) originalElementCounts[kv.first] = numElements(kv.second);

    // Chapter 8's TransformPass contract is exactly Graph(const Graph&) --
    // no room for reductionFusionPass()'s own second argument. The fix is
    // a closure that captures this SPECIFIC graph's own precomputed
    // element counts, rather than changing that contract.
    TransformPass reductionFusionTransform = [originalElementCounts](const Graph& g) {
        return reductionFusionPass(g, originalElementCounts).graph;
    };

    Graph afterPasses = runPasses(chain, {{"reductionFusionPass", reductionFusionTransform}});

    bool passManagerAccepted = (afterPasses.size() == 3);
    printf("\nself-check: runPasses() accepted reductionFusionTransform with ZERO changes to\n");
    printf("Chapter 8's own PassManager interface -- the closure carries the shape dependency\n");
    printf("reductionFusionPass() needs, so TransformPass itself stays exactly Graph(const Graph&) (%s)\n",
           passManagerAccepted ? "confirmed" : "MISMATCH");

    printf("\n=== Real bytes moved: before vs. after ===\n\n");

    FusionResult fusionResult = reductionFusionPass(chain, originalElementCounts);
    const Graph& fused = fusionResult.graph;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fusionResult.representativeOldId.at(n->id);
        fusedElementCounts[n->id] = originalElementCounts.at(oldId);
    }

    long long bytesBefore = bytesMoved(chain, originalElementCounts);
    long long bytesAfter = bytesMoved(fused, fusedElementCounts);

    printf("Bytes moved, BEFORE fusion (3 separate kernels -- t1, t2, s):   %lld\n", bytesBefore);
    printf("Bytes moved, AFTER reductionFusionPass() (t1/t2/s fuse into\n");
    printf("             ONE kernel -- no sharing anywhere in this chain): %lld\n", bytesAfter);

    bool realSavings = (bytesAfter < bytesBefore);
    double reduction = 1.0 - static_cast<double>(bytesAfter) / static_cast<double>(bytesBefore);
    printf("\nself-check: bytes moved strictly decreased (%lld -> %lld, a %.1f%% reduction) (%s)\n",
           bytesBefore, bytesAfter, reduction * 100.0, realSavings ? "confirmed" : "MISMATCH");

    long long flops = totalFlops(chain, originalShapes);
    double aiBefore = static_cast<double>(flops) / static_cast<double>(bytesBefore);
    double aiAfter = static_cast<double>(flops) / static_cast<double>(bytesAfter);
    static constexpr double kRidgePointFlopsPerByte = 12.5402;  // same A100 figure as Chapter 12's own 12.2
    printf("\nTotal FLOPs (identical either way -- fusion doesn't change the arithmetic, only where\n");
    printf("it happens; Sum's own N-1-additions correction from this section is what makes this\n");
    printf("number honest): %lld\n", flops);
    printf("AI_before = %lld / %lld bytes = %.6f FLOPs/byte\n", flops, bytesBefore, aiBefore);
    printf("AI_after  = %lld / %lld bytes = %.6f FLOPs/byte\n", flops, bytesAfter, aiAfter);
    printf("(ridge point, same A100 numbers as Chapter 12's own Section 12.2: %.4f FLOPs/byte --\n",
           kRidgePointFlopsPerByte);
    printf(" still memory-bound on both sides at this toy graph's tiny scale, consistent with\n");
    printf(" Chapter 12's own scale caveat)\n");
    bool aiImproved = aiAfter > aiBefore;

    printf("\n=== Correctness: evaluate() agreement, before vs. after ===\n\n");
    std::map<std::string, float> in = {{"a", 2.0f}, {"b", 3.0f}};
    float origOut = evaluate(chain, in, originalElementCounts).at("s");
    float fusedOut = evaluate(fused, in, {}).at("s");
    bool sameAnswer = (origOut == fusedOut);
    printf("self-check: evaluate(original, a=2,b=3).s = %g, evaluate(fused, ...).s = %g (%s)\n",
           origOut, fusedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = passManagerAccepted && realSavings && aiImproved && sameAnswer;
    return allOk ? 0 : 1;
}
