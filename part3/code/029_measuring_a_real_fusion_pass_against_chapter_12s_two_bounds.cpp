// Chapter 13: Elementwise Fusion
// 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds.cpp
//
// Section 13.3 -- closes the chapter two ways: plugging
// elementwiseFusionPass() (Sections 13.1/13.2, reused verbatim) into
// Chapter 8's own unmodified runPasses(), and measuring REAL bytes moved
// on the diamond graph both before and after the pass actually runs,
// against the two bounds Chapter 12 computed by formula (Section 12.3's
// File 026): 496 bytes for the fully unfused graph, and an IDEALIZED
// 112 bytes for a single kernel covering the whole graph at once --
// a bound Chapter 12 never had to justify was achievable, because it
// never ran an actual correctness-preserving pass to produce it.
//
// elementwiseFusionPass() returns a FusionResult, not a bare Graph --
// its representativeOldId side table is genuinely useful (this section
// uses it directly, to carry Chapter 6's inferShapes() results across
// the pass), but it is ANALYSIS information, not the transformation
// itself, the same distinction Chapter 2 drew between a pass that reads
// an IR and a pass that produces a new one. A one-line wrapper adapts
// the pass to Chapter 8's exact TransformPass contract -- Graph(const
// Graph&) -- for use inside runPasses(), which only ever needs the new
// graph itself.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds.cpp -o 029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds
// Run:     ./029_measuring_a_real_fusion_pass_against_chapter_12s_two_bounds
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Files 027/028, unchanged) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU, FusedElementwise };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input:            return "Input";
        case OpKind::Const:            return "Const";
        case OpKind::Add:              return "Add";
        case OpKind::Mul:              return "Mul";
        case OpKind::ReLU:             return "ReLU";
        default:                       return "FusedElementwise";
    }
}

enum class OperandKind { ExternalInput, PriorStep };
struct FusedOperand { OperandKind kind; int index; };
struct FusedStep { OpKind op; std::vector<FusedOperand> operands; };

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
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) || (consumers.at(oldId) != 1);
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

static FusionResult elementwiseFusionPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("elementwiseFusionPass: graph is not acyclic");
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

// TransformPass-compatible wrapper -- Chapter 8's PassManager only ever
// needs the new Graph itself; the representativeOldId side table is
// extra analysis this chapter's own measurement code below asks for
// directly, calling elementwiseFusionPass() (not this wrapper).
static Graph elementwiseFusionTransform(const Graph& g) { return elementwiseFusionPass(g).graph; }

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise) {
            printf("  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            if (n->op == OpKind::Const) printf("%g", n->constValue);
            else {
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    if (i) printf(", ");
                    printf("%%%d", n->inputs[i].nodeId);
                }
            }
            printf(")\n");
        } else {
            printf("  %%%d %s = FusedElementwise(external inputs: ", n->id, n->debugName.c_str());
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) printf(", ");
                printf("ext%zu=%%%d", i, n->inputs[i].nodeId);
            }
            printf(")\n");
            for (size_t s = 0; s < n->fusedSteps.size(); ++s) {
                const FusedStep& step = n->fusedSteps[s];
                printf("      step%zu = %s(", s, opKindStr(step.op).c_str());
                for (size_t i = 0; i < step.operands.size(); ++i) {
                    if (i) printf(", ");
                    const FusedOperand& o = step.operands[i];
                    printf("%s%d", o.kind == OperandKind::ExternalInput ? "ext" : "step", o.index);
                }
                printf(")%s\n", (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output" : "");
            }
        }
    }
}

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName) {
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
                else sv = read(step.operands[0]) * read(step.operands[1]);
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== Shape / broadcastShapes / inferShapes (from Chapter 6, unchanged) ====================

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
        } else {
            shapes[id] = shapes.at(n->inputs[0].nodeId);  // ReLU -- shape-preserving
        }
    }
    return shapes;
}

// ==================== Chapter 8's TransformPass / runPasses() (unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;

struct NamedPass {
    std::string name;
    TransformPass pass;
};

static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass (%zu nodes) ---\n", initial.size());
    printGraph(initial);

    // Graph deliberately has no copy constructor (it owns its nodes
    // through unique_ptr, exactly like Chapter 4's own Graph) -- so the
    // FIRST pass reads `initial` directly by const reference, and every
    // pass after that reads the previous pass's own returned Graph by
    // reference, moved into `current` only once it has already been
    // validated. Same discipline as Chapter 8's own runPasses().
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

// ==================== Section 13.3: real bytes moved, before vs. after ====================
//
// Uniform for ANY graph this chapter's IR can build: a leaf (Input/Const)
// moves nothing; every other node -- a plain Add/Mul/ReLU OR a
// FusedElementwise -- reads every one of its own `inputs` (its EXTERNAL
// reads) from memory and writes its own single output back. A
// FusedElementwise node's internal steps never appear in this count at
// all -- exactly the point of building one.
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
    printf("=== Section 13.3: elementwiseFusionPass() through Chapter 8's own runPasses() ===\n\n");

    Graph diamond;
    Value a  = diamond.addInput("a");
    Value b  = diamond.addInput("b");
    Value t1 = diamond.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = diamond.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = diamond.addUnary(OpKind::ReLU, t1, "t3");
    Value out = diamond.addBinary(OpKind::Add, t2, t3, "out");
    (void)out;

    Graph afterPasses = runPasses(diamond, {{"elementwiseFusionPass", elementwiseFusionTransform}});

    bool passManagerAccepted = (afterPasses.size() == 4);
    printf("\nself-check: runPasses() accepted elementwiseFusionTransform with ZERO changes to\n");
    printf("Chapter 8's own PassManager -- the fifth real OPTIMIZATION pass (after constant\n");
    printf("folding, dead code elimination, CSE, and algebraic simplification) to plug into\n");
    printf("this same infrastructure unmodified (%s)\n", passManagerAccepted ? "confirmed" : "MISMATCH");

    printf("\n=== Real bytes moved: before vs. after, against Chapter 12's two bounds ===\n\n");

    std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> originalShapes = inferShapes(diamond, declared);
    std::map<int, long long> originalElementCounts;
    for (const auto& kv : originalShapes) originalElementCounts[kv.first] = numElements(kv.second);

    FusionResult fusionResult = elementwiseFusionPass(diamond);
    const Graph& fused = fusionResult.graph;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fusionResult.representativeOldId.at(n->id);
        fusedElementCounts[n->id] = originalElementCounts.at(oldId);
    }

    long long bytesBefore = bytesMoved(diamond, originalElementCounts);
    long long bytesAfter = bytesMoved(fused, fusedElementCounts);

    printf("Bytes moved, BEFORE fusion (4 separate kernels -- t1, t2, t3, out):     %lld\n", bytesBefore);
    printf("Bytes moved, AFTER elementwiseFusionPass() (t1 stays a boundary,\n");
    printf("             t2/t3/out fuse into one kernel):                          %lld\n", bytesAfter);
    printf("\nChapter 12's own two bounds for this exact graph (Section 12.3, computed by\n");
    printf("formula, never by an actual pass):\n");
    printf("  fully UNFUSED (4 kernels):                          496 bytes  (matches 'BEFORE' above)\n");
    printf("  IDEALIZED single kernel for the WHOLE graph at once: 112 bytes  (never actually built)\n");

    bool beforeMatchesCh12Unfused = (bytesBefore == 496);
    bool afterIsBetweenTheBounds = (bytesAfter > 112 && bytesAfter < 496);
    printf("\nself-check: bytes moved before fusion exactly matches Chapter 12's own 496-byte\n");
    printf("unfused figure for this graph (%s)\n", beforeMatchesCh12Unfused ? "confirmed" : "MISMATCH");
    printf("self-check: bytes moved after this REAL pass (%lld) is strictly BETWEEN Chapter 12's\n", bytesAfter);
    printf("idealized 112-byte bound and its 496-byte unfused figure -- real, substantial\n");
    printf("savings, honestly short of an ideal that never had to respect t1's own sharing (%s)\n",
           afterIsBetweenTheBounds ? "confirmed" : "MISMATCH");

    double flops = 48.0;  // identical either way -- computed by hand from Chapter 12's own File 026
    double aiBefore = flops / static_cast<double>(bytesBefore);
    double aiAfter = flops / static_cast<double>(bytesAfter);
    printf("\nAI_before = 48 / %lld bytes = %.6f FLOPs/byte\n", bytesBefore, aiBefore);
    printf("AI_after  = 48 / %lld bytes = %.6f FLOPs/byte\n", bytesAfter, aiAfter);
    bool aiImproved = aiAfter > aiBefore;
    printf("self-check: arithmetic intensity strictly improved from a REAL pass run, not just a\n");
    printf("formula (%s)\n", aiImproved ? "confirmed" : "MISMATCH");

    std::map<std::string, float> in = {{"a", 4.0f}, {"b", 9.0f}};
    float origOut = evaluate(diamond, in).at("out");
    float fusedOut = evaluate(fused, in).at("out");
    bool sameAnswer = (origOut == fusedOut);
    printf("\nself-check: evaluate(original, a=4,b=9).out = %g, evaluate(fused, ...).out = %g (%s)\n",
           origOut, fusedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = passManagerAccepted && beforeMatchesCh12Unfused && afterIsBetweenTheBounds &&
                 aiImproved && sameAnswer;
    return allOk ? 0 : 1;
}
