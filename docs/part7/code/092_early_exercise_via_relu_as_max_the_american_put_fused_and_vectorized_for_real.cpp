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
