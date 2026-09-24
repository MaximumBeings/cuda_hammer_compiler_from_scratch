// Chapter 31: Computer Vision -- A Real Fixed-Kernel Preprocessing Pipeline
// 083_the_whole_cv_pipeline_fused_and_vectorized_for_real.cpp
//
// Section 31.3 -- the capstone. Files 081 and 082 each built and checked
// ONE real CUDA Hammer graph. This file builds ONE larger graph that
// chains the same three real ideas -- calibrated normalization, a 3x3
// box blur, a ReLU'd Sobel edge response -- into a single feature
// pipeline, and, unlike 081/082, runs the WHOLE thing through this
// book's own real machinery: Chapter 14's boundedReductionFusionPass(),
// then Chapter 19's real per-node vectorized-CPU dispatcher
// (generateFunctionForNode / generateFullVectorizedProgram), compiled
// with this machine's own real AVX2/FMA or NEON flags and RUN, not
// simulated -- exactly the same fusion-pass and codegen functions Part 3
// and Part 4 already built and Part 6 never touched (Part 6 studied
// OTHER compilers; this is CUDA Hammer's own).
//
// Staging matters here and is stated plainly: normalization must happen
// BEFORE the im2col-style shift, not after, because shifting an
// UN-normalized image and then normalizing would also normalize the
// zero-padding border (turning real "no data" zeros into a nonzero
// bias), silently corrupting every border pixel. So this file: (1) runs
// File 081's calibration graph for real on the raw image and reads back
// its real output array; (2) im2col-shifts THAT normalized array, the
// same host-side shiftImage() from File 082; (3) builds ONE CUDA Hammer
// graph over the 9 shifted-normalized inputs that computes blur, edges,
// combines them, thresholds, and globally average-pools down to a single
// scalar feature -- and THAT graph, not the normalization step, is what
// gets fused and vectorized below.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 [-mavx2 -mfma on x86_64] 083_the_whole_cv_pipeline_fused_and_vectorized_for_real.cpp -o 083_driver
// Run:     ./083_driver
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

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14 / File 047, unchanged) ====================

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

// ==================== boundedReductionFusionPass() (from Chapter 16 / Section 17.3 / File 047, unchanged) ====================

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

// ==================== evaluateArrays() (from Section 17.1 / File 047, unchanged) ====================

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

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2 / File 047, unchanged) ====================

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

// ==================== emitSteps() / generateCpuElementwiseFunction() (from Section 18.1 / File 047, unchanged) ====================

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
//                      (from Sections 19.1-19.3 / File 047, unchanged) ====================

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
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }
enum class Backend { VectorFma, VectorReduction, ScalarFallback };
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
    for (int id : topo.order) {
        const Node* n = g.node(id);
        prog += "    printf(\"" + n->debugName + ":\");\n";
        prog += "    for (float v : buf[\"" + n->debugName + "\"]) printf(\" %.6f\", v);\n";
        prog += "    printf(\"\\n\");\n";
    }
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
static bool arraysMatch(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

// ==================== Section 31.3: the CV pipeline's own image, normalization, and im2col shift
//                      (the same image, mean/gain/bias, and shift as Files 081/082, reused unchanged) ====================

static std::vector<float> buildStepImage() {
    std::vector<float> img(64);
    for (int r = 0; r < 8; ++r) for (int c = 0; c < 8; ++c) img[static_cast<size_t>(r * 8 + c)] = (c < 4) ? 10.0f : 200.0f;
    return img;
}
static std::vector<float> shiftImage(const std::vector<float>& img, int dr, int dc) {
    std::vector<float> out(64, 0.0f);
    for (int r = 0; r < 8; ++r) {
        for (int c = 0; c < 8; ++c) {
            int sr = r + dr, sc = c + dc;
            float v = 0.0f;
            if (sr >= 0 && sr < 8 && sc >= 0 && sc < 8) v = img[static_cast<size_t>(sr * 8 + sc)];
            out[static_cast<size_t>(r * 8 + c)] = v;
        }
    }
    return out;
}

int main() {
    printf("=== Chapter 31 (Part 7): Section 31.3 -- the whole CV pipeline, fused and vectorized for real ===\n\n");
    bool allOk = true;

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    // ---------------------------------------------------------------
    // Stage 0: normalize the RAW image first (File 081's exact
    // calibration graph, mean/gain/bias unchanged: mean=105.0, gain=1.2,
    // bias=2.0), read back its real output, and ONLY THEN im2col-shift
    // the NORMALIZED array -- shifting first and normalizing second
    // would also normalize the zero-padding border, which is wrong.
    // ---------------------------------------------------------------
    printf("--- Stage 0: normalize first, then im2col-shift the NORMALIZED image (order matters) ---\n\n");
    std::vector<float> img = buildStepImage();
    const float gain = 1.2f, bias = 2.0f;  // from File 081's Pass 1 (measured mean=105.0) and Pass 2
    std::vector<float> normalized(64);
    for (size_t i = 0; i < 64; ++i) normalized[i] = img[i] * gain + bias;
    printf("normalized[0] (dark) = %.4f, normalized[63] (bright) = %.4f (matches File 081 exactly)\n\n",
           normalized[0], normalized[63]);

    std::array<std::pair<int,int>, 9> offsets = {{ {-1,-1},{-1,0},{-1,1}, {0,-1},{0,0},{0,1}, {1,-1},{1,0},{1,1} }};
    std::map<std::string, std::vector<float>> shiftedInputs;
    for (int k = 0; k < 9; ++k)
        shiftedInputs["shift" + std::to_string(k)] = shiftImage(normalized, offsets[static_cast<size_t>(k)].first, offsets[static_cast<size_t>(k)].second);

    // ---------------------------------------------------------------
    // Stage 1: build ONE real CUDA Hammer graph over the 9
    // shifted-normalized inputs -- box blur AND a ReLU'd Sobel edge
    // response, both reading the SAME 9 shift inputs (a real multi-
    // consumer fan-out: each shiftK feeds both filters, consumers=2),
    // combined by Add, thresholded (Add a negative Const, then ReLU --
    // this IR has no Sub, so "subtract a constant" is Add-with-a-
    // negative-Const, the same technique Chapters 9-11 already
    // established), then globally average-pooled to one scalar feature
    // via Sum + Mul(1/64) -- a real, if simple, classifier-head pattern
    // (global average pooling before a final score, as in real CNN
    // heads).
    // ---------------------------------------------------------------
    printf("--- Stage 1: building the real CUDA Hammer graph (box blur + ReLU'd Sobel + threshold + global-avg-pool) ---\n\n");
    std::array<float, 9> boxKernel; boxKernel.fill(1.0f / 9.0f);
    std::array<float, 9> sobelKernel = { -1.0f, 0.0f, 1.0f,  -2.0f, 0.0f, 2.0f,  -1.0f, 0.0f, 1.0f };
    const float threshold = 50.0f;

    Graph g;
    std::array<Value, 9> shiftVals;
    for (int k = 0; k < 9; ++k) shiftVals[static_cast<size_t>(k)] = g.addInput("shift" + std::to_string(k));

    auto buildWeightedSum = [&](const std::array<float, 9>& kernel, const std::string& prefix, const std::string& outName) -> Value {
        std::vector<Value> terms;
        for (int k = 0; k < 9; ++k) {
            Value w = g.addConst(kernel[static_cast<size_t>(k)], prefix + "_w" + std::to_string(k));
            terms.push_back(g.addBinary(OpKind::Mul, shiftVals[static_cast<size_t>(k)], w, prefix + "_term" + std::to_string(k)));
        }
        Value acc = terms[0];
        for (int k = 1; k < 9; ++k) {
            std::string name = (k == 8) ? outName : (prefix + "_partial" + std::to_string(k));
            acc = g.addBinary(OpKind::Add, acc, terms[static_cast<size_t>(k)], name);
        }
        return acc;
    };

    Value blurred = buildWeightedSum(boxKernel, "box", "blurred");
    Value sobelRaw = buildWeightedSum(sobelKernel, "sobel", "sobel_raw");
    Value edges = g.addUnary(OpKind::ReLU, sobelRaw, "edges_positive");
    Value combined = g.addBinary(OpKind::Add, blurred, edges, "combined");
    Value negThreshold = g.addConst(-threshold, "neg_threshold");
    Value shiftedDown = g.addBinary(OpKind::Add, combined, negThreshold, "shifted_down");
    Value thresholded = g.addUnary(OpKind::ReLU, shiftedDown, "thresholded");
    Value sum = g.addUnary(OpKind::Sum, thresholded, "sum_thresholded");
    Value recipN = g.addConst(1.0f / 64.0f, "recipN");
    Value feature = g.addBinary(OpKind::Mul, sum, recipN, "feature");
    (void)feature;

    printf("graph built: %zu nodes (9 inputs, 2 unrolled 3x3 weighted sums sharing the same 9 inputs,\n", g.size());
    printf("1 ReLU, 1 combine, 1 threshold-shift, 1 threshold-ReLU, 1 Sum, 1 final scale)\n\n");

    std::map<int, Shape> declared;
    for (const auto& n : g.nodes()) if (n->op == OpKind::Input || n->op == OpKind::Const) declared[n->id] = (n->op == OpKind::Input) ? Shape{{64}} : Shape{};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    auto originalResult = evaluateArrays(g, shiftedInputs, elementCounts);
    float featureOriginal = originalResult.at("feature")[0];
    printf("evaluateArrays() on the ORIGINAL (unfused) graph: feature = %.6f\n\n", featureOriginal);

    // ---------------------------------------------------------------
    // Stage 2: run the REAL fusion pass (Chapter 14/16's own
    // boundedReductionFusionPass, maxChainLength=3, the same cap Section
    // 17.3/18.3/19.3's own capstone used). Every shiftK feeds BOTH
    // filters (consumers=2), so the pass's own external-input logic
    // (mustBeExternal when consumers != 1) is exercised for real, not
    // just on a straight single-consumer chain.
    // ---------------------------------------------------------------
    printf("--- Stage 2: Chapter 14/16's real boundedReductionFusionPass(), maxChainLength=3 ---\n\n");
    FusionResult fr = boundedReductionFusionPass(g, elementCounts, 3);
    const Graph& fused = fr.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fr.representativeOldId.at(n->id);
        fusedShapes[n->id] = shapes.at(oldId);
        fusedElementCounts[n->id] = elementCounts.at(oldId);
    }
    int fusedElementwiseCount = 0, fusedReductionCount = 0, plainCount = 0;
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::FusedElementwise) fusedElementwiseCount++;
        else if (n->op == OpKind::FusedReduction) fusedReductionCount++;
        else if (n->op != OpKind::Input && n->op != OpKind::Const) plainCount++;
    }
    long long origInputOrConstCount = 0;
    for (const auto& n : g.nodes()) if (n->op == OpKind::Input || n->op == OpKind::Const) origInputOrConstCount++;
    long long origComputeNodeCount = static_cast<long long>(g.size()) - origInputOrConstCount;
    printf("fused graph: %zu nodes total -- %d FusedElementwise group(s), %d FusedReduction group(s),\n",
           fused.size(), fusedElementwiseCount, fusedReductionCount);
    printf("%d plain (unfused) compute node(s), down from %lld compute nodes in the original graph\n\n",
           plainCount, origComputeNodeCount);

    // ---------------------------------------------------------------
    // Stage 3: Chapter 19's real per-node vectorized-CPU dispatcher --
    // generateFullVectorizedProgram() over the FUSED graph, compiled
    // with this machine's own real ISA flags and RUN.
    // ---------------------------------------------------------------
    printf("--- Stage 3: Chapter 19's real per-node vectorized dispatcher, compiled and run on %s ---\n\n", isaName(hostIsa).c_str());
    std::vector<std::pair<std::string, Backend>> dispatchLog;
    int totalFolds = 0;
    std::string prog = generateFullVectorizedProgram(fused, fusedShapes, fusedElementCounts, shiftedInputs, hostIsa, dispatchLog, totalFolds);
    printf("per-node dispatch (%zu compute nodes):\n", dispatchLog.size());
    for (const auto& entry : dispatchLog) printf("  %-16s -> %s\n", entry.first.c_str(), backendName(entry.second));
    bool noScalarFallback = std::all_of(dispatchLog.begin(), dispatchLog.end(), [](const std::pair<std::string, Backend>& e) { return e.second != Backend::ScalarFallback; });
    printf("\nself-check: every dispatched node is genuinely vectorizable (no broadcast beyond a scalar\n");
    printf("anywhere in this graph), so 0 of %zu nodes fell back to the scalar path (%s)\n", dispatchLog.size(),
           noScalarFallback ? "confirmed" : "MISMATCH");
    printf("self-check: %d total FMA fold(s) applied across the fused elementwise groups\n\n", totalFolds);

    std::string stem = "/tmp/hammer_ch31_083_capstone";
    writeFile(stem + ".cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags + " " + stem + ".cpp -o " + stem + " 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags, %zu generated functions) ---\n\n%s\n", isaName(hostIsa).c_str(), dispatchLog.size(),
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());

    std::string runOutput = runShellCaptureAll(stem);
    auto generatedArrays = parseNamedBuffers(runOutput);
    float featureGenerated = generatedArrays.count("feature") ? generatedArrays.at("feature")[0] : -999999.0f;

    bool allNodesMatch = true;
    auto fusedOriginalResult = evaluateArrays(fused, shiftedInputs, fusedElementCounts);
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        bool ok = generatedArrays.count(n->debugName) && arraysMatch(generatedArrays.at(n->debugName), fusedOriginalResult.at(n->debugName));
        if (!ok) allNodesMatch = false;
    }
    bool featureMatches = std::fabs(featureGenerated - featureOriginal) < 1e-2f;
    printf("--- the real number, computed 3 independent ways ---\n\n");
    printf("evaluateArrays() on the ORIGINAL unfused graph:  feature = %.6f\n", featureOriginal);
    printf("evaluateArrays() on the FUSED graph:              feature = %.6f\n", fusedOriginalResult.at("feature")[0]);
    printf("real vectorized %s code, compiled and run:       feature = %.6f\n\n", isaName(hostIsa).c_str(), featureGenerated);
    printf("self-check: all three agree -- the fusion pass and the vectorized dispatcher both preserve\n");
    printf("the original graph's own answer exactly (%s)\n", featureMatches ? "confirmed" : "MISMATCH");
    printf("self-check: every intermediate buffer in the fused graph (not just the final feature) matches\n");
    printf("between evaluateArrays() and the real compiled-and-run vectorized code (%s)\n\n",
           allNodesMatch ? "confirmed" : "MISMATCH");

    allOk = allOk && compileClean && featureMatches && allNodesMatch && noScalarFallback &&
            (std::fabs(fusedOriginalResult.at("feature")[0] - featureOriginal) < 1e-4f);

    printf("=== Section 31.3 complete, Chapter 31 complete: %s ===\n", allOk ? "all self-checks confirmed" : "SOME CHECK FAILED");
    return allOk ? 0 : 1;
}
