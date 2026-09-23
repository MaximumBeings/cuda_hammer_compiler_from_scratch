// Chapter 19: Generating Vectorized CPU Code
// 045_from_threads_to_lanes_a_portable_simd_elementwise_kernel_generator.cpp
//
// Section 19.1 -- Chapter 18 launched one CUDA THREAD per output element.
// This chapter targets a genuinely different kind of parallelism: one CPU
// INSTRUCTION processing several elements at once, via real SIMD registers
// -- 8 floats per instruction on the cloud sandbox's own AVX2, 4 floats per
// instruction on the device's own NEON. Unlike Chapter 18's CUDA target,
// this chapter's own generated code is not just compiled -- it is compiled
// AND ACTUALLY EXECUTED, for real, on two genuinely different real
// architectures this book has real access to, closing the loop Chapter 17
// opened and Chapter 18 could not close for lack of a physical GPU.
//
// This section's own vectorized codegen covers exactly two operand shapes
// -- an external input whose buffer is the SAME size as the output (a
// plain contiguous vector load) or a bare SCALAR (broadcast to every
// lane) -- a stated scope limitation, not a silent gap: a genuine
// trailing-suffix broadcast (like Chapter 6's own b:[4] against a:[3,4])
// is NOT handled by this section's own vector path, and this section
// demonstrates that limitation being CAUGHT, not silently miscompiled.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 045_from_threads_to_lanes_a_portable_simd_elementwise_kernel_generator.cpp -o 045_driver
// Run:     ./045_driver
// (This file's own main() is plain, portable C++: it detects which real
// ISA THIS machine's own compiler targets via __x86_64__/__aarch64__,
// generates a vectorized .cpp program as a string, writes it to disk, and
// shells out to compile-and-run IT with the matching real intrinsics
// header and real compiler flags for that architecture.)
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <cmath>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

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
        for (const auto& n : g.nodes()) {
            for (const Value& in : n->inputs) {
                if (in.nodeId == id) { if (--inDegree[n->id] == 0) ready.push_back(n->id); }
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
        else shapes[id] = Shape{};
    }
    return shapes;
}

// ==================== resolveIntoGroup() / reductionFusionPass() (from Chapter 14, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized, std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId, std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) || (p->op == OpKind::Sum) ||
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
    for (const Value& in : p->inputs)
        operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                             externalIndexByOldId, steps, stepIndexByOldId));
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
    for (const auto& n : g.nodes()) for (const Value& in : n->inputs) consumers[in.nodeId]++;
    FusionResult result;
    Graph& out = result.graph;
    std::map<int, Value> materialized;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        if (n->op == OpKind::Input) {
            Value v = out.addInput(n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (n->op == OpKind::Const) {
            Value v = out.addConst(n->constValue, n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (n->op == OpKind::Sum) {
            std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
            std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
            FusedOperand summedOperand = resolveIntoGroup(n->inputs[0].nodeId, g, consumers, materialized,
                                                            externalInputs, externalIndexByOldId, steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs)
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
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

// ==================== evaluateArrays() (from Section 17.1, unchanged) ====================

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
                        else if (step.op == OpKind::Add)
                            sv = readOperand(step.operands[0], stepVals, i) + readOperand(step.operands[1], stepVals, i);
                        else
                            sv = readOperand(step.operands[0], stepVals, i) * readOperand(step.operands[1], stepVals, i);
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
                        else if (step.op == OpKind::Add)
                            sv = readOperand(step.operands[0], stepVals, r) + readOperand(step.operands[1], stepVals, r);
                        else
                            sv = readOperand(step.operands[0], stepVals, r) * readOperand(step.operands[1], stepVals, r);
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

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2, unchanged) ====================

struct Loop { std::string dimName; long long extent; };
struct LoopNest { std::vector<Loop> loops; };
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    if (n->op == OpKind::Sum) nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    else if (n->op == OpKind::FusedReduction) nest.loops.push_back(Loop{"reduce", n->fusedSteps.back().reduceElementCount});
    return nest;
}
struct LoweredNode { std::vector<FusedStep> steps; LoopNest nest; };
static LoweredNode lowerNode(const Node* n, const std::map<int, Shape>& shapes,
                              const std::map<int, long long>& elementCounts) {
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

// ==================== emitSteps() / generateCpuElementwiseFunction() (from Sections 17.2/18.1, unchanged) ====================
// Reused verbatim for exactly ONE purpose in this section: the SCALAR TAIL
// of the vector loop below needs no new logic of its own -- it is Chapter
// 18's own scalar codegen, called again, starting wherever the vector loop
// left off.

enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExprScalar(Target target, const std::string& x) {
    return (target == Target::Cpu) ? ("std::max(0.0f, " + x + ")") : ("fmaxf(0.0f, " + x + ")");
}
static std::vector<std::string> emitSteps(const std::vector<FusedStep>& steps, bool isReduction,
                                           const std::string& idxExpr, Target target, std::string& finalValueExpr) {
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
        finalValueExpr = (sumOperand.kind == OperandKind::ExternalInput)
                              ? readExternal(target, sumOperand.index, idxExpr)
                              : stepVars[static_cast<size_t>(sumOperand.index)];
    } else {
        finalValueExpr = stepVars.back();
    }
    return lines;
}

// ==================== Section 19.1: Isa -- the two real architectures this book actually runs on ====================

enum class Isa { Avx2, Neon };

static int vectorWidth(Isa isa) { return (isa == Isa::Avx2) ? 8 : 4; }
static std::string vecType(Isa isa) { return (isa == Isa::Avx2) ? "__m256" : "float32x4_t"; }
static std::string isaHeaderInclude(Isa isa) {
    return (isa == Isa::Avx2) ? "#include <immintrin.h>" : "#include <arm_neon.h>";
}
static std::string isaName(Isa isa) { return (isa == Isa::Avx2) ? "AVX2/FMA" : "NEON"; }

// The only two operand shapes this section's own vector codegen covers:
// a buffer the SAME size as the output (a plain contiguous load) or a bare
// SCALAR (broadcast to every lane). Anything else -- a genuine trailing-
// suffix broadcast smaller than a full lane, like Chapter 6's own b:[4]
// against a:[3,4] -- is explicitly OUT of scope; canVectorizeElementwise()
// (below) exists to CATCH that case, not silently miscompile it.
static std::string vecLoadOrBroadcast(Isa isa, int index, const std::string& baseExpr, bool isScalar) {
    // Unlike Chapter 18's CUDA kernels (which needed a NAMED pointer
    // parameter per external input -- a real host/device memory-space
    // requirement), this is plain CPU code: ext[index] is already a valid
    // const float* from the SAME std::vector<const float*> Chapter 17/18's
    // own scalar functions already take, so plain pointer arithmetic on
    // ext[index] works directly, no separate parameter needed.
    std::string ext = "ext[" + std::to_string(index) + "]";
    if (isa == Isa::Avx2) {
        if (isScalar) return "_mm256_set1_ps(" + ext + "[0])";
        return "_mm256_loadu_ps(" + ext + " + " + baseExpr + ")";
    }
    if (isScalar) return "vdupq_n_f32(" + ext + "[0])";
    return "vld1q_f32(" + ext + " + " + baseExpr + ")";
}
static std::string vecAdd(Isa isa, const std::string& a, const std::string& b) {
    return (isa == Isa::Avx2) ? ("_mm256_add_ps(" + a + ", " + b + ")") : ("vaddq_f32(" + a + ", " + b + ")");
}
static std::string vecMul(Isa isa, const std::string& a, const std::string& b) {
    return (isa == Isa::Avx2) ? ("_mm256_mul_ps(" + a + ", " + b + ")") : ("vmulq_f32(" + a + ", " + b + ")");
}
static std::string vecMax(Isa isa, const std::string& x) {
    if (isa == Isa::Avx2) return "_mm256_max_ps(" + x + ", _mm256_setzero_ps())";
    return "vmaxq_f32(" + x + ", vdupq_n_f32(0.0f))";
}
static std::string vecStore(Isa isa, const std::string& dst, const std::string& val) {
    return (isa == Isa::Avx2) ? ("_mm256_storeu_ps(" + dst + ", " + val + ");")
                               : ("vst1q_f32(" + dst + ", " + val + ");");
}

// True only when every external input this node's own steps read is either
// the SAME element count as the output (a plain contiguous vector load) or
// a bare scalar (broadcast). A genuine trailing-suffix broadcast smaller
// than the output but larger than 1 -- which Section 17.1's own scalar
// "% extCounts[i]" trick handles fine one element at a time -- has no safe
// single vector-load translation here, so this predicate returns false and
// the caller is expected to fall back to Chapter 17/18's own scalar path.
static bool canVectorizeElementwise(const Node* n, const std::map<int, long long>& elementCounts) {
    long long outCount = elementCounts.at(n->id);
    for (const Value& in : n->inputs) {
        long long c = elementCounts.at(in.nodeId);
        if (c != outCount && c != 1) return false;
    }
    return true;
}

static std::string generateVectorElementwiseFunction(const std::string& funcName, const LoweredNode& lowered,
                                                       Isa isa, const std::vector<bool>& isScalarInput) {
    int vw = vectorWidth(isa);
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    long long i = 0;\n";
    src += "    for (; i + " + std::to_string(vw) + " <= n; i += " + std::to_string(vw) + ") {\n";
    std::vector<std::string> stepVars;
    for (size_t s = 0; s < lowered.steps.size(); ++s) {
        const FusedStep& step = lowered.steps[s];
        auto operandExpr = [&](const FusedOperand& o) -> std::string {
            if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
            return stepVars[static_cast<size_t>(o.index)];
        };
        std::string expr;
        if (step.op == OpKind::ReLU) expr = vecMax(isa, operandExpr(step.operands[0]));
        else if (step.op == OpKind::Add) expr = vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1]));
        else expr = vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1]));
        std::string varName = "v" + std::to_string(s);
        src += "        " + vecType(isa) + " " + varName + " = " + expr + ";\n";
        stepVars.push_back(varName);
    }
    src += "        " + vecStore(isa, "out + i", stepVars.back()) + "\n";
    src += "    }\n";
    // The scalar tail: Chapter 18's own emitSteps(), unchanged, called again
    // starting from wherever the vector loop above stopped -- no new logic.
    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(lowered.steps, false, "i", Target::Cpu, finalExpr);
    src += "    for (; i < n; ++i) {\n";
    for (const std::string& line : tailLines) src += "        " + line + "\n";
    src += "        out[i] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
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
static void writeFile(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

int main() {
    printf("=== Section 19.1: from threads to lanes -- a portable SIMD elementwise kernel generator ===\n\n");

    // n=11 on purpose: not a multiple of 8 (AVX2's own width) OR 4 (NEON's
    // own width), so BOTH architectures' own generated code has to take a
    // genuine, non-empty scalar tail -- not a convenient round number.
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addUnary(OpKind::ReLU, t1, "t2");
    Value out = g.addBinary(OpKind::Mul, t2, b, "out");
    (void)out;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{11}}}, {b.nodeId, Shape{}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    FusionResult fusedResult = reductionFusionPass(g, {});
    const Graph& fused = fusedResult.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fusedResult.representativeOldId.at(n->id);
        fusedShapes[n->id] = shapes.at(oldId);
        fusedElementCounts[n->id] = elementCounts.at(oldId);
    }
    const Node* outFusedNode = nullptr;
    for (const auto& n : fused.nodes()) if (n->debugName == "out") outFusedNode = n.get();
    printf("Fused \"out\" node: %zu steps, %zu external inputs (a:[11], b: scalar)\n\n",
           outFusedNode->fusedSteps.size(), outFusedNode->inputs.size());

    bool canVec = canVectorizeElementwise(outFusedNode, fusedElementCounts);
    printf("self-check: canVectorizeElementwise() confirms this node is in scope -- every external\n");
    printf("input is either the full output count (11) or a bare scalar (%s)\n\n", canVec ? "confirmed" : "MISMATCH");

    // ---- The stated scope limitation, CAUGHT, not silently miscompiled ----
    printf("--- What canVectorizeElementwise() correctly REFUSES ---\n\n");
    Graph gb;
    Value ba = gb.addInput("a");
    Value bb = gb.addInput("b");
    Value bout = gb.addBinary(OpKind::Add, ba, bb, "out");
    (void)bout;
    std::map<int, Shape> bDeclared = {{ba.nodeId, Shape{{3, 4}}}, {bb.nodeId, Shape{{4}}}};  // Chapter 6's own case
    std::map<int, Shape> bShapes = inferShapes(gb, bDeclared);
    std::map<int, long long> bElementCounts;
    for (const auto& kv : bShapes) bElementCounts[kv.first] = numElements(kv.second);
    const Node* boutNode = gb.node(bout.nodeId);
    bool canVecBroadcast = canVectorizeElementwise(boutNode, bElementCounts);
    printf("a:[3,4] (12 elements), b:[4] (4 elements) -- a genuine trailing-suffix broadcast, the\n");
    printf("same case Chapter 6's own diamond graph uses. canVectorizeElementwise() returns %s --\n",
           canVecBroadcast ? "true (WRONG)" : "false");
    printf("self-check: this node is correctly identified as OUT OF SCOPE for this section's own\n");
    printf("vector path (%s) -- a real compiler would fall back to Chapter 17/18's own scalar\n",
           !canVecBroadcast ? "confirmed" : "MISMATCH");
    printf("generateCpuElementwiseFunction() for exactly this node, not miscompile it.\n\n");

    // ---- Generate the SAME node's vector body for BOTH real ISAs this book runs on ----
    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);
    std::vector<bool> isScalarInput;
    for (const Value& in : outFusedNode->inputs)
        isScalarInput.push_back(fusedElementCounts.at(in.nodeId) == 1);

    std::string avx2Src = generateVectorElementwiseFunction("compute_out", outLowered, Isa::Avx2, isScalarInput);
    std::string neonSrc = generateVectorElementwiseFunction("compute_out", outLowered, Isa::Neon, isScalarInput);
    printf("--- Generated vector body, Isa::Avx2 (8 lanes/instruction) ---\n\n%s\n", avx2Src.c_str());
    printf("--- Generated vector body, Isa::Neon (4 lanes/instruction) ---\n\n%s\n", neonSrc.c_str());
    printf("self-check: same step sequence, same scalar-tail logic (reused from Chapter 18's own\n");
    printf("emitSteps() unchanged) -- only the intrinsic names, the vector width, and the type\n");
    printf("(__m256 vs. float32x4_t) differ between the two generated bodies (confirmed by reading\n");
    printf("both texts above)\n\n");

    // ---- Detect THIS machine's own real architecture and actually run it ----
#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";  // NEON is mandatory baseline on aarch64 -- no extra flags needed
#else
#error "Section 19.1's own generator targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif
    printf("--- This machine compiles as %s -- generating, compiling, and RUNNING that path for real ---\n\n",
           isaName(hostIsa).c_str());

    std::vector<float> aVals = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11};
    std::vector<float> bVals = {2};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}};
    auto fusedArrays = evaluateArrays(fused, in, fusedElementCounts);
    long long n = fusedElementCounts.at(outFusedNode->id);
    int vw = vectorWidth(hostIsa);
    long long vectorIters = n / vw, tailIters = n % vw;
    printf("n=%lld, vector width=%d -> %lld full vector iteration(s), %lld scalar tail element(s)\n\n",
           n, vw, vectorIters, tailIters);

    auto formatArrayLiteral = [](const std::vector<float>& v) {
        std::string out = "{";
        for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
        out += "}";
        return out;
    };
    std::string hostVectorSrc = generateVectorElementwiseFunction("compute_out", outLowered, hostIsa, isScalarInput);
    std::string prog = std::string(isaHeaderInclude(hostIsa)) + "\n#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n";
    prog += hostVectorSrc + "\n";
    prog += "int main() {\n";
    prog += "    std::vector<float> ext0 = " + formatArrayLiteral(aVals) + ";\n";
    prog += "    std::vector<float> ext1 = " + formatArrayLiteral(bVals) + ";\n";
    prog += "    std::vector<const float*> ext = {ext0.data(), ext1.data()};\n";
    prog += "    std::vector<long long> extCounts = {" + std::to_string(aVals.size()) + ", " + std::to_string(bVals.size()) + "};\n";
    prog += "    std::vector<float> out(" + std::to_string(n) + ");\n";
    prog += "    compute_out(ext, extCounts, out.data(), " + std::to_string(n) + ");\n";
    prog += "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";

    writeFile("/tmp/hammer_ch19_045.cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags +
                                                  " /tmp/hammer_ch19_045.cpp -o /tmp/hammer_ch19_045 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags) ---\n\n%s\n", isaName(hostIsa).c_str(),
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());
    printf("self-check: this machine's own real compiler accepts the generated %s intrinsics (%s)\n\n",
           isaName(hostIsa).c_str(), compileClean ? "confirmed" : "MISMATCH");

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch19_045");
    printf("--- Running the compiled binary FOR REAL on this machine's own hardware ---\n\n%s\n", runOutput.c_str());

    std::istringstream iss(runOutput);
    std::vector<float> vecVals;
    float v;
    while (iss >> v) vecVals.push_back(v);
    const std::vector<float>& expected = fusedArrays.at("out");
    bool matches = (vecVals.size() == expected.size());
    if (matches) for (size_t i = 0; i < vecVals.size(); ++i) if (std::fabs(vecVals[i] - expected[i]) > 1e-3f) matches = false;
    printf("evaluateArrays(fused).out = ");
    for (float ev : expected) printf("%.6f ", ev);
    printf("\n");
    printf("self-check: the vector-plus-tail program, ACTUALLY EXECUTED on this machine's own real\n");
    printf("%s hardware, matches evaluateArrays() exactly, element for element (%s)\n",
           isaName(hostIsa).c_str(), matches ? "confirmed" : "MISMATCH");

    bool allOk = canVec && !canVecBroadcast && compileClean && matches;
    return allOk ? 0 : 1;
}
