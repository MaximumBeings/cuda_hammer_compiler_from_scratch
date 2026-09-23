// Chapter 19: Generating Vectorized CPU Code
// 046_fusing_multiply_and_add_a_real_peephole_optimization_vectorized.cpp
//
// Section 19.2 -- Section 19.1's own vector codegen emits one instruction
// per FusedStep, unconditionally. But real hardware -- both of this book's
// own real targets -- offers a genuinely cheaper instruction for a very
// common pattern: multiply, then immediately add. AVX2/FMA's own
// _mm256_fmadd_ps and NEON's own vfmaq_f32 each compute a*b+c in ONE
// instruction instead of two, and this section adds a real peephole
// optimization to Section 19.1's own generator that finds a Mul step
// immediately followed by an Add step consuming it, and emits ONE fused
// instruction instead of two separate ones -- a genuine, measurable
// instruction-count reduction, not a cosmetic rewrite.
//
// The interesting question a peephole fusion like this always raises is
// SAFETY: folding Mul step M into Add step A only makes sense if M's own
// separately-computed value is never needed anywhere else -- if it were,
// this section would have to keep computing it anyway, and the fold would
// save nothing. This section proves that safety condition holds here not
// by testing it against a constructed counterexample (this book's own
// fusion passes make an actual unsafe case impossible to construct in the
// first place -- explained below), but by tracing it back to an invariant
// Chapter 13 established five chapters ago.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 046_fusing_multiply_and_add_a_real_peephole_optimization_vectorized.cpp -o 046_driver
// Run:     ./046_driver
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
//
// The invariant this whole section leans on lives HERE, unchanged since
// Chapter 13: `mustBeExternal` includes `consumers.at(oldId) != 1`. Any
// node read by more than one other node in the ORIGINAL graph is always
// externalized -- forced OUT of the group, never inlined as a PriorStep --
// before it could ever be shared by two steps within one fused group's own
// step list. The consequence: within any ONE group's own fusedSteps,
// every step except the group's own final output is read by AT MOST one
// later step. That is exactly the safety condition this section's own FMA
// fold needs -- and it comes for free, not from new logic in this file.

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

// ==================== emitSteps() (from Section 18.1, unchanged -- reused for the scalar tail) ====================

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

// ==================== Isa / vector primitives (from Section 19.1, unchanged) ====================

enum class Isa { Avx2, Neon };

static int vectorWidth(Isa isa) { return (isa == Isa::Avx2) ? 8 : 4; }
static std::string vecType(Isa isa) { return (isa == Isa::Avx2) ? "__m256" : "float32x4_t"; }
static std::string isaHeaderInclude(Isa isa) { return (isa == Isa::Avx2) ? "#include <immintrin.h>" : "#include <arm_neon.h>"; }
static std::string isaName(Isa isa) { return (isa == Isa::Avx2) ? "AVX2/FMA" : "NEON"; }

static std::string vecLoadOrBroadcast(Isa isa, int index, const std::string& baseExpr, bool isScalar) {
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
    return (isa == Isa::Avx2) ? ("_mm256_storeu_ps(" + dst + ", " + val + ");") : ("vst1q_f32(" + dst + ", " + val + ");");
}
// Section 19.1's own canVectorizeElementwise() scope check isn't repeated
// here -- both test graphs below are plain elementwise groups the same
// shape as Section 19.1's own worked example, so there is nothing new to
// gate. It returns unchanged in Section 19.3's capstone, where a graph with
// a genuine broadcast reappears.

// ==================== Section 19.2: the FMA fold ====================

// x86's own _mm256_fmadd_ps(a, b, c) computes a*b + c directly. NEON's own
// vfmaq_f32(acc, a, b) computes acc + a*b -- the SAME three operands, in a
// DIFFERENT argument order (accumulator first). This helper hides that
// difference the same way Chapter 18's own maxExpr() hid std::max vs.
// fmaxf: one call site, one meaning ("mulLhs*mulRhs + addend"), two real
// argument orders underneath.
static std::string vecFma(Isa isa, const std::string& mulLhs, const std::string& mulRhs, const std::string& addend) {
    if (isa == Isa::Avx2) return "_mm256_fmadd_ps(" + mulLhs + ", " + mulRhs + ", " + addend + ")";
    return "vfmaq_f32(" + addend + ", " + mulLhs + ", " + mulRhs + ")";
}

// Counts how many times each step is read by a LATER step (PriorStep) or
// by the group's own final returned value. Given Chapter 13's own
// single-consumer inlining invariant (explained above the fusion pass),
// this always evaluates to exactly 1 for every non-final step in a real
// fused group -- so this function exists to make that invariant CHECKABLE,
// not because the fold below needs to defend against a case that can
// actually arise from this book's own passes.
static std::vector<int> computeStepUseCounts(const std::vector<FusedStep>& steps, bool isReduction) {
    size_t stepCount = isReduction ? steps.size() - 1 : steps.size();
    std::vector<int> useCounts(stepCount, 0);
    for (size_t s = 0; s < stepCount; ++s)
        for (const FusedOperand& o : steps[s].operands)
            if (o.kind == OperandKind::PriorStep) useCounts[o.index]++;
    if (isReduction) {
        const FusedOperand& sumOperand = steps.back().operands[0];
        if (sumOperand.kind == OperandKind::PriorStep) useCounts[sumOperand.index]++;
    } else if (stepCount > 0) {
        // the group's own final value (the last step) is its own "use"
        useCounts[stepCount - 1]++;
    }
    return useCounts;
}

static bool stepUsesOperand(const FusedStep& step, OperandKind kind, int index) {
    for (const FusedOperand& o : step.operands) if (o.kind == kind && o.index == index) return true;
    return false;
}

// Extends Section 19.1's own generateVectorElementwiseFunction() with
// exactly one new rule: when step s is Add and its immediately preceding
// step s-1 is a Mul used by NOTHING except step s, emit ONE vecFma() line
// for step s and skip step s-1's own line entirely -- the Mul's own
// operands get read directly into the FMA instead of being materialized
// as their own separate vector variable.
static std::string generateVectorElementwiseFunctionFma(const std::string& funcName, const LoweredNode& lowered,
                                                          Isa isa, const std::vector<bool>& isScalarInput,
                                                          int& foldCount, int& rawStepCount) {
    const std::vector<FusedStep>& steps = lowered.steps;
    std::vector<int> useCounts = computeStepUseCounts(steps, false);
    rawStepCount = static_cast<int>(steps.size());
    foldCount = 0;

    // Pass 1: decide which Mul steps get folded into the Add immediately after them.
    std::vector<bool> foldedAway(steps.size(), false);
    for (size_t s = 0; s < steps.size(); ++s) {
        if (steps[s].op != OpKind::Add || s == 0) continue;
        if (steps[s - 1].op != OpKind::Mul) continue;
        if (useCounts[s - 1] != 1) continue;  // always true here, per the invariant above -- checked, not assumed
        if (!stepUsesOperand(steps[s], OperandKind::PriorStep, static_cast<int>(s - 1))) continue;
        foldedAway[s - 1] = true;
        foldCount++;
    }

    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    int vw = vectorWidth(isa);
    src += "    long long i = 0;\n";
    src += "    for (; i + " + std::to_string(vw) + " <= n; i += " + std::to_string(vw) + ") {\n";

    std::vector<std::string> stepVars(steps.size());
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
        return stepVars[static_cast<size_t>(o.index)];
    };
    // Pass 2: emit.
    for (size_t s = 0; s < steps.size(); ++s) {
        if (foldedAway[s]) continue;  // this Mul's own line is skipped -- folded into the Add right after it
        const FusedStep& step = steps[s];
        std::string varName = "v" + std::to_string(s);
        if (step.op == OpKind::Add && s > 0 && foldedAway[s - 1]) {
            const FusedStep& mulStep = steps[s - 1];
            std::string mulLhs = operandExpr(mulStep.operands[0]);
            std::string mulRhs = operandExpr(mulStep.operands[1]);
            std::string addend;
            for (const FusedOperand& o : step.operands)
                if (!(o.kind == OperandKind::PriorStep && o.index == static_cast<int>(s - 1))) addend = operandExpr(o);
            src += "        " + vecType(isa) + " " + varName + " = " + vecFma(isa, mulLhs, mulRhs, addend) +
                   ";  // fused: v" + std::to_string(s - 1) + " folded in\n";
        } else if (step.op == OpKind::ReLU) {
            src += "        " + vecType(isa) + " " + varName + " = " + vecMax(isa, operandExpr(step.operands[0])) + ";\n";
        } else if (step.op == OpKind::Add) {
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        } else {
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        }
        stepVars[s] = varName;
    }
    src += "        " + vecStore(isa, "out + i", stepVars.back()) + "\n";
    src += "    }\n";
    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(steps, false, "i", Target::Cpu, finalExpr);
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
    printf("=== Section 19.2: fusing multiply and add -- a real peephole optimization, vectorized ===\n\n");

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    // t1 = Mul(a, b), out = Add(t1, c) -- a genuine multiply-then-add
    // pattern. n=9: one full vector iteration on either ISA (8 or 4 wide)
    // plus a real remainder, same discipline as Section 19.1.
    Graph g;
    Value a = g.addInput("a");
    Value b = g.addInput("b");
    Value c = g.addInput("c");
    Value t1 = g.addBinary(OpKind::Mul, a, b, "t1");
    Value out = g.addBinary(OpKind::Add, t1, c, "out");
    (void)out;
    std::map<int, Shape> declared = {{a.nodeId, Shape{{9}}}, {b.nodeId, Shape{}}, {c.nodeId, Shape{}}};
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
    printf("Fused \"out\" node: %zu raw steps (Mul, then Add) -- a fold candidate\n\n", outFusedNode->fusedSteps.size());

    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);
    std::vector<bool> isScalarInput;
    for (const Value& in : outFusedNode->inputs) isScalarInput.push_back(fusedElementCounts.at(in.nodeId) == 1);

    int foldCount = 0, rawStepCount = 0;
    std::string foldedSrc = generateVectorElementwiseFunctionFma("compute_out", outLowered, hostIsa, isScalarInput,
                                                                    foldCount, rawStepCount);
    printf("--- Generated vector body WITH the FMA fold (%s) ---\n\n%s\n", isaName(hostIsa).c_str(), foldedSrc.c_str());
    printf("self-check: %d raw step(s) (Mul, Add) generate exactly 1 vector instruction inside the\n", rawStepCount);
    printf("main loop -- %d fold(s) applied, the Mul's own line never separately emitted (%s)\n\n",
           foldCount, foldCount == 1 ? "confirmed" : "MISMATCH");

    // ---- A pattern that should NOT fold: ReLU immediately before Add (no Mul involved) ----
    Graph g2;
    Value a2 = g2.addInput("a");
    Value b2 = g2.addInput("b");
    Value r1 = g2.addUnary(OpKind::ReLU, a2, "r1");
    Value out2 = g2.addBinary(OpKind::Add, r1, b2, "out2");
    (void)out2;
    std::map<int, Shape> declared2 = {{a2.nodeId, Shape{{9}}}, {b2.nodeId, Shape{}}};
    std::map<int, Shape> shapes2 = inferShapes(g2, declared2);
    std::map<int, long long> elementCounts2;
    for (const auto& kv : shapes2) elementCounts2[kv.first] = numElements(kv.second);
    FusionResult fusedResult2 = reductionFusionPass(g2, {});
    std::map<int, Shape> fusedShapes2;
    std::map<int, long long> fusedElementCounts2;
    for (const auto& n : fusedResult2.graph.nodes()) {
        int oldId = fusedResult2.representativeOldId.at(n->id);
        fusedShapes2[n->id] = shapes2.at(oldId);
        fusedElementCounts2[n->id] = elementCounts2.at(oldId);
    }
    const Node* out2FusedNode = nullptr;
    for (const auto& n : fusedResult2.graph.nodes()) if (n->debugName == "out2") out2FusedNode = n.get();
    LoweredNode out2Lowered = lowerNode(out2FusedNode, fusedShapes2, fusedElementCounts2);
    std::vector<bool> isScalarInput2;
    for (const Value& in : out2FusedNode->inputs) isScalarInput2.push_back(fusedElementCounts2.at(in.nodeId) == 1);
    int foldCount2 = 0, rawStepCount2 = 0;
    generateVectorElementwiseFunctionFma("compute_out2", out2Lowered, hostIsa, isScalarInput2, foldCount2, rawStepCount2);
    printf("--- A pattern that correctly does NOT fold: ReLU(a) then Add(that, b) -- no Mul at all ---\n\n");
    printf("self-check: %d raw step(s) (ReLU, Add), %d fold(s) applied -- the fold rule only ever\n",
           rawStepCount2, foldCount2);
    printf("fires on an Add whose immediately preceding step is a Mul, and correctly leaves this\n");
    printf("ReLU-then-Add pattern alone (%s)\n\n", foldCount2 == 0 ? "confirmed" : "MISMATCH");

    // ---- Compile and run the FOLDED version for real, on this host ----
    std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> bVals = {2};
    std::vector<float> cVals = {10};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}, {"c", cVals}};
    auto fusedArrays = evaluateArrays(fused, in, fusedElementCounts);
    long long n = fusedElementCounts.at(outFusedNode->id);

    auto formatArrayLiteral = [](const std::vector<float>& v) {
        std::string out = "{";
        for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
        out += "}";
        return out;
    };
    std::string prog = std::string(isaHeaderInclude(hostIsa)) + "\n#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n";
    prog += foldedSrc + "\n";
    prog += "int main() {\n";
    prog += "    std::vector<float> ext0 = " + formatArrayLiteral(aVals) + ";\n";
    prog += "    std::vector<float> ext1 = " + formatArrayLiteral(bVals) + ";\n";
    prog += "    std::vector<float> ext2 = " + formatArrayLiteral(cVals) + ";\n";
    prog += "    std::vector<const float*> ext = {ext0.data(), ext1.data(), ext2.data()};\n";
    prog += "    std::vector<long long> extCounts = {" + std::to_string(aVals.size()) + ", " +
            std::to_string(bVals.size()) + ", " + std::to_string(cVals.size()) + "};\n";
    prog += "    std::vector<float> out(" + std::to_string(n) + ");\n";
    prog += "    compute_out(ext, extCounts, out.data(), " + std::to_string(n) + ");\n";
    prog += "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";

    writeFile("/tmp/hammer_ch19_046.cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags +
                                                  " /tmp/hammer_ch19_046.cpp -o /tmp/hammer_ch19_046 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags, folded FMA body) ---\n\n%s\n", isaName(hostIsa).c_str(),
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch19_046");
    printf("--- Running the folded FMA program FOR REAL on this machine's own %s hardware ---\n\n%s\n",
           isaName(hostIsa).c_str(), runOutput.c_str());

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
    printf("self-check: a single fmadd/vfma instruction computes the identical result an unfused\n");
    printf("mul+add pair would -- confirmed against evaluateArrays() (%s)\n", matches ? "confirmed" : "MISMATCH");

    bool allOk = compileClean && matches && (foldCount == 1) && (foldCount2 == 0);
    return allOk ? 0 : 1;
}
