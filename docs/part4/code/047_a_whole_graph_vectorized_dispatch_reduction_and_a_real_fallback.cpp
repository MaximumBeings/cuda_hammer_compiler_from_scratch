// Chapter 19: Generating Vectorized CPU Code
// 047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback.cpp
//
// Section 19.3 -- the capstone. Sections 19.1 and 19.2 each vectorized ONE
// node at a time, chosen by hand. A real backend has to look at an entire
// fused graph and decide, node by node, which of three things to generate:
// the FMA-fold-aware vector elementwise body (19.1 + 19.2), a genuinely new
// vectorized REDUCTION body this section adds, or -- when a node's own
// shapes don't fit this backend's scope, exactly as canVectorizeElementwise()
// already checks for -- the plain scalar body Chapter 18 already knows how
// to generate. This section builds that per-node dispatch, proves the
// fallback path is real (not just a returned "false") by compiling and
// running a genuinely non-vectorizable broadcast case, and then runs
// Chapter 16's own 10-node capstone graph -- the SAME graph Section 17.3
// and Section 18.3 already made real, on a CPU loop nest and then on a GPU
// -- through the dispatcher, compiling and executing the WHOLE thing for
// real on both of this book's own real architectures (AVX2/FMA and NEON).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback.cpp -o 047_driver
// Run:     ./047_driver
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
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

// ==================== boundedReductionFusionPass() (from Chapter 16 / Section 17.3, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static std::map<int, long long> computeChainDepths(const Graph& g, const std::map<int, int>& consumers,
                                                     const TopoResult& topo) {
    std::map<int, long long> depth;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        bool isChainCandidate = (n->op == OpKind::Add || n->op == OpKind::Mul || n->op == OpKind::ReLU) &&
                                 consumers.at(id) == 1;
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
static FusedOperand resolveIntoGroupBounded(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                             const std::set<int>& sizeCapBoundaries,
                                             const std::map<int, Value>& materialized,
                                             std::vector<Value>& externalInputs,
                                             std::map<int, int>& externalIndexByOldId,
                                             std::vector<FusedStep>& steps,
                                             std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) || (consumers.at(oldId) != 1) ||
                           (sizeCapBoundaries.count(oldId) > 0);
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
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (n->op == OpKind::Const) {
            Value v = out.addConst(n->constValue, n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (n->op == OpKind::Sum) {
            std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
            std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
            FusedOperand summedOperand = resolveIntoGroupBounded(n->inputs[0].nodeId, g, consumers, sizeCapBoundaries,
                                                                   materialized, externalInputs, externalIndexByOldId,
                                                                   steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0) continue;
        std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
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

// ==================== emitSteps() (from Section 18.1, unchanged -- the scalar backend, still used for
//                      every scalar tail loop AND for the whole-function scalar fallback) ====================

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

// ---- Section 18.1's own scalar elementwise generator: the fallback path
// for any node canVectorizeElementwise() below rejects. ----
static std::string generateCpuElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
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

// This section is where canVectorizeElementwise() finally earns its keep:
// every node in this file's own dispatcher runs through it for real, and a
// node it rejects gets the scalar fallback above -- generated, compiled,
// and run, not just described.
static bool canVectorizeElementwise(const Node* n, const std::map<int, long long>& elementCounts) {
    long long outCount = elementCounts.at(n->id);
    for (const Value& in : n->inputs) {
        long long c = elementCounts.at(in.nodeId);
        if (c != outCount && c != 1) return false;
    }
    return true;
}

// ==================== Section 19.2's own FMA fold (unchanged) ====================

static std::string vecFma(Isa isa, const std::string& mulLhs, const std::string& mulRhs, const std::string& addend) {
    if (isa == Isa::Avx2) return "_mm256_fmadd_ps(" + mulLhs + ", " + mulRhs + ", " + addend + ")";
    return "vfmaq_f32(" + addend + ", " + mulLhs + ", " + mulRhs + ")";
}
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
        useCounts[stepCount - 1]++;
    }
    return useCounts;
}
static bool stepUsesOperand(const FusedStep& step, OperandKind kind, int index) {
    for (const FusedOperand& o : step.operands) if (o.kind == kind && o.index == index) return true;
    return false;
}
static std::string generateVectorElementwiseFunctionFma(const std::string& funcName, const LoweredNode& lowered,
                                                          Isa isa, const std::vector<bool>& isScalarInput,
                                                          int& foldCount, int& rawStepCount) {
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
    for (size_t s = 0; s < steps.size(); ++s) {
        if (foldedAway[s]) continue;
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

// ==================== Section 19.3: vectorized reduction ====================
//
// A FusedReduction group's own steps are, structurally, the SAME per-element
// step sequence Section 19.1/19.2 already vectorize, plus one trailing Sum
// step that folds every one of those per-element RESULTS into a single
// scalar. Vectorizing it needs exactly one new idea: accumulate each lane's
// own running total into a VECTOR accumulator across the main loop, instead
// of storing per-lane results -- and only collapse that accumulator down to
// one float ("horizontal sum") once, after the loop ends, not once per
// element.
//
// The horizontal sum below uses the simplest technique that is genuinely
// portable across BOTH of this book's own real ISAs: store the
// accumulator's own lanes out to a small array and add them up in plain
// scalar code. NEON offers a real single-instruction alternative here --
// vaddvq_f32(acc) sums all four lanes in one instruction -- but AVX2 has no
// equivalent single instruction that reduces all eight lanes without
// reaching for AVX-512, so this section deliberately uses the
// slower-but-uniform array-extraction approach on BOTH backends rather than
// special-casing NEON's own faster path. A production compiler would
// dispatch to vaddvq_f32 on NEON and keep the array-extraction fallback
// only for AVX2.
static std::string generateVectorReductionFunction(const std::string& funcName, const LoweredNode& lowered,
                                                     Isa isa, const std::vector<bool>& isScalarInput) {
    const std::vector<FusedStep>& steps = lowered.steps;
    size_t stepCount = steps.size() - 1;  // every step except the trailing Sum
    long long reduceExtent = lowered.nest.loops.back().extent;
    int vw = vectorWidth(isa);

    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    (void)extCounts;\n";
    src += "    " + vecType(isa) + " acc = " + (isa == Isa::Avx2 ? "_mm256_setzero_ps()" : "vdupq_n_f32(0.0f)") + ";\n";
    src += "    long long i = 0;\n";
    src += "    for (; i + " + std::to_string(vw) + " <= " + std::to_string(reduceExtent) + "; i += " +
           std::to_string(vw) + ") {\n";

    std::vector<std::string> stepVars(stepCount);
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < stepCount; ++s) {
        const FusedStep& step = steps[s];
        std::string varName = "v" + std::to_string(s);
        if (step.op == OpKind::ReLU)
            src += "        " + vecType(isa) + " " + varName + " = " + vecMax(isa, operandExpr(step.operands[0])) + ";\n";
        else if (step.op == OpKind::Add)
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        else
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        stepVars[s] = varName;
    }
    const FusedOperand& sumOperand = steps.back().operands[0];
    std::string sumExpr = (sumOperand.kind == OperandKind::ExternalInput)
                               ? vecLoadOrBroadcast(isa, sumOperand.index, "i", isScalarInput[sumOperand.index])
                               : stepVars[static_cast<size_t>(sumOperand.index)];
    src += "        acc = " + vecAdd(isa, "acc", sumExpr) + ";\n";
    src += "    }\n";
    src += "    float lanes[" + std::to_string(vw) + "];\n";
    src += "    " + vecStore(isa, "lanes", "acc") + "\n";
    src += "    float hsum = 0.0f;\n";
    src += "    for (int lane = 0; lane < " + std::to_string(vw) + "; ++lane) hsum += lanes[lane];\n";

    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(steps, true, "i", Target::Cpu, finalExpr);
    src += "    for (; i < " + std::to_string(reduceExtent) + "; ++i) {\n";
    for (const std::string& line : tailLines) src += "        " + line + "\n";
    src += "        hsum += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = hsum;\n}\n";
    return src;
}

// ==================== Section 19.3: the per-node dispatcher ====================
//
// One function decides, per node, which of three generators produced its
// body: a reduction node always takes the vectorized-reduction path above;
// an elementwise node takes the FMA-fold-aware vector path from Section
// 19.2 when canVectorizeElementwise() allows it, and Chapter 18's own plain
// scalar path otherwise. No other logic -- this IS the whole dispatcher.
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }

enum class Backend { VectorFma, VectorReduction, ScalarFallback };

static std::string generateFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered,
                                            Isa isa, const std::map<int, long long>& elementCounts,
                                            Backend& backendUsed, int& foldCount) {
    foldCount = 0;
    std::vector<bool> isScalarInput;
    for (const Value& in : n->inputs) isScalarInput.push_back(elementCounts.at(in.nodeId) == 1);

    if (isReductionNode(n)) {
        backendUsed = Backend::VectorReduction;
        return generateVectorReductionFunction(funcName, lowered, isa, isScalarInput);
    }
    if (canVectorizeElementwise(n, elementCounts)) {
        backendUsed = Backend::VectorFma;
        int rawStepCount = 0;
        return generateVectorElementwiseFunctionFma(funcName, lowered, isa, isScalarInput, foldCount, rawStepCount);
    }
    backendUsed = Backend::ScalarFallback;
    return generateCpuElementwiseFunction(funcName, lowered);
}
static const char* backendName(Backend b) {
    return b == Backend::VectorFma ? "VECTOR+FMA" : b == Backend::VectorReduction ? "VECTOR-REDUCTION" : "SCALAR-FALLBACK";
}

// ==================== Whole-graph driver, dispatched per node (mirrors Section 18.3's own
//                      generateFullCpuProgram(), with generateFunctionForNode() in place of
//                      the always-scalar generateCpuFunctionForNode()) ====================

static std::string formatLiteralArray(const std::vector<float>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
    out += "}";
    return out;
}

static std::string generateFullVectorizedProgram(const Graph& g, const std::map<int, Shape>& shapes,
                                                   const std::map<int, long long>& elementCounts,
                                                   const std::map<std::string, std::vector<float>>& inputArrays,
                                                   Isa isa, std::vector<std::pair<std::string, Backend>>& dispatchLog,
                                                   int& totalFolds) {
    TopoResult topo = topologicalSort(g);
    std::string prog = std::string(isaHeaderInclude(isa)) +
                        "\n#include <cstdio>\n#include <vector>\n#include <map>\n#include <string>\n#include <algorithm>\n\n";

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

    prog += "int main() {\n";
    prog += "    std::map<std::string, std::vector<float>> buf;\n";
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
            prog += "        ext.push_back(buf[\"" + inNode->debugName + "\"].data()); extCounts.push_back(" +
                    std::to_string(elementCounts.at(in.nodeId)) + ");\n";
        }
        if (isReduction) prog += "        " + funcNameById.at(id) + "(ext, extCounts, buf[\"" + n->debugName + "\"].data());\n    }\n";
        else prog += "        " + funcNameById.at(id) + "(ext, extCounts, buf[\"" + n->debugName + "\"].data(), " +
                     std::to_string(elementCounts.at(id)) + ");\n    }\n";
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

// ==================== Shell-out / parsing helpers ====================

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
static std::map<std::string, std::vector<float>> parseNamedBuffers(const std::string& text) {
    std::map<std::string, std::vector<float>> result;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string name = line.substr(0, colon);
        std::istringstream rest(line.substr(colon + 1));
        std::vector<float> vals;
        float v;
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

int main() {
    printf("=== Section 19.3: a whole graph, vectorized dispatch, reduction, and a real fallback ===\n\n");

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    bool allOk = true;

    // ---------------------------------------------------------------
    // Part 1: prove the fallback is REAL. a genuine trailing-dimension
    // broadcast -- a:[3,4], b:[4], out = Add(a,b) -- is exactly the shape
    // canVectorizeElementwise() was always documented to reject (count(a)=12,
    // count(b)=4: neither equals the other nor is it a scalar). This isn't
    // a hypothetical: the dispatcher below hits it, chooses the scalar
    // path, and that scalar path is compiled and run for real, matching
    // evaluateArrays() exactly -- "not vectorized" and "wrong" are
    // different things, and this proves it.
    // ---------------------------------------------------------------
    printf("--- Part 1: a genuine broadcast the vector path can't safely handle ---\n\n");
    {
        Graph gb;
        Value ba = gb.addInput("a");
        Value bb = gb.addInput("b");
        Value bout = gb.addBinary(OpKind::Add, ba, bb, "out");
        std::map<int, Shape> declaredB = {{ba.nodeId, Shape{{3, 4}}}, {bb.nodeId, Shape{{4}}}};
        std::map<int, Shape> shapesB = inferShapes(gb, declaredB);
        std::map<int, long long> countsB;
        for (const auto& kv : shapesB) countsB[kv.first] = numElements(kv.second);

        const Node* outNode = gb.node(bout.nodeId);
        bool canVec = canVectorizeElementwise(outNode, countsB);
        printf("a:[3,4] (12 elements), b:[4] (4 elements) -- canVectorizeElementwise(out) = %s\n",
               canVec ? "true" : "false");
        printf("self-check: a real trailing-dimension broadcast is correctly rejected by the vector\n");
        printf("path's own scope check (%s)\n\n", !canVec ? "confirmed" : "MISMATCH");

        LoweredNode boutLowered = lowerNode(outNode, shapesB, countsB);
        Backend backendUsed; int foldCount = 0;
        std::string fnSrc = generateFunctionForNode("compute_out", outNode, boutLowered, hostIsa, countsB, backendUsed, foldCount);
        printf("dispatcher chose: %s\n\n", backendName(backendUsed));

        std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<float> bVals = {100, 200, 300, 400};
        std::map<std::string, std::vector<float>> inB = {{"a", aVals}, {"b", bVals}};
        auto expectedB = evaluateArrays(gb, inB, countsB);

        std::string prog = "#include <cstdio>\n#include <vector>\n\n" + fnSrc +
                            "\nint main() {\n    std::vector<float> ext0 = " + formatLiteralArray(aVals) +
                            ";\n    std::vector<float> ext1 = " + formatLiteralArray(bVals) +
                            ";\n    std::vector<const float*> ext = {ext0.data(), ext1.data()};\n"
                            "    std::vector<long long> extCounts = {12, 4};\n"
                            "    std::vector<float> out(12);\n"
                            "    compute_out(ext, extCounts, out.data(), 12);\n"
                            "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";
        writeFile("/tmp/hammer_ch19_047_fallback.cpp", prog);
        std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2 /tmp/hammer_ch19_047_fallback.cpp "
                                                      "-o /tmp/hammer_ch19_047_fallback 2>&1");
        bool compileClean = compileLog.empty();
        std::string runOutput = runShellCaptureAll("/tmp/hammer_ch19_047_fallback");
        std::istringstream iss(runOutput);
        std::vector<float> got;
        float v;
        while (iss >> v) got.push_back(v);
        bool matches = arraysMatch(got, expectedB.at("out"));
        printf("scalar fallback, compiled and run for real: %s. output %s evaluateArrays() (%s)\n\n",
               compileClean ? "clean compile" : "COMPILE FAILED", matches ? "matches" : "does NOT match",
               (compileClean && matches) ? "confirmed" : "MISMATCH");
        allOk = allOk && compileClean && matches && !canVec && (backendUsed == Backend::ScalarFallback);
    }

    // ---------------------------------------------------------------
    // Part 2: Chapter 16's own 10-node capstone graph, reused verbatim --
    // the same graph Section 17.3 made real on a CPU loop nest and Section
    // 18.3 made real on a GPU. This time every fused node is dispatched
    // through generateFunctionForNode(), compiled, and run for real on
    // THIS machine's own real vector hardware.
    // ---------------------------------------------------------------
    printf("--- Part 2: Chapter 16's 10-node graph, vectorized end to end ---\n\n");
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
    (void)t1; (void)t2; (void)t3; (void)t4; (void)y2; (void)y;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{8}}}, {b.nodeId, Shape{}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    std::vector<float> aVals = {1, -2, 3, -4, 5, -6, 7, -8};
    std::vector<float> bVals = {2};
    std::map<std::string, std::vector<float>> inputArrays = {{"a", aVals}, {"b", bVals}};
    auto origArrays = evaluateArrays(g, inputArrays, elementCounts);
    printf("a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = ");
    for (float v : origArrays.at("y")) printf("%.6f ", v);
    printf("\n\n");

    FusionResult fr = boundedReductionFusionPass(g, elementCounts, 3);
    const Graph& fused = fr.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fr.representativeOldId.at(n->id);
        fusedShapes[n->id] = shapes.at(oldId);
        fusedElementCounts[n->id] = elementCounts.at(oldId);
    }
    printf("fused graph (maxChainLength=3, same cap Section 17.3/18.3 used): ");
    for (const auto& n : fused.nodes()) {
        printf("%s", n->debugName.c_str());
        if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
        else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
        printf(" ");
    }
    printf("\n\n");

    std::vector<std::pair<std::string, Backend>> dispatchLog;
    int totalFolds = 0;
    std::string prog = generateFullVectorizedProgram(fused, fusedShapes, fusedElementCounts, inputArrays,
                                                       hostIsa, dispatchLog, totalFolds);
    printf("per-node dispatch (%s):\n", isaName(hostIsa).c_str());
    for (const auto& entry : dispatchLog) printf("  %-4s -> %s\n", entry.first.c_str(), backendName(entry.second));
    printf("\nself-check: every fused node in this graph -- 3 elementwise groups plus the\n");
    printf("reduction -- is genuinely vectorizable (no broadcast beyond a scalar anywhere\n");
    printf("in this graph), so the dispatcher chose VECTOR paths for all %zu nodes, 0\n", dispatchLog.size());
    printf("scalar fallbacks (%s)\n\n", (dispatchLog.size() > 0 && std::all_of(dispatchLog.begin(), dispatchLog.end(),
           [](const std::pair<std::string, Backend>& e) { return e.second != Backend::ScalarFallback; }))
           ? "confirmed" : "MISMATCH");
    printf("self-check: %d total FMA fold(s) applied across the 3 elementwise groups --\n", totalFolds);
    printf("t3's own ReLU-then-Mul-then-Add folds its trailing Mul+Add, t5's own Mul-then-Add\n");
    printf("folds, and y's own Mul-then-Add (s + t5*b) folds too, for exactly 3 (%s)\n\n",
           totalFolds == 3 ? "confirmed" : "MISMATCH");

    std::string cpuStem = "/tmp/hammer_ch19_047_capstone";
    writeFile(cpuStem + ".cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags + " " + cpuStem +
                                                  ".cpp -o " + cpuStem + " 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags, %zu generated functions) ---\n\n%s\n", isaName(hostIsa).c_str(),
           dispatchLog.size(), compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());

    std::string runOutput = runShellCaptureAll(cpuStem);
    printf("--- Running the whole vectorized graph FOR REAL on this machine's own %s hardware ---\n\n%s\n",
           isaName(hostIsa).c_str(), runOutput.c_str());
    auto generatedArrays = parseNamedBuffers(runOutput);

    bool allNodesMatch = true;
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        bool ok = generatedArrays.count(n->debugName) && arraysMatch(generatedArrays.at(n->debugName),
                                                                       evaluateArrays(fused, inputArrays, fusedElementCounts).at(n->debugName));
        if (!ok) allNodesMatch = false;
    }
    const std::vector<float>& yGenerated = generatedArrays.at("y");
    const std::vector<float>& yExpected = origArrays.at("y");
    bool yMatches = arraysMatch(yGenerated, yExpected);
    printf("evaluateArrays(original).y = ");
    for (float v : yExpected) printf("%.6f ", v);
    printf("\ngenerated (vectorized, %zu functions dispatched, compiled and run for real).y = ", dispatchLog.size());
    for (float v : yGenerated) printf("%.6f ", v);
    printf("\n\n");
    printf("self-check: y[0] = %.6f -- the SAME number Section 17.3's own CPU loop nest and\n", yGenerated.empty() ? 0.0f : yGenerated[0]);
    printf("Section 18.3's own CPU reference both reported for this exact graph, now produced\n");
    printf("by real vector instructions on real %s hardware (%s)\n\n", isaName(hostIsa).c_str(),
           (!yGenerated.empty() && std::fabs(yGenerated[0] - 132.0f) < 1e-2f) ? "confirmed" : "MISMATCH");
    printf("self-check: every generated buffer (not just y) matches evaluateArrays() on the\n");
    printf("SAME fused graph -- the vector path and the interpreter agree at every node, not\n");
    printf("just at the output (%s)\n", allNodesMatch ? "confirmed" : "MISMATCH");

    allOk = allOk && compileClean && yMatches && allNodesMatch && (totalFolds == 3) &&
            std::all_of(dispatchLog.begin(), dispatchLog.end(),
                        [](const std::pair<std::string, Backend>& e) { return e.second != Backend::ScalarFallback; });

    return allOk ? 0 : 1;
}
