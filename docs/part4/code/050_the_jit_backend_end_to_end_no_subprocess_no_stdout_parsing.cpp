// Chapter 20: A JIT Backend: Compiling and Loading Generated Code at Runtime
// 050_the_jit_backend_end_to_end_no_subprocess_no_stdout_parsing.cpp
//
// Section 20.3 -- the capstone. Sections 20.1 and 20.2 proved the JIT
// loading mechanism itself: extern "C", dlopen(), dlsym(), a real function
// pointer called directly. This section proves the loader is BACKEND
// AGNOSTIC -- it does not care whether a function's own BODY came from
// Chapter 18's plain scalar emitSteps() path or Chapter 19's vectorized
// AVX2/FMA-or-NEON intrinsics path, only that the function is extern "C"
// and matches an expected signature. Chapter 16's own 10-node capstone
// graph -- the SAME graph Sections 17.3, 18.3, and 19.3 have all already
// used -- gets JIT-compiled TWICE from this section, once through a scalar
// dispatcher and once through Chapter 19's own vector-FMA-and-reduction
// dispatcher, wrapped in extern "C" for the first time. Both compiled
// libraries are dlopen()'d, every one of their own node functions is
// dlsym()'d and called as a real function pointer -- ZERO subprocess
// launches and ZERO parsed stdout text anywhere in EXECUTION, for the
// first time in this book -- and both report the same y[0]=132.0 Chapters
// 16 through 19 have all already established.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 050_the_jit_backend_end_to_end_no_subprocess_no_stdout_parsing.cpp -o 050_driver -ldl
// Run:     ./050_driver
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
#include <dlfcn.h>

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

// ==================== emitSteps() (from Section 18.1, unchanged) ====================

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

// ==================== Section 20.1/20.2's own extern "C" scalar generators (unchanged) ====================

static std::string generateExternCElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
}
static std::string generateExternCReductionFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cpu, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    float acc = 0.0f;\n";
    src += "    for (long long r = 0; r < " + std::to_string(reduceExtent) + "; ++r) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        acc += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = acc;\n}\n";
    return src;
}
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }
static std::string generateExternCFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered) {
    return isReductionNode(n) ? generateExternCReductionFunction(funcName, lowered)
                               : generateExternCElementwiseFunction(funcName, lowered);
}
static std::string generateJitProgramForGraph(const Graph& g, const std::map<int, Shape>& shapes,
                                               const std::map<int, long long>& elementCounts,
                                               std::vector<std::string>& functionsGenerated) {
    TopoResult topo = topologicalSort(g);
    std::string prog = "#include <vector>\n#include <algorithm>\n\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        prog += generateExternCFunctionForNode(funcName, n, lowered) + "\n";
        functionsGenerated.push_back(n->debugName);
    }
    return prog;
}

// ==================== Section 19's own Isa / vector primitives, FMA fold, and vectorized
//                      reduction (unchanged), now wrapped in extern "C" for the first time ====================

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
static bool canVectorizeElementwise(const Node* n, const std::map<int, long long>& elementCounts) {
    long long outCount = elementCounts.at(n->id);
    for (const Value& in : n->inputs) {
        long long c = elementCounts.at(in.nodeId);
        if (c != outCount && c != 1) return false;
    }
    return true;
}
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
// The ONLY change from Section 19.2's own generateVectorElementwiseFunctionFma():
// the emitted signature now starts with extern "C", exactly the same change
// Section 20.1 made to Chapter 18's own scalar generator.
static std::string generateExternCVectorElementwiseFunctionFma(const std::string& funcName, const LoweredNode& lowered,
                                                                 Isa isa, const std::vector<bool>& isScalarInput,
                                                                 int& foldCount) {
    const std::vector<FusedStep>& steps = lowered.steps;
    std::vector<int> useCounts = computeStepUseCounts(steps, false);
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

    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
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
// The ONLY change from Section 19.3's own generateVectorReductionFunction():
// extern "C" on the signature.
static std::string generateExternCVectorReductionFunction(const std::string& funcName, const LoweredNode& lowered,
                                                            Isa isa, const std::vector<bool>& isScalarInput) {
    const std::vector<FusedStep>& steps = lowered.steps;
    size_t stepCount = steps.size() - 1;
    long long reduceExtent = lowered.nest.loops.back().extent;
    int vw = vectorWidth(isa);

    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
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

// Section 19.3's own three-way dispatcher, now emitting extern "C" bodies
// and never writing a standalone program with its own main() -- every
// function it returns is meant to be dlsym()'d, not run as a process.
static std::string generateJitFunctionForNodeVectorized(const std::string& funcName, const Node* n,
                                                          const LoweredNode& lowered, Isa isa,
                                                          const std::map<int, long long>& elementCounts,
                                                          int& foldCount) {
    foldCount = 0;
    std::vector<bool> isScalarInput;
    for (const Value& in : n->inputs) isScalarInput.push_back(elementCounts.at(in.nodeId) == 1);
    if (isReductionNode(n)) return generateExternCVectorReductionFunction(funcName, lowered, isa, isScalarInput);
    if (canVectorizeElementwise(n, elementCounts))
        return generateExternCVectorElementwiseFunctionFma(funcName, lowered, isa, isScalarInput, foldCount);
    return generateExternCElementwiseFunction(funcName, lowered);  // scalar fallback, same as Section 19.3
}
static std::string generateJitProgramForGraphVectorized(const Graph& g, const std::map<int, Shape>& shapes,
                                                          const std::map<int, long long>& elementCounts, Isa isa,
                                                          std::vector<std::string>& functionsGenerated, int& totalFolds) {
    TopoResult topo = topologicalSort(g);
    std::string prog = std::string(isaHeaderInclude(isa)) + "\n#include <vector>\n#include <algorithm>\n\n";
    totalFolds = 0;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        int foldCount = 0;
        prog += generateJitFunctionForNodeVectorized(funcName, n, lowered, isa, elementCounts, foldCount) + "\n";
        functionsGenerated.push_back(n->debugName);
        totalFolds += foldCount;
    }
    return prog;
}

// ==================== Section 20.1's own JitModule (unchanged) ====================

class JitModule {
public:
    explicit JitModule(const std::string& sharedLibraryPath) {
        handle_ = dlopen(sharedLibraryPath.c_str(), RTLD_NOW);
        if (!handle_) throw std::runtime_error("JitModule: dlopen failed: " + std::string(dlerror()));
    }
    ~JitModule() {
        if (handle_) dlclose(handle_);
    }
    JitModule(const JitModule&) = delete;
    JitModule& operator=(const JitModule&) = delete;
    JitModule(JitModule&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    JitModule& operator=(JitModule&& other) noexcept {
        if (this != &other) {
            if (handle_) dlclose(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    template <typename FnPtr>
    FnPtr getFunction(const std::string& symbolName) const {
        dlerror();
        void* sym = dlsym(handle_, symbolName.c_str());
        const char* err = dlerror();
        if (err) throw std::runtime_error("JitModule::getFunction(\"" + symbolName + "\"): dlsym failed: " + err);
        return reinterpret_cast<FnPtr>(sym);
    }

private:
    void* handle_ = nullptr;
};

// ==================== Shell-out (COMPILATION ONLY) ====================

static int compileInvocationCount = 0;
static int executionSubprocessCount = 0;  // stays 0 for the whole file -- the point of this chapter
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
static bool compileToSharedLibrary(const std::string& source, const std::string& cppPath, const std::string& soPath,
                                    const std::string& extraFlags, std::string& compileLog) {
    writeFile(cppPath, source);
    compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2 -shared -fPIC" + extraFlags + " " +
                                     cppPath + " -o " + soPath + " 2>&1");
    ++compileInvocationCount;
    return compileLog.empty();
}

using ElementwiseFn = void (*)(const std::vector<const float*>&, const std::vector<long long>&, float*, long long);
using ReductionFn = void (*)(const std::vector<const float*>&, const std::vector<long long>&, float*);

// Runs every one of a JIT-compiled graph's own functions, in topological
// order, chained through a shared buffer map -- via REAL FUNCTION POINTERS,
// never a subprocess, never a line of parsed stdout.
static std::map<std::string, std::vector<float>> runJitGraph(const JitModule& module, const Graph& g,
                                                               const std::map<int, long long>& elementCounts,
                                                               const std::vector<std::string>& functionOrder,
                                                               std::map<std::string, std::vector<float>> buf) {
    for (const std::string& name : functionOrder) {
        const Node* n = nullptr;
        for (const auto& nn : g.nodes()) if (nn->debugName == name) n = nn.get();
        bool isReduction = isReductionNode(n);
        std::vector<const float*> ext;
        std::vector<long long> extCounts;
        for (const Value& in : n->inputs) {
            const Node* inNode = g.node(in.nodeId);
            ext.push_back(buf.at(inNode->debugName).data());
            extCounts.push_back(elementCounts.at(in.nodeId));
        }
        long long outCount = elementCounts.at(n->id);
        buf[name] = std::vector<float>(static_cast<size_t>(outCount));
        if (isReduction) {
            ReductionFn fn = module.getFunction<ReductionFn>("compute_" + name);
            fn(ext, extCounts, buf[name].data());
        } else {
            ElementwiseFn fn = module.getFunction<ElementwiseFn>("compute_" + name);
            fn(ext, extCounts, buf[name].data(), outCount);
        }
    }
    return buf;
}

static bool arraysMatch(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    printf("=== Section 20.3: the JIT backend end to end -- no subprocess, no stdout parsing ===\n\n");

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string vectorExtraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string vectorExtraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    // Chapter 16's own 10-node capstone graph, reused verbatim -- the SAME
    // graph Sections 17.3, 18.3, and 19.3 have all already made real, on a
    // CPU loop nest, on a (compile-only) GPU, and vectorized.
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
    printf("fused graph (maxChainLength=3, same cap Section 17.3/18.3/19.3 all used): ");
    for (const auto& n : fused.nodes()) {
        printf("%s", n->debugName.c_str());
        if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
        else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
        printf(" ");
    }
    printf("\n\n");

    bool allOk = true;

    // ---- Backend 1: SCALAR (Chapter 18's own emitSteps() body, extern "C") ----
    printf("--- Backend 1: SCALAR (Chapter 18's own emitSteps() body, JIT-compiled) ---\n\n");
    std::vector<std::string> scalarFns;
    std::string scalarProg = generateJitProgramForGraph(fused, fusedShapes, fusedElementCounts, scalarFns);
    std::string scalarCompileLog;
    bool scalarClean = compileToSharedLibrary(scalarProg, "/tmp/hammer_ch20_050_scalar.cpp",
                                               "/tmp/hammer_ch20_050_scalar.so", "", scalarCompileLog);
    printf("g++ -shared -fPIC compile (%zu functions, scalar): %s\n", scalarFns.size(),
           scalarClean ? "clean" : ("FAILED:\n" + scalarCompileLog).c_str());
    std::map<std::string, std::vector<float>> scalarResult;
    bool scalarLoaded = false;
    {
        JitModule scalarModule("/tmp/hammer_ch20_050_scalar.so");
        scalarLoaded = true;
        std::map<std::string, std::vector<float>> initial = {{"a", aVals}, {"b", bVals}};
        scalarResult = runJitGraph(scalarModule, fused, fusedElementCounts, scalarFns, initial);
    }
    bool scalarMatchesY = arraysMatch(scalarResult.at("y"), origArrays.at("y"));
    printf("compute_y (scalar, real function pointer) = ");
    for (float v : scalarResult.at("y")) printf("%.6f ", v);
    printf("\nself-check: y[0]=%.6f, matches evaluateArrays() (%s)\n\n", scalarResult.at("y")[0],
           (scalarLoaded && scalarMatchesY) ? "confirmed" : "MISMATCH");
    allOk = allOk && scalarClean && scalarLoaded && scalarMatchesY;

    // ---- Backend 2: VECTOR (Chapter 19's own FMA-fold + reduction body, extern "C") ----
    printf("--- Backend 2: VECTOR (Chapter 19's own FMA-fold + reduction body, JIT-compiled, %s) ---\n\n",
           isaName(hostIsa).c_str());
    std::vector<std::string> vectorFns;
    int totalFolds = 0;
    std::string vectorProg = generateJitProgramForGraphVectorized(fused, fusedShapes, fusedElementCounts, hostIsa,
                                                                    vectorFns, totalFolds);
    std::string vectorCompileLog;
    bool vectorClean = compileToSharedLibrary(vectorProg, "/tmp/hammer_ch20_050_vector.cpp",
                                               "/tmp/hammer_ch20_050_vector.so", vectorExtraFlags, vectorCompileLog);
    printf("g++ -shared -fPIC compile (%zu functions, %s, %d FMA fold(s)): %s\n", vectorFns.size(),
           isaName(hostIsa).c_str(), totalFolds, vectorClean ? "clean" : ("FAILED:\n" + vectorCompileLog).c_str());
    std::map<std::string, std::vector<float>> vectorResult;
    bool vectorLoaded = false;
    {
        JitModule vectorModule("/tmp/hammer_ch20_050_vector.so");
        vectorLoaded = true;
        std::map<std::string, std::vector<float>> initial = {{"a", aVals}, {"b", bVals}};
        vectorResult = runJitGraph(vectorModule, fused, fusedElementCounts, vectorFns, initial);
    }
    bool vectorMatchesY = arraysMatch(vectorResult.at("y"), origArrays.at("y"));
    printf("compute_y (vector, real function pointer) = ");
    for (float v : vectorResult.at("y")) printf("%.6f ", v);
    printf("\nself-check: y[0]=%.6f, matches evaluateArrays() (%s), 3 FMA folds applied (%s)\n\n",
           vectorResult.at("y")[0], (vectorLoaded && vectorMatchesY) ? "confirmed" : "MISMATCH",
           totalFolds == 3 ? "confirmed" : "MISMATCH");
    allOk = allOk && vectorClean && vectorLoaded && vectorMatchesY && (totalFolds == 3);

    printf("--- Tally: what this JIT backend needed, versus Chapters 17-19's own harness ---\n\n");
    printf("g++ invocations (COMPILATION -- unavoidable, this book's own JIT never embeds a\n");
    printf("compiler as a library): %d (one per backend, same count Chapter 17-19's own\n", compileInvocationCount);
    printf("harness needed for a chained whole-graph program)\n");
    printf("subprocess launches for EXECUTION: %d -- every one of this section's own %zu real\n",
           executionSubprocessCount, scalarFns.size() + vectorFns.size());
    printf("results (%zu nodes x 2 backends) crossed from generated code into this program as a\n", scalarFns.size());
    printf("genuine C++ function return, not a line of stdout text parsed back (%s)\n",
           executionSubprocessCount == 0 ? "confirmed" : "MISMATCH");

    allOk = allOk && (executionSubprocessCount == 0);
    return allOk ? 0 : 1;
}
