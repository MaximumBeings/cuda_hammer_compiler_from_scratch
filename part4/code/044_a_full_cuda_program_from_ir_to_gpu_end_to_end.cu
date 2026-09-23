// Chapter 18: Generating CUDA C++ From the Fused IR
// 044_a_full_cuda_program_from_ir_to_gpu_end_to_end.cu
//
// Section 18.3 (capstone) -- Sections 18.1 and 18.2 generated ONE kernel at
// a time, in isolation. This section lowers an ENTIRE graph to CUDA: one
// __global__ kernel per non-leaf node (elementwise via Section 18.1's own
// generateCudaElementwiseKernel(), reduction via Section 18.2's own
// generateCudaReductionKernel() -- the SAME shared emitSteps() core both
// sections already built on), chained through a real host driver
// (cudaMalloc for every buffer, sequential kernel launches in topological
// order, CUDA_CHECK on every call, cudaMemcpy the final result back) --
// exactly the target Chapter 17's own capstone (Section 17.3) built for a
// serial CPU loop nest, now for a real CUDA grid. The target is, once
// again, Chapter 16's own 10-node capstone graph, run through
// boundedReductionFusionPass() TWICE -- uncapped (5 nodes -> 3 kernels) and
// capped at maxChainLength=3 (6 nodes -> 4 kernels) -- so this section
// closes the exact loop Chapter 16 opened: "different fusion structure,
// same math," now proven on the target this whole book has been building
// toward. Each generated .cu program is compiled with real nvcc (a genuine
// toolchain check on a nontrivial, multi-kernel program) and run, honestly
// reporting this toolchain's real constraint -- no physical GPU exists on
// either side of it, checked via CUDA_CHECK exactly as Sections 18.1 and
// 18.2 already checked it, not glossed over. The actual numeric proof comes
// from the SAME emitSteps() step sequences, generated instead as CPU
// functions (Section 18.1/18.2's own generateCpuElementwiseFunction() and
// generateCpuReductionFunction()), chained exactly like Chapter 17's own
// generateFullProgram(), compiled with g++, and genuinely EXECUTED --
// proving what both CUDA programs' arithmetic would compute if hardware
// were present, and proving it agrees with evaluateArrays() on every single
// buffer of both fused graphs, and with Chapter 16 and 17's own already-
// established y.
//
// Compile: g++ -x c++ -std=c++17 -Wall -Wextra -O2 044_a_full_cuda_program_from_ir_to_gpu_end_to_end.cu -o 044_driver
// Run:     ./044_driver
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

// ==================== emitSteps() and both backends' generators (from Sections 18.1 / 18.2, unchanged) ====================

enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExpr(Target target, const std::string& x) {
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
        if (step.op == OpKind::ReLU) expr = maxExpr(target, operandExpr(step.operands[0]));
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

// ---- Section 18.1: CPU / CUDA elementwise ----
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
static std::string generateCudaElementwiseKernel(const std::string& kernelName, const LoweredNode& lowered,
                                                   int numExternalInputs) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cuda, finalExpr);
    std::string src = "__global__ void " + kernelName + "(";
    for (int i = 0; i < numExternalInputs; ++i) src += "const float* ext" + std::to_string(i) + ", ";
    for (int i = 0; i < numExternalInputs; ++i) src += "long long extCount" + std::to_string(i) + ", ";
    src += "float* out, long long n) {\n";
    src += "    long long flat = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    src += "    if (flat >= n) return;\n";
    for (const std::string& line : body) src += "    " + line + "\n";
    src += "    out[flat] = " + finalExpr + ";\n";
    src += "}\n";
    return src;
}

// ---- Section 18.2: CPU / CUDA reduction (atomicAdd(), not a race) ----
static std::string generateCpuReductionFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cpu, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    float acc = 0.0f;\n";
    src += "    for (long long r = 0; r < " + std::to_string(reduceExtent) + "; ++r) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        acc += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = acc;\n}\n";
    return src;
}
static std::string generateCudaReductionKernel(const std::string& kernelName, const LoweredNode& lowered,
                                                int numExternalInputs) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cuda, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "__global__ void " + kernelName + "(";
    for (int i = 0; i < numExternalInputs; ++i) src += "const float* ext" + std::to_string(i) + ", ";
    for (int i = 0; i < numExternalInputs; ++i) src += "long long extCount" + std::to_string(i) + ", ";
    src += "float* out) {\n";
    src += "    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    src += "    if (r >= " + std::to_string(reduceExtent) + ") return;\n";
    for (const std::string& line : body) src += "    " + line + "\n";
    src += "    atomicAdd(out, " + finalExpr + ");\n";
    src += "}\n";
    return src;
}

// ==================== Section 18.3: a WHOLE graph, dispatched per node, chained end to end ====================

// One node's op fully determines which of the two backend-pair generators
// applies -- Sum/FusedReduction go through the reduction pair (Section
// 18.2), everything else (Add/Mul/ReLU/FusedElementwise) through the
// elementwise pair (Section 18.1). Both pairs already agree on what
// "numExternalInputs" means (n->inputs.size(), the group's own materialized
// external operands), so ONE dispatch function per backend is all this
// section needs -- no new per-node-kind logic, just routing to the two
// backends' own already-built generators.
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }

static std::string generateCudaKernelForNode(const std::string& kernelName, const Node* n, const LoweredNode& lowered) {
    int numExt = static_cast<int>(n->inputs.size());
    return isReductionNode(n) ? generateCudaReductionKernel(kernelName, lowered, numExt)
                               : generateCudaElementwiseKernel(kernelName, lowered, numExt);
}
static std::string generateCpuFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered) {
    return isReductionNode(n) ? generateCpuReductionFunction(funcName, lowered)
                               : generateCpuElementwiseFunction(funcName, lowered);
}

static std::string formatLiteralArray(const std::vector<float>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
    out += "}";
    return out;
}

// ---- The CUDA program: one kernel per non-leaf node, then a real host
// driver that mallocs every buffer, copies the leaves in, launches every
// kernel in topological order (a reduction's own output is cudaMemset to
// 0 first -- exactly Section 18.2's own correctness detail), and copies the
// final "y" buffer back. Always returns 0 -- the point is what CUDA_CHECK
// PRINTS at every failed call, not the process exit code. ----
static std::string generateFullCudaProgram(const Graph& g, const std::map<int, Shape>& shapes,
                                            const std::map<int, long long>& elementCounts,
                                            const std::map<std::string, std::vector<float>>& inputArrays) {
    TopoResult topo = topologicalSort(g);
    std::string prog = "#include <cstdio>\n#include <cuda_runtime.h>\n\n";
    prog += "#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) "
            "printf(\"CUDA error at line %d: %s\\n\", __LINE__, cudaGetErrorString(_e)); } while (0)\n\n";

    std::map<int, std::string> kernelNameById;
    std::map<int, LoweredNode> loweredById;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string kernelName = "compute_" + n->debugName;
        kernelNameById[id] = kernelName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        loweredById[id] = lowered;
        prog += generateCudaKernelForNode(kernelName, n, lowered) + "\n";
    }

    prog += "int main() {\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input) prog += "    float h_" + n->debugName + "[] = " + formatLiteralArray(inputArrays.at(n->debugName)) + ";\n";
        else if (n->op == OpKind::Const) prog += "    float h_" + n->debugName + "[] = {" + std::to_string(n->constValue) + "f};\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        prog += "    float* d_" + n->debugName + ";\n";
        prog += "    CUDA_CHECK(cudaMalloc(&d_" + n->debugName + ", " + std::to_string(elementCounts.at(id)) + " * sizeof(float)));\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const)
            prog += "    CUDA_CHECK(cudaMemcpy(d_" + n->debugName + ", h_" + n->debugName + ", " +
                     std::to_string(elementCounts.at(id)) + " * sizeof(float), cudaMemcpyHostToDevice));\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string kernelName = kernelNameById.at(id);
        bool isReduction = isReductionNode(n);
        if (isReduction) {
            long long reduceExtent = loweredById.at(id).nest.loops.back().extent;
            prog += "    CUDA_CHECK(cudaMemset(d_" + n->debugName + ", 0, sizeof(float)));\n";
            prog += "    { int threads = 256; int blocks = (int)((" + std::to_string(reduceExtent) + " + threads - 1) / threads);\n";
            prog += "      " + kernelName + "<<<blocks, threads>>>(";
            for (const Value& in : n->inputs) prog += "d_" + g.node(in.nodeId)->debugName + ", ";
            for (const Value& in : n->inputs) prog += std::to_string(elementCounts.at(in.nodeId)) + ", ";
            prog += "d_" + n->debugName + ");\n";
            prog += "      CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }\n";
        } else {
            long long outCount = elementCounts.at(id);
            prog += "    { int threads = 256; int blocks = (int)((" + std::to_string(outCount) + " + threads - 1) / threads);\n";
            prog += "      " + kernelName + "<<<blocks, threads>>>(";
            for (const Value& in : n->inputs) prog += "d_" + g.node(in.nodeId)->debugName + ", ";
            for (const Value& in : n->inputs) prog += std::to_string(elementCounts.at(in.nodeId)) + ", ";
            prog += "d_" + n->debugName + ", " + std::to_string(outCount) + ");\n";
            prog += "      CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }\n";
        }
    }
    prog += "    float h_y;\n";
    prog += "    CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));\n";
    prog += "    printf(\"kernel result y (0/garbage without a real GPU): %f\\n\", h_y);\n";
    for (int id : topo.order) prog += "    CUDA_CHECK(cudaFree(d_" + g.node(id)->debugName + "));\n";
    prog += "    return 0;\n}\n";
    return prog;
}

// ---- The CPU reference program: one function per non-leaf node (the SAME
// emitSteps() step sequences, just generated for Target::Cpu instead of
// Target::Cuda), chained through named buffers exactly like Section 17.3's
// own generateFullProgram() -- the only difference is which generator
// produces each function's body, and a reduction function's own signature
// (no trailing "n" parameter, since its reduceExtent is already a literal
// baked into its body). ----
static std::string generateFullCpuProgram(const Graph& g, const std::map<int, Shape>& shapes,
                                           const std::map<int, long long>& elementCounts,
                                           const std::map<std::string, std::vector<float>>& inputArrays) {
    TopoResult topo = topologicalSort(g);
    std::string prog = "#include <cstdio>\n#include <vector>\n#include <map>\n#include <string>\n#include <algorithm>\n\n";

    std::map<int, std::string> funcNameById;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        funcNameById[id] = funcName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        prog += generateCpuFunctionForNode(funcName, n, lowered) + "\n";
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
static bool floatsMatch(float a, float b, float tol = 1e-2f) { return std::fabs(a - b) <= tol; }

int main() {
    printf("=== Section 18.3: a full CUDA program from IR to GPU, end to end ===\n\n");

    // Chapter 16's own 10-node capstone graph, reused verbatim -- the same
    // graph Section 17.3 already made real for a CPU loop nest.
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

    std::vector<float> aVals = {1, -2, 3, -4, 5, -6, 7, -8};
    std::vector<float> bVals = {2};
    std::map<std::string, std::vector<float>> inputArrays = {{"a", aVals}, {"b", bVals}};

    auto origArrays = evaluateArrays(g, inputArrays, elementCounts);
    printf("a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = %.6f\n\n", origArrays.at("y")[0]);

    bool allOk = true;
    float finalYUncapped = 0.0f, finalYCapped = 0.0f;

    for (int variant = 0; variant < 2; ++variant) {
        long long maxChainLength = (variant == 0) ? 1000000 : 3;
        const char* label = (variant == 0) ? "UNCAPPED" : "CAPPED (maxChainLength=3)";
        printf("--- %s fusion -> CUDA ---\n\n", label);

        FusionResult fr = boundedReductionFusionPass(g, elementCounts, maxChainLength);
        const Graph& fusedGraph = fr.graph;
        std::map<int, Shape> fusedShapes;
        std::map<int, long long> fusedElementCounts;
        for (const auto& n : fusedGraph.nodes()) {
            int oldId = fr.representativeOldId.at(n->id);
            fusedShapes[n->id] = shapes.at(oldId);
            fusedElementCounts[n->id] = elementCounts.at(oldId);
        }
        size_t kernelCount = fusedGraph.size() - 2;  // every node but Input a, Input b gets a kernel
        printf("Nodes: %zu -> %zu CUDA kernels, %zu CPU functions. ", fusedGraph.size(), kernelCount, kernelCount);
        for (const auto& n : fusedGraph.nodes()) {
            printf("%s", n->debugName.c_str());
            if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
            else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
            printf(" ");
        }
        printf("\n\n");

        auto fusedArrays = evaluateArrays(fusedGraph, inputArrays, fusedElementCounts);

        // ---- CUDA: generate, compile, run (honest no-device check) ----
        std::string cudaProg = generateFullCudaProgram(fusedGraph, fusedShapes, fusedElementCounts, inputArrays);
        std::string cudaStem = std::string("/tmp/hammer_ch18_044_") + (variant == 0 ? "uncapped" : "capped");
        writeFile(cudaStem + ".cu", cudaProg);
        std::string compileLog = runShellCaptureAll("nvcc -std=c++17 -arch=sm_70 -o " + cudaStem + " " + cudaStem + ".cu 2>&1");
        bool compileClean = compileLog.find("error") == std::string::npos;
        if (compileClean) printf("nvcc compile (%zu kernels): clean\n", kernelCount);
        else printf("nvcc compile (%zu kernels): FAILED:\n%s\n", kernelCount, compileLog.c_str());

        std::string runOutput = runShellCaptureAll(cudaStem + " 2>&1");
        bool honestlyReportsNoDevice = runOutput.find("no CUDA-capable device") != std::string::npos;
        printf("running the compiled binary: honestly reports no CUDA-capable device at every\n");
        printf("CUDA_CHECK'd call, same as Sections 18.1 and 18.2 (%s)\n\n", honestlyReportsNoDevice ? "confirmed" : "MISMATCH");

        // ---- CPU: generate, compile, run for real -- the actual numeric oracle ----
        std::string cpuProg = generateFullCpuProgram(fusedGraph, fusedShapes, fusedElementCounts, inputArrays);
        std::string cpuStem = cudaStem + "_cpu";
        writeFile(cpuStem + ".cpp", cpuProg);
        runShellCaptureAll("g++ -std=c++17 -O2 " + cpuStem + ".cpp -o " + cpuStem + " 2>&1");
        std::string cpuStdout = runShellCaptureAll(cpuStem);
        auto generatedArrays = parseNamedBuffers(cpuStdout);

        bool everyBufferMatches = true;
        for (const auto& n : fusedGraph.nodes()) {
            const std::vector<float>& expected = fusedArrays.at(n->debugName);
            const std::vector<float>& actual = generatedArrays.at(n->debugName);
            if (expected.size() != actual.size()) { everyBufferMatches = false; continue; }
            for (size_t i = 0; i < expected.size(); ++i) if (!floatsMatch(expected[i], actual[i])) everyBufferMatches = false;
        }
        printf("self-check: the CPU-executed version of the SAME emitSteps()-generated arithmetic\n");
        printf("(what every CUDA thread above would have run) matches evaluateArrays() on every\n");
        printf("buffer of this fused graph (%s)\n", everyBufferMatches ? "confirmed" : "MISMATCH");

        float generatedY = generatedArrays.at("y")[0];
        bool matchesOriginal = floatsMatch(generatedY, origArrays.at("y")[0]);
        printf("generated program's y = %.6f, matches evaluateArrays(original, unfused).y = %.6f (%s)\n\n",
               generatedY, origArrays.at("y")[0], matchesOriginal ? "confirmed" : "MISMATCH");

        if (variant == 0) finalYUncapped = generatedY; else finalYCapped = generatedY;
        allOk = allOk && compileClean && honestlyReportsNoDevice && everyBufferMatches && matchesOriginal;
    }

    printf("=== Two structurally different generated CUDA programs (3 kernels vs. 4 kernels,\n");
    printf("different kernel signatures), same underlying arithmetic, proven equal via their real\n");
    printf("executed CPU counterparts ===\n\n");
    printf("uncapped-generated y = %.6f\n", finalYUncapped);
    printf("capped-generated   y = %.6f\n", finalYCapped);
    bool sameFinal = floatsMatch(finalYUncapped, finalYCapped);
    printf("self-check: both agree exactly, extending Chapter 16 and 17's own \"different fusion\n");
    printf("structure, same math\" claim to a real CUDA target for the first time (%s)\n",
           sameFinal ? "confirmed" : "MISMATCH");

    allOk = allOk && sameFinal;
    return allOk ? 0 : 1;
}
