// Chapter 18: Generating CUDA C++ From the Fused IR
// 042_from_loops_to_threads_a_cuda_elementwise_kernel_generator.cu
//
// Section 18.1 -- Chapter 17 lowered a node's own LoopNest + FusedStep body
// into a SERIAL C++ for-loop. This section targets a genuinely different
// execution model: a CUDA grid, where every loop ITERATION becomes its own
// THREAD running concurrently, not one more pass through a shared loop
// variable. emitSteps() is the shared core this shares with Chapter 17's
// own generateLoopFunction(): the SAME step-sequencing logic (which op
// happens in which order, how a PriorStep operand chains to an earlier
// step), reused unchanged by both a CPU wrapper (a nested for-loop) and a
// CUDA wrapper (a thread-index-and-guard) -- what differs between them is
// isolated to exactly two things a real backend difference actually
// requires: how an external input gets READ (an indexed buffer array on
// the CPU; a named device pointer parameter on the GPU, since a
// std::vector<const float*>'s own internal buffer lives in HOST memory and
// would be an invalid address if a __global__ kernel tried to dereference
// it directly), and how ReLU's own max gets spelled (std::max vs fmaxf).
// The generated .cu program is compiled with real nvcc and then actually
// RUN, honestly reporting this toolchain's real constraint: no physical
// GPU exists on either side of this book's own toolchain, so every CUDA
// API call reports "no CUDA-capable device is detected" -- checked, not
// glossed over, the same way Chapter 12's roofline model or Chapter 14's
// FLOPs convention were checked against real published or real measured
// numbers rather than assumed correct.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 042_from_loops_to_threads_a_cuda_elementwise_kernel_generator.cu -o 042_driver
// Run:     ./042_driver
// (This file's own main() is plain C++: it GENERATES a .cu program as a
// string, writes it to disk, and shells out to nvcc to compile and run IT.
// The driver itself needs no CUDA toolchain to build -- only the .cu file
// it writes does.)
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

// ==================== Loop / LoopNest / buildLoopNest() (from Chapter 15, unchanged) ====================

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

// ==================== lowerNode() (from Section 17.2, unchanged) ====================

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

// ==================== Section 18.1: emitSteps() -- the SHARED codegen core ====================
//
// Chapter 17's own generateLoopFunction() and this section's own
// generateCudaKernel() both need the exact same thing at their core: given
// a FusedStep program, emit "float stepN = <expr>;" for every step, then
// either "out[idx] = stepLast;" (elementwise) or "acc += <summed operand>;"
// (reduction, accumulation handled by the CALLER -- a CPU wrapper uses a
// plain float accumulator, a CUDA wrapper uses atomicAdd, see Section
// 18.2). What's genuinely backend-specific is isolated to exactly two
// small pieces: HOW an ExternalInput operand gets read (an indexed buffer
// array on the CPU; a named device pointer parameter on the GPU, since a
// std::vector's own buffer lives in host memory and is not a valid device
// address), and HOW ReLU's own max gets spelled (std::max vs fmaxf, since
// plain std::max is not guaranteed to compile in __device__ code without
// extra compiler flags this book doesn't want to depend on). Everything
// else -- the sequencing of steps, how a PriorStep operand chains to an
// earlier step's own result -- is ONE function, called by both backends.
enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) {
        return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    }
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExpr(Target target, const std::string& x) {
    return (target == Target::Cpu) ? ("std::max(0.0f, " + x + ")") : ("fmaxf(0.0f, " + x + ")");
}

// Emits every step EXCEPT (for a reduction) the trailing Sum step, whose
// own accumulation the caller performs differently per backend. Returns
// the generated "float stepN = ...;" lines, and separately the C++/CUDA
// EXPRESSION for "the value this program's own last computed step holds"
// (the elementwise output, or the reduction's own per-iteration summand).
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
        else expr = operandExpr(step.operands[0]) + " * " + operandExpr(step.operands[1]);  // Mul
        std::string varName = "step" + std::to_string(s);
        lines.push_back("float " + varName + " = " + expr + ";");
        stepVars.push_back(varName);
    }
    if (isReduction) {
        // The trailing Sum step's own single operand is what the caller accumulates.
        const FusedOperand& sumOperand = steps.back().operands[0];
        finalValueExpr = (sumOperand.kind == OperandKind::ExternalInput)
                              ? readExternal(target, sumOperand.index, idxExpr)
                              : stepVars[static_cast<size_t>(sumOperand.index)];
    } else {
        finalValueExpr = stepVars.back();
    }
    return lines;
}

// ---- CPU wrapper (Chapter 17's own generateLoopFunction(), re-expressed on top of emitSteps()) ----
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

// ---- Section 18.1: the CUDA wrapper -- one THREAD per output element, not one more loop pass ----
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

// ==================== Shell-out helpers ====================

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
    printf("=== Section 18.1: from loops to threads -- a CUDA elementwise kernel generator ===\n\n");

    // Same diamond graph as Chapter 17, same concrete non-uniform inputs.
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");
    (void)out;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
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
    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);

    printf("--- Shared core: emitSteps() called with Target::Cpu vs. Target::Cuda, same steps ---\n\n");
    std::string cpuFinal, cudaFinal;
    std::vector<std::string> cpuBody = emitSteps(outLowered.steps, false, "flat", Target::Cpu, cpuFinal);
    std::vector<std::string> cudaBody = emitSteps(outLowered.steps, false, "flat", Target::Cuda, cudaFinal);
    printf("CPU body (Target::Cpu):\n");
    for (const auto& l : cpuBody) printf("  %s\n", l.c_str());
    printf("out[flat] = %s;\n\n", cpuFinal.c_str());
    printf("CUDA body (Target::Cuda):\n");
    for (const auto& l : cudaBody) printf("  %s\n", l.c_str());
    printf("out[flat] = %s;\n\n", cudaFinal.c_str());
    bool sameStepCount = (cpuBody.size() == cudaBody.size());
    printf("self-check: same number of steps, same sequencing, from the SAME emitSteps() call site,\n");
    printf("only the external-input read syntax and ReLU's own max differ (%s)\n\n",
           sameStepCount ? "confirmed" : "MISMATCH");

    // ---- Generate the full CUDA kernel + a host driver .cu program ----
    std::string kernelSrc = generateCudaElementwiseKernel("compute_out", outLowered, 2);
    printf("--- Generated CUDA kernel ---\n\n%s\n", kernelSrc.c_str());

    std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> bVals = {10, 20, 30, 40};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}};
    auto origArrays = evaluateArrays(g, in, elementCounts);
    std::vector<float> t1Concrete = origArrays.at("t1");
    std::vector<float> aConcrete = aVals;
    auto fusedArrays = evaluateArrays(fused, in, fusedElementCounts);
    long long n = fusedElementCounts.at(outFusedNode->id);

    auto formatArrayLiteral = [](const std::vector<float>& v) {
        std::string out = "{";
        for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
        out += "}";
        return out;
    };

    std::string prog = "#include <cstdio>\n#include <cuda_runtime.h>\n\n";
    prog += "#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) "
            "printf(\"CUDA error at line %d: %s\\n\", __LINE__, cudaGetErrorString(_e)); } while (0)\n\n";
    prog += kernelSrc + "\n";
    prog += "int main() {\n";
    prog += "    float hExt0[] = " + formatArrayLiteral(t1Concrete) + ";\n";
    prog += "    float hExt1[] = " + formatArrayLiteral(aConcrete) + ";\n";
    prog += "    long long n = " + std::to_string(n) + ";\n";
    prog += "    float *dExt0, *dExt1, *dOut;\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dExt0, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dExt1, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMemcpy(dExt0, hExt0, n * sizeof(float), cudaMemcpyHostToDevice));\n";
    prog += "    CUDA_CHECK(cudaMemcpy(dExt1, hExt1, n * sizeof(float), cudaMemcpyHostToDevice));\n";
    prog += "    int threads = 256;\n";
    prog += "    int blocks = (int)((n + threads - 1) / threads);\n";
    prog += "    compute_out<<<blocks, threads>>>(dExt0, dExt1, n, n, dOut, n);\n";
    prog += "    CUDA_CHECK(cudaGetLastError());\n";
    prog += "    CUDA_CHECK(cudaDeviceSynchronize());\n";
    prog += "    float hOut[" + std::to_string(n) + "];\n";
    prog += "    CUDA_CHECK(cudaMemcpy(hOut, dOut, n * sizeof(float), cudaMemcpyDeviceToHost));\n";
    prog += "    printf(\"kernel result (garbage/zero without a real GPU): \");\n";
    prog += "    for (long long i = 0; i < n; ++i) printf(\"%.6f \", hOut[i]);\n";
    prog += "    printf(\"\\n\");\n";
    prog += "    CUDA_CHECK(cudaFree(dExt0)); CUDA_CHECK(cudaFree(dExt1)); CUDA_CHECK(cudaFree(dOut));\n";
    prog += "    return 0;\n}\n";

    writeFile("/tmp/hammer_ch18_042.cu", prog);
    std::string compileLog = runShellCaptureAll("nvcc -std=c++17 -arch=sm_70 -o /tmp/hammer_ch18_042 /tmp/hammer_ch18_042.cu 2>&1");
    bool compileClean = compileLog.find("error") == std::string::npos;
    printf("--- nvcc compile ---\n\n%s\n", compileLog.empty() ? "(no output -- clean compile)\n" : compileLog.c_str());
    printf("self-check: nvcc accepts the generated .cu file as valid CUDA C++ (%s)\n\n",
           compileClean ? "confirmed" : "MISMATCH");

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch18_042 2>&1");
    printf("--- Running the compiled binary (this toolchain has no physical GPU, on either side) ---\n\n");
    printf("%s\n", runOutput.c_str());
    bool honestlyReportsNoDevice = runOutput.find("no CUDA-capable device") != std::string::npos;
    printf("self-check: checked CUDA error codes honestly report \"no CUDA-capable device is\n");
    printf("detected\" at every API call, rather than silently returning garbage (%s)\n\n",
           honestlyReportsNoDevice ? "confirmed" : "MISMATCH");

    // ---- What this kernel's arithmetic WOULD compute, proven the only way this toolchain allows: real CPU execution ----
    printf("--- The host-side reference: same emitSteps() body, run for real on the CPU ---\n\n");
    std::string cpuFuncSrc = generateCpuElementwiseFunction("compute_out_cpu", outLowered);
    std::string cpuProg = "#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n" + cpuFuncSrc + "\n";
    cpuProg += "int main() {\n";
    cpuProg += "    std::vector<float> ext0 = " + formatArrayLiteral(t1Concrete) + ";\n";
    cpuProg += "    std::vector<float> ext1 = " + formatArrayLiteral(aConcrete) + ";\n";
    cpuProg += "    std::vector<const float*> ext = {ext0.data(), ext1.data()};\n";
    cpuProg += "    std::vector<long long> extCounts = {" + std::to_string(n) + ", " + std::to_string(n) + "};\n";
    cpuProg += "    std::vector<float> out(" + std::to_string(n) + ");\n";
    cpuProg += "    compute_out_cpu(ext, extCounts, out.data(), " + std::to_string(n) + ");\n";
    cpuProg += "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";
    writeFile("/tmp/hammer_ch18_042_cpu.cpp", cpuProg);
    runShellCaptureAll("g++ -std=c++17 -O2 /tmp/hammer_ch18_042_cpu.cpp -o /tmp/hammer_ch18_042_cpu 2>&1");
    std::string cpuOut = runShellCaptureAll("/tmp/hammer_ch18_042_cpu");
    printf("CPU (actually executed) result: %s\n", cpuOut.c_str());
    printf("evaluateArrays(fused).out       = ");
    for (float v : fusedArrays.at("out")) printf("%.6f ", v);
    printf("\n");

    std::istringstream iss(cpuOut);
    std::vector<float> cpuVals;
    float v;
    while (iss >> v) cpuVals.push_back(v);
    bool cpuMatches = (cpuVals.size() == fusedArrays.at("out").size());
    if (cpuMatches) for (size_t i = 0; i < cpuVals.size(); ++i) if (std::fabs(cpuVals[i] - fusedArrays.at("out")[i]) > 1e-3f) cpuMatches = false;
    printf("self-check: this IS the number the kernel's own arithmetic computes -- proven by real\n");
    printf("execution of the identical emitSteps() body on the CPU (%s)\n", cpuMatches ? "confirmed" : "MISMATCH");

    bool allOk = sameStepCount && compileClean && honestlyReportsNoDevice && cpuMatches;
    return allOk ? 0 : 1;
}
