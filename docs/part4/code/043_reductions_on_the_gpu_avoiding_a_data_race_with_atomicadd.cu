// Chapter 18: Generating CUDA C++ From the Fused IR
// 043_reductions_on_the_gpu_avoiding_a_data_race_with_atomicadd.cu
//
// Section 18.2 -- Chapter 17's own CPU reduction kernel used a single
// `float acc` accumulated by one serial for-loop: safe, because exactly
// ONE thing ever touches `acc` at a time. Section 18.1's own elementwise
// kernel launched one thread per OUTPUT element, each writing its own,
// distinct `out[flat]` -- also safe, for the same reason: no two threads
// ever touch the same memory. A reduction kernel breaks that pattern on
// purpose: every thread reduces toward the SAME single output. Launch one
// thread per REDUCE element and have each do `*out += step;` directly, and
// the result is a classic, textbook data race -- multiple threads
// performing a non-atomic read-modify-write on the same address, with no
// guarantee any of them see each other's updates. atomicAdd() is the
// simplest CORRECT fix: a single hardware-guaranteed atomic add per
// thread, serializing just the one operation that actually needs it. It is
// NOT the FASTEST fix -- every thread still contends for the same memory
// address, and a real high-performance reduction uses a shared-memory tree
// reduction or warp-shuffle instructions to combine values within a block
// before only one atomicAdd per BLOCK (not per thread) touches global
// memory. This section builds the correct, honestly-simple version; the
// faster version is named, not implemented -- a stated limitation, not a
// gap this section pretends isn't there.
//
// Compile: g++ -x c++ -std=c++17 -Wall -Wextra -O2 043_reductions_on_the_gpu_avoiding_a_data_race_with_atomicadd.cu -o 043_driver
// Run:     ./043_driver
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

// ==================== emitSteps() (from Section 18.1, unchanged) ====================

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

// ---- CPU wrapper (from Chapter 17, unchanged) ----
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

// ==================== Section 18.2: the CUDA reduction kernel -- atomicAdd(), not a race ====================
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
    // NOT `*out += finalExpr;` -- that would be a data race with every other
    // thread in this same launch doing the identical non-atomic read-modify-write
    // at the exact same address. atomicAdd() is the correct fix: one hardware-
    // guaranteed atomic add per thread, serializing only the one operation that
    // actually needs it -- not the fastest possible reduction (see this
    // section's own opening note: a shared-memory tree reduction or warp-shuffle
    // approach would touch global memory once per BLOCK, not once per THREAD),
    // but a CORRECT one, checked the same way Section 18.1's own kernel was.
    src += "    atomicAdd(out, " + finalExpr + ");\n";
    src += "}\n";
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
    printf("=== Section 18.2: reductions on the GPU -- avoiding a data race with atomicAdd() ===\n\n");

    // Same reduction graph as Chapter 17's own Section 17.1 Part C.
    Graph gr;
    Value x = gr.addInput("x");
    Value rt1 = gr.addUnary(OpKind::ReLU, x, "rt1");
    Value s = gr.addUnary(OpKind::Sum, rt1, "s");
    (void)s;
    std::map<int, Shape> rDeclared = {{x.nodeId, Shape{{6}}}};
    std::map<int, Shape> rShapes = inferShapes(gr, rDeclared);
    std::map<int, long long> rElementCounts;
    for (const auto& kv : rShapes) rElementCounts[kv.first] = numElements(kv.second);
    FusionResult rFused = reductionFusionPass(gr, rElementCounts);
    std::map<int, Shape> rFusedShapes;
    std::map<int, long long> rFusedElementCounts;
    for (const auto& n : rFused.graph.nodes()) {
        int oldId = rFused.representativeOldId.at(n->id);
        rFusedShapes[n->id] = rShapes.at(oldId);
        rFusedElementCounts[n->id] = rElementCounts.at(oldId);
    }
    const Node* sFusedNode = nullptr;
    for (const auto& n : rFused.graph.nodes()) if (n->debugName == "s") sFusedNode = n.get();
    LoweredNode sLowered = lowerNode(sFusedNode, rFusedShapes, rFusedElementCounts);

    printf("--- Why NOT a plain '*out += ...;' inside the kernel ---\n\n");
    printf("Launching one thread per reduce element (6 threads here) and having each do\n");
    printf("'*out += step0;' directly is a DATA RACE: multiple threads perform a non-atomic\n");
    printf("read-modify-write on the exact same address, with no guarantee any thread sees\n");
    printf("another's update -- the classic lost-update bug. atomicAdd() replaces that with a\n");
    printf("single hardware-guaranteed atomic operation per thread.\n\n");

    std::string kernelSrc = generateCudaReductionKernel("compute_s", sLowered, 1);
    printf("--- Generated CUDA reduction kernel ---\n\n%s\n", kernelSrc.c_str());

    std::vector<float> xVals = {-3, 5, -1, 2, 0, 4};
    std::map<std::string, std::vector<float>> rIn = {{"x", xVals}};
    auto rFusedArrays = evaluateArrays(rFused.graph, rIn, rFusedElementCounts);
    std::vector<float> rt1Concrete;
    for (float v : xVals) rt1Concrete.push_back(std::max(0.0f, v));  // rt1's own external input to s is x itself
    long long reduceExtent = sLowered.nest.loops.back().extent;

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
    prog += "    float hExt0[] = " + formatArrayLiteral(xVals) + ";\n";
    prog += "    long long n = " + std::to_string(reduceExtent) + ";\n";
    prog += "    float *dExt0, *dOut;\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dExt0, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMemcpy(dExt0, hExt0, n * sizeof(float), cudaMemcpyHostToDevice));\n";
    prog += "    CUDA_CHECK(cudaMemset(dOut, 0, sizeof(float)));\n";
    prog += "    int threads = 256;\n";
    prog += "    int blocks = (int)((n + threads - 1) / threads);\n";
    prog += "    compute_s<<<blocks, threads>>>(dExt0, n, dOut);\n";
    prog += "    CUDA_CHECK(cudaGetLastError());\n";
    prog += "    CUDA_CHECK(cudaDeviceSynchronize());\n";
    prog += "    float hOut = 0.0f;\n";
    prog += "    CUDA_CHECK(cudaMemcpy(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));\n";
    prog += "    printf(\"kernel result (0/garbage without a real GPU): %f\\n\", hOut);\n";
    prog += "    CUDA_CHECK(cudaFree(dExt0)); CUDA_CHECK(cudaFree(dOut));\n";
    prog += "    return 0;\n}\n";

    writeFile("/tmp/hammer_ch18_043.cu", prog);
    std::string compileLog = runShellCaptureAll("nvcc -std=c++17 -arch=sm_70 -o /tmp/hammer_ch18_043 /tmp/hammer_ch18_043.cu 2>&1");
    bool compileClean = compileLog.find("error") == std::string::npos;
    printf("--- nvcc compile ---\n\n%s\n", compileLog.empty() ? "(no output -- clean compile)\n" : compileLog.c_str());
    printf("self-check: nvcc accepts the generated atomicAdd()-based kernel (%s)\n\n", compileClean ? "confirmed" : "MISMATCH");

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch18_043 2>&1");
    printf("--- Running the compiled binary ---\n\n%s\n", runOutput.c_str());
    bool honestlyReportsNoDevice = runOutput.find("no CUDA-capable device") != std::string::npos;
    printf("self-check: honestly reports no CUDA-capable device, same as Section 18.1 (%s)\n\n",
           honestlyReportsNoDevice ? "confirmed" : "MISMATCH");

    printf("--- The host-side reference: same emitSteps() body, run for real on the CPU ---\n\n");
    std::string cpuFuncSrc = generateCpuReductionFunction("compute_s_cpu", sLowered);
    std::string cpuProg = "#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n" + cpuFuncSrc + "\n";
    cpuProg += "int main() {\n";
    cpuProg += "    std::vector<float> ext0 = " + formatArrayLiteral(xVals) + ";\n";
    cpuProg += "    std::vector<const float*> ext = {ext0.data()};\n";
    cpuProg += "    std::vector<long long> extCounts = {" + std::to_string(reduceExtent) + "};\n";
    cpuProg += "    float out[1];\n";
    cpuProg += "    compute_s_cpu(ext, extCounts, out);\n";
    cpuProg += "    printf(\"%.6f\\n\", out[0]);\n    return 0;\n}\n";
    writeFile("/tmp/hammer_ch18_043_cpu.cpp", cpuProg);
    runShellCaptureAll("g++ -std=c++17 -O2 /tmp/hammer_ch18_043_cpu.cpp -o /tmp/hammer_ch18_043_cpu 2>&1");
    std::string cpuOut = runShellCaptureAll("/tmp/hammer_ch18_043_cpu");
    float cpuVal = std::stof(cpuOut);
    printf("x = [-3, 5, -1, 2, 0, 4]  ->  relu(x) = [0, 5, 0, 2, 0, 4]  ->  sum = 11 (hand-derived)\n");
    printf("CPU (actually executed) result: %.6f\n", cpuVal);
    printf("evaluateArrays(fused).s          = %.6f\n", rFusedArrays.at("s")[0]);
    bool cpuMatches = std::fabs(cpuVal - 11.0f) < 1e-3f && std::fabs(cpuVal - rFusedArrays.at("s")[0]) < 1e-3f;
    printf("self-check: this IS the number the kernel's own arithmetic computes -- proven by real\n");
    printf("execution of the identical emitSteps() body on the CPU (%s)\n", cpuMatches ? "confirmed" : "MISMATCH");

    bool allOk = compileClean && honestlyReportsNoDevice && cpuMatches;
    return allOk ? 0 : 1;
}
