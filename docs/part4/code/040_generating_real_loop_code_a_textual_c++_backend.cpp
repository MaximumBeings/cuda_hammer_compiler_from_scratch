// Chapter 17: Lowering CUDA Hammer's IR to Loops
// 040_generating_real_loop_code_a_textual_c++_backend.cpp
//
// Section 17.2 -- the first chapter in this book to generate code that
// could actually run, rather than counting bytes/FLOPs about code that
// hypothetically would. generateLoopFunction() turns ONE node -- its own
// LoopNest from Chapter 15, and its own FusedStep body from Chapters 13-14
// -- into the TEXT of a real C++ function: nested for-loops computing a
// row-major flat index, a straight-line per-element (or per-reduce-step)
// program inside the innermost loop, exactly mirroring what Section 17.1's
// evaluateArrays() already does in C++ itself, just now emitted as a
// STRING instead of executed directly. That string is spliced into a full,
// standalone .cpp program, compiled with g++ via a plain shell-out (no real
// JIT machinery yet -- Chapter 20 builds that), run, and its printed output
// is diffed against evaluateArrays()'s own array, element by element.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 040_generating_real_loop_code_a_textual_c++_backend.cpp -o 040_generating_real_loop_code_a_textual_c++_backend
// Run:     ./040_generating_real_loop_code_a_textual_c++_backend
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

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand {
    OperandKind kind;
    int index;
};

struct FusedStep {
    OpKind op;
    std::vector<FusedOperand> operands;
    long long reduceElementCount = 1;
};

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
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps,
                             const std::string& name) {
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

struct TopoResult {
    std::vector<int> order;
    bool ok = true;
};

static TopoResult topologicalSort(const Graph& g) {
    std::map<int, int> inDegree;
    for (const auto& n : g.nodes()) inDegree[n->id] = static_cast<int>(n->inputs.size());
    std::deque<int> ready;
    for (const auto& n : g.nodes()) {
        if (inDegree[n->id] == 0) ready.push_back(n->id);
    }
    TopoResult result;
    while (!ready.empty()) {
        int id = ready.front();
        ready.pop_front();
        result.order.push_back(id);
        for (const auto& n : g.nodes()) {
            for (const Value& in : n->inputs) {
                if (in.nodeId == id) {
                    if (--inDegree[n->id] == 0) ready.push_back(n->id);
                }
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
        } else if (n->op == OpKind::ReLU) {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        } else {  // Sum
            shapes[id] = Shape{};
        }
    }
    return shapes;
}

// ==================== resolveIntoGroup() / reductionFusionPass() (from Chapter 14, unchanged) ====================

struct FusionResult {
    Graph graph;
    std::map<int, int> representativeOldId;
};

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized,
                                      std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId,
                                      std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) {
        return FusedOperand{OperandKind::PriorStep, stepIt->second};
    }
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
                           (consumers.at(oldId) != 1);
    if (mustBeExternal) {
        auto extIt = externalIndexByOldId.find(oldId);
        if (extIt != externalIndexByOldId.end()) {
            return FusedOperand{OperandKind::ExternalInput, extIt->second};
        }
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

static FusionResult reductionFusionPass(const Graph& g, const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("reductionFusionPass: graph is not acyclic");
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
        if (n->op == OpKind::Sum) {
            std::vector<Value> externalInputs;
            std::map<int, int> externalIndexByOldId;
            std::vector<FusedStep> steps;
            std::map<int, int> stepIndexByOldId;
            FusedOperand summedOperand = resolveIntoGroup(n->inputs[0].nodeId, g, consumers, materialized,
                                                            externalInputs, externalIndexByOldId,
                                                            steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
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
        } else {  // FusedElementwise or FusedReduction
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

// ==================== Section 15.1's Loop / LoopNest / buildLoopNest() (unchanged) ====================

struct Loop {
    std::string dimName;
    long long extent;
};
struct LoopNest {
    std::vector<Loop> loops;
};
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    if (n->op == OpKind::Input || n->op == OpKind::Const) {
        throw std::runtime_error("buildLoopNest: " + n->debugName + " is not computed by a loop");
    }
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) {
        nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    }
    if (n->op == OpKind::Sum) {
        nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    } else if (n->op == OpKind::FusedReduction) {
        nest.loops.push_back(Loop{"reduce", n->fusedSteps.back().reduceElementCount});
    }
    return nest;
}

// ==================== Section 17.2: lowerNode() and generateLoopFunction() ====================
//
// lowerNode() gives every non-leaf node the SAME shape for codegen purposes:
// a list of FusedSteps plus a LoopNest -- whether the node arrived already
// fused (FusedElementwise/FusedReduction, whose own fusedSteps are used
// as-is) or is a plain, unfused Add/Mul/ReLU/Sum (synthesized into a
// single-step list on the fly). generateLoopFunction() below never needs to
// know which case it is looking at.
struct LoweredNode {
    std::vector<FusedStep> steps;
    LoopNest nest;
};
static LoweredNode lowerNode(const Node* n, const std::map<int, Shape>& shapes,
                              const std::map<int, long long>& elementCounts) {
    LoweredNode result;
    if (n->op == OpKind::FusedElementwise || n->op == OpKind::FusedReduction) {
        result.steps = n->fusedSteps;
    } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
        result.steps = {FusedStep{n->op, {FusedOperand{OperandKind::ExternalInput, 0},
                                           FusedOperand{OperandKind::ExternalInput, 1}}}};
    } else if (n->op == OpKind::ReLU) {
        result.steps = {FusedStep{OpKind::ReLU, {FusedOperand{OperandKind::ExternalInput, 0}}}};
    } else {  // Sum
        long long count = elementCounts.at(n->inputs[0].nodeId);
        result.steps = {FusedStep{OpKind::Sum, {FusedOperand{OperandKind::ExternalInput, 0}}, count}};
    }
    result.nest = buildLoopNest(n, shapes, elementCounts);
    return result;
}

// generateLoopFunction() emits ONE C++ function: given `ext` (one pointer
// per external input, in the SAME order lowerNode()'s ExternalInput indices
// use) and `extCounts` (each input's own element count, needed for the
// broadcast-safe `% extCounts[i]` read Section 17.1 already established),
// it fills `out` with this node's own LoopNest worth of results. Shape
// information is passed at RUNTIME (extCounts, loop extents baked as literal
// constants into the loop bounds themselves) rather than specialized further
// at compile time -- a real backend would often bake shapes in as compile
// -time constants too, enabling unrolling and other shape-specific
// optimizations Chapters 18-19 will start to explore; this backend keeps
// that specialization to the loop BOUNDS alone, which is already enough to
// prove the concept.
static std::string generateLoopFunction(const std::string& funcName, const LoweredNode& lowered) {
    const std::vector<FusedStep>& steps = lowered.steps;
    const LoopNest& nest = lowered.nest;
    bool isReduction = !nest.loops.empty() && nest.loops.back().dimName == "reduce";
    size_t outerCount = nest.loops.size() - (isReduction ? 1 : 0);

    std::vector<long long> strides(outerCount);
    {
        long long stride = 1;
        for (int i = static_cast<int>(outerCount) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= nest.loops[i].extent;
        }
    }

    std::string src;
    src += "void " + funcName + "(const std::vector<const float*>& ext, "
           "const std::vector<long long>& extCounts, float* out) {\n";
    std::string indent = "    ";
    for (size_t i = 0; i < outerCount; ++i) {
        src += indent + "for (long long i" + std::to_string(i) + " = 0; i" + std::to_string(i) +
               " < " + std::to_string(nest.loops[i].extent) + "; ++i" + std::to_string(i) + ") {\n";
        indent += "    ";
    }
    std::string flatExpr = "0";
    for (size_t i = 0; i < outerCount; ++i) {
        flatExpr += " + i" + std::to_string(i) + " * " + std::to_string(strides[i]);
    }

    auto operandExpr = [&](const FusedOperand& o, const std::vector<std::string>& stepVars,
                            const std::string& idxExpr) -> std::string {
        if (o.kind == OperandKind::ExternalInput) {
            return "ext[" + std::to_string(o.index) + "][(" + idxExpr + ") % extCounts[" +
                   std::to_string(o.index) + "]]";
        }
        return stepVars[static_cast<size_t>(o.index)];
    };
    auto emitStepExpr = [&](const FusedStep& step, const std::vector<std::string>& stepVars,
                             const std::string& idxExpr) -> std::string {
        if (step.op == OpKind::ReLU) return "std::max(0.0f, " + operandExpr(step.operands[0], stepVars, idxExpr) + ")";
        if (step.op == OpKind::Add)
            return operandExpr(step.operands[0], stepVars, idxExpr) + " + " + operandExpr(step.operands[1], stepVars, idxExpr);
        return operandExpr(step.operands[0], stepVars, idxExpr) + " * " + operandExpr(step.operands[1], stepVars, idxExpr);  // Mul
    };

    if (!isReduction) {
        std::vector<std::string> stepVars;
        for (size_t s = 0; s < steps.size(); ++s) {
            std::string varName = "step" + std::to_string(s);
            src += indent + "float " + varName + " = " + emitStepExpr(steps[s], stepVars, flatExpr) + ";\n";
            stepVars.push_back(varName);
        }
        src += indent + "out[" + flatExpr + "] = " + stepVars.back() + ";\n";
    } else {
        long long reduceExtent = nest.loops.back().extent;
        src += indent + "float acc = 0.0f;\n";
        src += indent + "for (long long r = 0; r < " + std::to_string(reduceExtent) + "; ++r) {\n";
        std::string rindent = indent + "    ";
        std::vector<std::string> stepVars;
        for (size_t s = 0; s + 1 < steps.size(); ++s) {
            std::string varName = "step" + std::to_string(s);
            src += rindent + "float " + varName + " = " + emitStepExpr(steps[s], stepVars, "r") + ";\n";
            stepVars.push_back(varName);
        }
        src += rindent + "acc += " + operandExpr(steps.back().operands[0], stepVars, "r") + ";\n";
        src += indent + "}\n";
        src += indent + "out[" + flatExpr + "] = acc;\n";
    }
    for (size_t i = 0; i < outerCount; ++i) {
        indent = indent.substr(0, indent.size() - 4);
        src += indent + "}\n";
    }
    src += "}\n";
    return src;
}

// ==================== Shell-out compile-and-run harness ====================
//
// No real JIT machinery yet -- this is a plain "write a .cpp, invoke g++,
// run the binary, read stdout" pipeline. Chapter 20 ("A JIT Backend:
// Compiling and Loading Generated Code at Runtime") builds the real thing,
// loading generated code into THIS process's own address space with dlopen
// instead of shelling out to a separate binary. That gap is deliberate, not
// an oversight: proving generated code is correct doesn't yet require
// avoiding a process launch.
static std::string runShellCaptureStdout(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed for: " + cmd);
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    int rc = pclose(pipe);
    if (rc != 0) throw std::runtime_error("shell command failed (" + std::to_string(rc) + "): " + cmd);
    return out;
}
static std::vector<float> parseFloats(const std::string& text) {
    std::vector<float> out;
    std::istringstream iss(text);
    float v;
    while (iss >> v) out.push_back(v);
    return out;
}
static bool arraysMatch(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-3f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}
static std::string formatLiteralArray(const std::vector<float>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
    out += "}";
    return out;
}

// Compiles+runs one generated loop function on given concrete inputs, and
// returns the printed output array.
static std::vector<float> compileAndRun(const std::string& funcSrc, const std::string& funcName,
                                         const std::vector<std::vector<float>>& extArrays,
                                         long long outCount, const std::string& fileStem) {
    std::string prog = "#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n" + funcSrc + "\n";
    prog += "int main() {\n";
    prog += "    std::vector<std::vector<float>> extStorage = {\n";
    for (size_t i = 0; i < extArrays.size(); ++i) {
        prog += "        " + formatLiteralArray(extArrays[i]) + (i + 1 < extArrays.size() ? ",\n" : "\n");
    }
    prog += "    };\n";
    prog += "    std::vector<const float*> ext;\n";
    prog += "    std::vector<long long> extCounts;\n";
    prog += "    for (auto& b : extStorage) { ext.push_back(b.data()); extCounts.push_back((long long)b.size()); }\n";
    prog += "    std::vector<float> out(" + std::to_string(outCount) + ");\n";
    prog += "    " + funcName + "(ext, extCounts, out.data());\n";
    prog += "    for (float v : out) printf(\"%.6f \", v);\n";
    prog += "    printf(\"\\n\");\n";
    prog += "    return 0;\n}\n";

    std::string srcPath = fileStem + ".cpp";
    std::string binPath = fileStem;
    std::ofstream f(srcPath);
    f << prog;
    f.close();

    runShellCaptureStdout("g++ -std=c++17 -O2 " + srcPath + " -o " + binPath + " 2>&1");
    std::string stdoutText = runShellCaptureStdout(binPath);
    return parseFloats(stdoutText);
}

int main() {
    printf("=== Section 17.2: generateLoopFunction() -- real, compiled, executed loop code ===\n\n");

    // Same diamond graph as Section 17.1.
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

    std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> bVals = {10, 20, 30, 40};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}};
    auto origArrays = evaluateArrays(g, in, elementCounts);
    auto fusedArrays = evaluateArrays(fused, in, fusedElementCounts);

    // ---- Target 1: t1 = Add(a, b) -- a PLAIN node, exercising broadcast-safe indexing ----
    printf("--- Target 1: t1 = Add(a, b), a plain (unfused) node -- a:[3,4], b:[4] broadcasts ---\n\n");
    const Node* t1Node = g.node(t1.nodeId);
    LoweredNode t1Lowered = lowerNode(t1Node, shapes, elementCounts);
    std::string t1Src = generateLoopFunction("compute_t1", t1Lowered);
    printf("Generated C++ for t1:\n\n%s\n", t1Src.c_str());

    std::vector<float> t1Generated = compileAndRun(t1Src, "compute_t1", {aVals, bVals},
                                                     elementCounts.at(t1.nodeId), "/tmp/hammer_ch17_t1");
    bool t1Ok = arraysMatch(t1Generated, origArrays.at("t1"));
    printf("evaluateArrays().t1   = ");
    for (float v : origArrays.at("t1")) printf("%.6f ", v);
    printf("\ngenerated+compiled+run.t1 = ");
    for (float v : t1Generated) printf("%.6f ", v);
    printf("\nself-check: generated code matches evaluateArrays(), broadcasting included (%s)\n",
           t1Ok ? "confirmed" : "MISMATCH");

    // ---- Target 2: out = FusedElementwise(t1, a) -- a FUSED node, 3 internal steps ----
    printf("\n--- Target 2: out (FusedElementwise: ext0=t1, ext1=a; 3 internal steps) ---\n\n");
    const Node* outFusedNode = nullptr;
    for (const auto& n : fused.nodes()) if (n->debugName == "out") outFusedNode = n.get();
    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);
    std::string outSrc = generateLoopFunction("compute_out", outLowered);
    printf("Generated C++ for out:\n\n%s\n", outSrc.c_str());

    std::vector<float> t1Concrete = origArrays.at("t1");   // ext0
    std::vector<float> aConcrete = aVals;                  // ext1
    std::vector<float> outGenerated = compileAndRun(outSrc, "compute_out", {t1Concrete, aConcrete},
                                                      fusedElementCounts.at(outFusedNode->id), "/tmp/hammer_ch17_out");
    bool outOk = arraysMatch(outGenerated, fusedArrays.at("out"));
    printf("evaluateArrays(fused).out = ");
    for (float v : fusedArrays.at("out")) printf("%.6f ", v);
    printf("\ngenerated+compiled+run.out = ");
    for (float v : outGenerated) printf("%.6f ", v);
    printf("\nself-check: generated code for the FUSED node matches evaluateArrays() (%s)\n",
           outOk ? "confirmed" : "MISMATCH");

    bool allOk = t1Ok && outOk;
    return allOk ? 0 : 1;
}
