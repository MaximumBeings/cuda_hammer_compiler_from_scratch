// Chapter 17: Lowering CUDA Hammer's IR to Loops
// 041_lowering_a_whole_fused_graph.cpp
//
// Section 17.3 (capstone) -- Section 17.2 generated code for ONE node in
// isolation. This section lowers an ENTIRE graph: one generated function per
// non-leaf node, chained together through named buffers in topological
// order, assembled into a single standalone .cpp program, compiled, and
// run. The target is Chapter 16's own 10-node capstone graph, run through
// boundedReductionFusionPass() TWICE -- once uncapped (5 nodes -> fewer,
// larger generated functions) and once capped at maxChainLength=3 (6 nodes
// -> more, smaller generated functions) -- to make Chapter 16's own claim
// real and executable for the first time: two structurally different
// generated programs, built from the same original graph, that still
// compute the exact same final number.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 041_lowering_a_whole_fused_graph.cpp -o 041_lowering_a_whole_fused_graph
// Run:     ./041_lowering_a_whole_fused_graph
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

// ==================== boundedReductionFusionPass() (from Chapter 16, unchanged) ====================

struct FusionResult {
    Graph graph;
    std::map<int, int> representativeOldId;
};

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

// ==================== Loop / LoopNest / buildLoopNest() (from Chapter 15, unchanged) ====================

struct Loop { std::string dimName; long long extent; };
struct LoopNest { std::vector<Loop> loops; };
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    if (n->op == OpKind::Sum) {
        nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    } else if (n->op == OpKind::FusedReduction) {
        nest.loops.push_back(Loop{"reduce", n->fusedSteps.back().reduceElementCount});
    }
    return nest;
}

// ==================== lowerNode() / generateLoopFunction() (from Section 17.2, unchanged) ====================

struct LoweredNode { std::vector<FusedStep> steps; LoopNest nest; };
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
    } else {
        long long count = elementCounts.at(n->inputs[0].nodeId);
        result.steps = {FusedStep{OpKind::Sum, {FusedOperand{OperandKind::ExternalInput, 0}}, count}};
    }
    result.nest = buildLoopNest(n, shapes, elementCounts);
    return result;
}
static std::string generateLoopFunction(const std::string& funcName, const LoweredNode& lowered) {
    const std::vector<FusedStep>& steps = lowered.steps;
    const LoopNest& nest = lowered.nest;
    bool isReduction = !nest.loops.empty() && nest.loops.back().dimName == "reduce";
    size_t outerCount = nest.loops.size() - (isReduction ? 1 : 0);
    std::vector<long long> strides(outerCount);
    {
        long long stride = 1;
        for (int i = static_cast<int>(outerCount) - 1; i >= 0; --i) { strides[i] = stride; stride *= nest.loops[i].extent; }
    }
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    std::string indent = "    ";
    for (size_t i = 0; i < outerCount; ++i) {
        src += indent + "for (long long i" + std::to_string(i) + " = 0; i" + std::to_string(i) +
               " < " + std::to_string(nest.loops[i].extent) + "; ++i" + std::to_string(i) + ") {\n";
        indent += "    ";
    }
    std::string flatExpr = "0";
    for (size_t i = 0; i < outerCount; ++i) flatExpr += " + i" + std::to_string(i) + " * " + std::to_string(strides[i]);

    auto operandExpr = [&](const FusedOperand& o, const std::vector<std::string>& stepVars,
                            const std::string& idxExpr) -> std::string {
        if (o.kind == OperandKind::ExternalInput)
            return "ext[" + std::to_string(o.index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(o.index) + "]]";
        return stepVars[static_cast<size_t>(o.index)];
    };
    auto emitStepExpr = [&](const FusedStep& step, const std::vector<std::string>& stepVars,
                             const std::string& idxExpr) -> std::string {
        if (step.op == OpKind::ReLU) return "std::max(0.0f, " + operandExpr(step.operands[0], stepVars, idxExpr) + ")";
        if (step.op == OpKind::Add)
            return operandExpr(step.operands[0], stepVars, idxExpr) + " + " + operandExpr(step.operands[1], stepVars, idxExpr);
        return operandExpr(step.operands[0], stepVars, idxExpr) + " * " + operandExpr(step.operands[1], stepVars, idxExpr);
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
    for (size_t i = 0; i < outerCount; ++i) { indent = indent.substr(0, indent.size() - 4); src += indent + "}\n"; }
    src += "}\n";
    return src;
}

// ==================== Section 17.3: lowering a WHOLE graph into one program ====================

static std::string formatLiteralArray(const std::vector<float>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
    out += "}";
    return out;
}

// One generated function per non-leaf node (in topo order), chained through
// named buffers in a std::map<std::string, std::vector<float>>. Leaf
// (Input/Const) buffers are filled directly from concrete values baked in
// as literals; every other buffer is produced by calling that node's own
// generated function with pointers into its own inputs' buffers.
static std::string generateFullProgram(const Graph& g, const std::map<int, Shape>& shapes,
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
        prog += generateLoopFunction(funcName, lowered) + "\n";
    }

    prog += "int main() {\n";
    prog += "    std::map<std::string, std::vector<float>> buf;\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input) {
            prog += "    buf[\"" + n->debugName + "\"] = " + formatLiteralArray(inputArrays.at(n->debugName)) + ";\n";
        } else if (n->op == OpKind::Const) {
            prog += "    buf[\"" + n->debugName + "\"] = {" + std::to_string(n->constValue) + "f};\n";
        }
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        prog += "    buf[\"" + n->debugName + "\"] = std::vector<float>(" + std::to_string(elementCounts.at(id)) + ");\n";
        prog += "    {\n        std::vector<const float*> ext; std::vector<long long> extCounts;\n";
        for (const Value& in : n->inputs) {
            const Node* inNode = g.node(in.nodeId);
            prog += "        ext.push_back(buf[\"" + inNode->debugName + "\"].data()); extCounts.push_back(" +
                    std::to_string(elementCounts.at(in.nodeId)) + ");\n";
        }
        prog += "        " + funcNameById.at(id) + "(ext, extCounts, buf[\"" + n->debugName + "\"].data());\n    }\n";
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

static std::map<std::string, std::vector<float>> compileAndRunFullProgram(const std::string& prog,
                                                                            const std::string& fileStem) {
    std::string srcPath = fileStem + ".cpp";
    std::string binPath = fileStem;
    std::ofstream f(srcPath);
    f << prog;
    f.close();
    runShellCaptureStdout("g++ -std=c++17 -O2 " + srcPath + " -o " + binPath + " 2>&1");
    std::string stdoutText = runShellCaptureStdout(binPath);
    return parseNamedBuffers(stdoutText);
}

int main() {
    printf("=== Section 17.3: lowering a WHOLE graph -- Chapter 16's own capstone, now real and executable ===\n\n");

    // Chapter 16's own 10-node graph, reused verbatim.
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
        printf("--- %s fusion ---\n\n", label);

        FusionResult fr = boundedReductionFusionPass(g, elementCounts, maxChainLength);
        const Graph& fusedGraph = fr.graph;
        std::map<int, Shape> fusedShapes;
        std::map<int, long long> fusedElementCounts;
        for (const auto& n : fusedGraph.nodes()) {
            int oldId = fr.representativeOldId.at(n->id);
            fusedShapes[n->id] = shapes.at(oldId);
            fusedElementCounts[n->id] = elementCounts.at(oldId);
        }
        printf("Nodes: %zu. ", fusedGraph.size());
        for (const auto& n : fusedGraph.nodes()) {
            printf("%s", n->debugName.c_str());
            if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
            else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
            printf(" ");
        }
        printf("\n\n");

        auto fusedArrays = evaluateArrays(fusedGraph, inputArrays, fusedElementCounts);

        std::string prog = generateFullProgram(fusedGraph, fusedShapes, fusedElementCounts, inputArrays);
        std::string fileStem = (variant == 0) ? "/tmp/hammer_ch17_full_uncapped" : "/tmp/hammer_ch17_full_capped";
        auto generatedArrays = compileAndRunFullProgram(prog, fileStem);

        bool everyBufferMatches = true;
        for (const auto& n : fusedGraph.nodes()) {
            const std::vector<float>& expected = fusedArrays.at(n->debugName);
            const std::vector<float>& actual = generatedArrays.at(n->debugName);
            if (expected.size() != actual.size()) { everyBufferMatches = false; continue; }
            for (size_t i = 0; i < expected.size(); ++i) {
                if (!floatsMatch(expected[i], actual[i])) everyBufferMatches = false;
            }
        }
        printf("self-check: generated+compiled+run program's EVERY buffer matches evaluateArrays()\n");
        printf("on this same fused graph (%s)\n", everyBufferMatches ? "confirmed" : "MISMATCH");
        printf("generated program's y = %.6f\n\n", generatedArrays.at("y")[0]);

        bool matchesOriginal = floatsMatch(generatedArrays.at("y")[0], origArrays.at("y")[0]);
        printf("self-check: matches evaluateArrays(original, unfused).y = %.6f (%s)\n\n",
               origArrays.at("y")[0], matchesOriginal ? "confirmed" : "MISMATCH");

        if (variant == 0) finalYUncapped = generatedArrays.at("y")[0];
        else finalYCapped = generatedArrays.at("y")[0];

        allOk = allOk && everyBufferMatches && matchesOriginal;
    }

    printf("=== Two structurally different generated programs (different node counts, different\n");
    printf("loop nests, different generated functions), same final number ===\n\n");
    printf("uncapped-generated y = %.6f\n", finalYUncapped);
    printf("capped-generated   y = %.6f\n", finalYCapped);
    bool sameFinal = floatsMatch(finalYUncapped, finalYCapped);
    printf("self-check: both generated, compiled, and EXECUTED programs agree exactly (%s)\n",
           sameFinal ? "confirmed" : "MISMATCH");

    allOk = allOk && sameFinal;
    return allOk ? 0 : 1;
}
