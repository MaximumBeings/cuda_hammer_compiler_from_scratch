// Chapter 17: Lowering CUDA Hammer's IR to Loops
// 039_from_scalars_to_arrays_a_real_per_element_interpreter.cpp
//
// Section 17.1 -- every evaluate() call since Chapter 9 has tracked exactly
// ONE float per node: inputValuesByName supplies a single number per named
// input, standing in for "every element of this tensor holds this value."
// That was a deliberate simplification, not an oversight -- it let Chapters
// 9-16 check correctness cheaply while building the IR and the fusion
// passes. But it means every "evaluate() agreement confirmed" self-check in
// this book so far has only ever proven fusion correct in the DEGENERATE
// case where a tensor is uniform. This section builds evaluateArrays(), a
// real per-element interpreter operating on actual std::vector<float>
// buffers, and uses it to check two things no earlier chapter could: that
// it agrees with the old scalar evaluate() under uniform inputs (continuity
// with everything already proven), and that fusion is still correct when
// every element of a tensor holds a genuinely DIFFERENT value (something no
// earlier chapter's own checks could tell us at all).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 039_from_scalars_to_arrays_a_real_per_element_interpreter.cpp -o 039_from_scalars_to_arrays_a_real_per_element_interpreter
// Run:     ./039_from_scalars_to_arrays_a_real_per_element_interpreter
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

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

// ==================== evaluate() -- the OLD scalar-representative interpreter (from Chapter 14, unchanged) ====================

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName,
                                              const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");
    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) {
            v = inputValuesByName.at(n->debugName);
        } else if (n->op == OpKind::Const) {
            v = n->constValue;
        } else if (n->op == OpKind::Add) {
            v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        } else if (n->op == OpKind::Mul) {
            v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        } else if (n->op == OpKind::ReLU) {
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        } else if (n->op == OpKind::Sum) {
            v = valuesById.at(n->inputs[0].nodeId) * static_cast<float>(elementCounts.at(n->inputs[0].nodeId));
        } else {  // FusedElementwise or FusedReduction
            std::vector<float> externalVals;
            for (const Value& in : n->inputs) externalVals.push_back(valuesById.at(in.nodeId));
            std::vector<float> stepVals;
            for (const FusedStep& step : n->fusedSteps) {
                auto read = [&](const FusedOperand& o) {
                    return (o.kind == OperandKind::ExternalInput) ? externalVals[o.index] : stepVals[o.index];
                };
                float sv;
                if (step.op == OpKind::ReLU) sv = std::max(0.0f, read(step.operands[0]));
                else if (step.op == OpKind::Add) sv = read(step.operands[0]) + read(step.operands[1]);
                else if (step.op == OpKind::Mul) sv = read(step.operands[0]) * read(step.operands[1]);
                else sv = read(step.operands[0]) * static_cast<float>(step.reduceElementCount);  // Sum
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== Section 17.1: evaluateArrays() -- a REAL per-element interpreter ====================
//
// Same structure as evaluate() above, but every value is a std::vector<float>
// (one entry per tensor element) instead of a single float. A Const node's
// own buffer always has exactly one element (a literal is a scalar by
// definition). Every read of an operand's buffer uses (index % buffer.size())
// rather than a bare index -- this is what makes broadcasting work: a
// same-shape operand's own buffer size equals the loop's own element count,
// so index % size == index (no effect); a SCALAR operand's buffer has size 1,
// so index % 1 == 0 always (every iteration reads the same one element); and
// -- the case this book has never needed to handle per-element before -- a
// operand whose shape is a right-aligned SUFFIX of the output's shape (like
// Chapter 6's own b:[4] broadcasting against a:[3,4]) has its buffer read
// with a period equal to its own element count, which is exactly what
// row-major broadcasting requires. This only works because every broadcast
// this book's own graphs ever build broadcasts a TRAILING suffix (or a bare
// scalar) -- never an interior size-1 dimension, which would need a real
// per-dimension stride (0 along the broadcast axis) instead of one flat
// modulo. Stated here, not silently patched: the same kind of explained
// scope limitation as Chapter 10's own cseKey() or Chapter 15's own
// loopNestsCompatibleForFusion().
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

int main() {
    printf("=== Section 17.1: evaluateArrays() -- a real per-element interpreter ===\n\n");

    // Chapter 6's own diamond graph, reused verbatim: a:[3,4] (12 elements),
    // b:[4] (4 elements, broadcasts against a's trailing dimension). t1 has
    // TWO consumers (t2, t3), so it stays external under fusion; t2/t3/out
    // all have exactly one consumer each and fuse into one FusedElementwise.
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

    FusionResult fusedResult = reductionFusionPass(g, {});  // no Sum in this graph -- same as elementwiseFusionPass
    const Graph& fused = fusedResult.graph;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes())
        fusedElementCounts[n->id] = elementCounts.at(fusedResult.representativeOldId.at(n->id));

    printf("Original graph: %zu nodes. Fused graph: %zu nodes (t1 stays external -- shared; t2/t3/out\n", g.size(), fused.size());
    printf("fuse into one FusedElementwise, external inputs t1 and a).\n\n");

    // ---- Part A: UNIFORM inputs -- evaluateArrays() must agree with the OLD scalar evaluate() ----
    printf("--- Part A: uniform inputs (every element the same) -- bridging to everything Ch9-16 already proved ---\n\n");
    std::vector<float> aUniform(12, 2.0f), bUniform(4, 3.0f);
    std::map<std::string, std::vector<float>> uniformIn = {{"a", aUniform}, {"b", bUniform}};
    auto origArraysUniform = evaluateArrays(g, uniformIn, elementCounts);
    auto fusedArraysUniform = evaluateArrays(fused, uniformIn, fusedElementCounts);
    float scalarOut = evaluate(g, {{"a", 2.0f}, {"b", 3.0f}}, {}).at("out");
    printf("evaluate(a=2, b=3).out (OLD, scalar) = %g\n", scalarOut);

    bool everyElementUniformOrig = true, everyElementUniformFused = true, uniformMatchesScalar = true;
    for (float v : origArraysUniform.at("out")) if (v != scalarOut) everyElementUniformOrig = false;
    for (float v : fusedArraysUniform.at("out")) if (v != scalarOut) everyElementUniformFused = false;
    uniformMatchesScalar = everyElementUniformOrig && everyElementUniformFused;
    printf("evaluateArrays(original graph, uniform a=2,b=3).out: all 12 elements = %g? %s\n",
           origArraysUniform.at("out")[0], everyElementUniformOrig ? "yes" : "NO");
    printf("evaluateArrays(fused graph,    uniform a=2,b=3).out: all 12 elements = %g? %s\n",
           fusedArraysUniform.at("out")[0], everyElementUniformFused ? "yes" : "NO");
    printf("self-check: under uniform inputs, evaluateArrays() collapses to exactly what the OLD\n");
    printf("scalar evaluate() always reported, for BOTH graphs (%s)\n", uniformMatchesScalar ? "confirmed" : "MISMATCH");

    // ---- Part B: NON-UNIFORM inputs -- the check no earlier chapter's own tests could make ----
    printf("\n--- Part B: non-uniform inputs (every element genuinely different) ---\n\n");
    std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};   // a is [3,4], row-major
    std::vector<float> bVals = {10, 20, 30, 40};                          // b is [4], broadcasts per row
    std::map<std::string, std::vector<float>> nonUniformIn = {{"a", aVals}, {"b", bVals}};
    auto origArrays = evaluateArrays(g, nonUniformIn, elementCounts);
    auto fusedArrays = evaluateArrays(fused, nonUniformIn, fusedElementCounts);

    // Hand-derivation for two positions, BEFORE looking at program output.
    // Row-major flat index f = row*4 + col. out = t1*a + relu(t1), t1 = a + b[col].
    //   f=0  (row0,col0): a=1,  b=10 -> t1=11              -> out = 11*1  + 11 = 22
    //   f=5  (row1,col1): a=6,  b=20 -> t1=26              -> out = 26*6  + 26 = 182
    //   f=11 (row2,col3): a=12, b=40 -> t1=52              -> out = 52*12 + 52 = 676
    struct Check { long long idx; float expected; };
    std::vector<Check> checks = {{0, 22.0f}, {5, 182.0f}, {11, 676.0f}};
    bool handDerivedOk = true;
    for (const Check& c : checks) {
        float origV = origArrays.at("out")[c.idx];
        if (origV != c.expected) handDerivedOk = false;
        printf("out[%lld]: hand-derived %g, evaluateArrays(original) = %g\n", c.idx, c.expected, origV);
    }
    printf("self-check: hand-derived positions match evaluateArrays() on the ORIGINAL graph (%s)\n",
           handDerivedOk ? "confirmed" : "MISMATCH");

    bool fusionPreservesPerElement = true;
    for (size_t i = 0; i < origArrays.at("out").size(); ++i) {
        if (origArrays.at("out")[i] != fusedArrays.at("out")[i]) fusionPreservesPerElement = false;
    }
    printf("\nself-check: with genuinely DIFFERENT values at every one of the 12 positions, the\n");
    printf("ORIGINAL graph and the FUSED graph agree at EVERY position, not just in aggregate (%s)\n",
           fusionPreservesPerElement ? "confirmed" : "MISMATCH");

    // ---- Part C: a reduction, non-uniform, checked by direct arithmetic ----
    printf("\n--- Part C: a reduction (Sum), non-uniform input, checked by direct arithmetic ---\n\n");
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
    std::map<int, long long> rFusedElementCounts;
    for (const auto& n : rFused.graph.nodes())
        rFusedElementCounts[n->id] = rElementCounts.at(rFused.representativeOldId.at(n->id));

    std::vector<float> xVals = {-3, 5, -1, 2, 0, 4};  // relu -> 0,5,0,2,0,4 -> sum = 11
    std::map<std::string, std::vector<float>> rIn = {{"x", xVals}};
    float sOrig = evaluateArrays(gr, rIn, rElementCounts).at("s")[0];
    float sFused = evaluateArrays(rFused.graph, rIn, rFusedElementCounts).at("s")[0];
    float expectedSum = 0 + 5 + 0 + 2 + 0 + 4;
    printf("x = [-3, 5, -1, 2, 0, 4]  ->  relu(x) = [0, 5, 0, 2, 0, 4]  ->  sum = %g (hand-derived)\n", expectedSum);
    printf("evaluateArrays(original, unfused).s = %g\n", sOrig);
    printf("evaluateArrays(fused: rt1+s -> one FusedReduction).s = %g\n", sFused);
    bool reductionOk = (sOrig == expectedSum) && (sFused == expectedSum);
    printf("self-check: both agree with the hand-derived sum (%s)\n", reductionOk ? "confirmed" : "MISMATCH");

    bool allOk = uniformMatchesScalar && handDerivedOk && fusionPreservesPerElement && reductionOk;
    return allOk ? 0 : 1;
}
