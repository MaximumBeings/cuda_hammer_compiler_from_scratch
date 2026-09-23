# 17. Lowering CUDA Hammer's IR to Loops

**What you will understand:** `evaluateArrays()`, a real per-element interpreter that finally replaces the one-float-per-node simplification every `evaluate()` call since Chapter 9 has relied on; `generateLoopFunction()`, which turns a single node's own `LoopNest` (Chapter 15) and `FusedStep` body (Chapters 13-14) into the TEXT of a real C++ function; and a full lowering pipeline that assembles an entire fused graph into one standalone program, compiles it with `g++`, runs it, and diffs its real output against `evaluateArrays()` -- the first code in this book that is generated, compiled, and executed, rather than counted.

**What you need to know first:** Chapter 13's `FusedStep`/`OperandKind`/`resolveIntoGroup()`, Chapter 14's `OpKind::Sum`/`FusedReduction`/`reductionFusionPass()`, Chapter 15's `Loop`/`LoopNest`/`buildLoopNest()`, and Chapter 16's `boundedReductionFusionPass()` (reused verbatim in Section 17.3).

---

Part 3 measured fusion's payoff in bytes and FLOPs -- real numbers, honestly counted, but never numbers produced by running anything. Every `evaluate()` call since Chapter 9 has tracked exactly ONE float per node: `inputValuesByName["a"] = 2.0` implicitly means "every element of `a` holds this same value," a simplification that made checking fusion correctness cheap while Chapters 9-16 built the IR and the passes, but one that also means every "evaluate() agreement confirmed" self-check in this book so far has only ever proven fusion correct in the degenerate case where a tensor is uniform. This chapter opens Part 4, "Code Generation," and does two things Part 3 never needed to: it builds a real per-element interpreter to check what a uniform-value interpreter structurally cannot, and it turns CUDA Hammer's own fused IR into actual, compiled, executable loop code for the first time -- closing the gap between "this graph's structure is correct" and "this graph computes the right numbers, for real, on a real machine."

```text
FROM COUNTED BYTES (PART 3) TO GENERATED, COMPILED, EXECUTED CODE (THIS CHAPTER):

  +-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+
  | Fused IR (Ch13-16)     | --> | LoopNest + FusedStep   | --> | Generated C++ source   | --> | Compiled (g++), run,  |
  | Node / Graph            |     | body (Ch15's own       |     | (this chapter's own    |     | and diffed against    |
  | (structure only)        |     | buildLoopNest(), one    |     | textual loop backend)  |     | evaluateArrays()       |
  |                         |     | node at a time)         |     |                        |     | (real numbers)         |
  +-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+

  Section 17.1 -- evaluateArrays(): a real per-element interpreter, checked against
                  the OLD scalar evaluate() under uniform inputs, and against
                  genuinely non-uniform inputs no earlier chapter's checks could express

  Section 17.2 -- generateLoopFunction(): ONE node's own LoopNest + FusedStep body,
                  emitted as real C++, compiled, run, and diffed

  Section 17.3 -- an entire fused graph, lowered into one chained, compiled,
                  executed program -- proving Chapter 16's own "different fusion
                  structure, same math" claim with real, running code
```

## 17.1 From Scalars to Arrays: A Real Per-Element Interpreter

### Intuition

Every `evaluate()` call this book has ever made looks like `evaluate(g, {{"a", 2.0f}, {"b", 3.0f}}, elementCounts)` -- one number per input, standing in for the whole tensor. That was never presented as a full interpreter; it was presented, correctly, as a cheap correctness oracle for checking that fusion doesn't change a graph's answer. But "doesn't change the answer" and "doesn't change the answer AT EVERY POSITION" are different claims, and every self-check this book has run so far has only ever tested the first one, because a uniform input makes them indistinguishable: if every element of `a` equals 2.0, then a fused node and its unfused equivalent can disagree at some positions and agree at others while still reporting the exact same single scalar, simply because every position happens to compute the same thing. `evaluateArrays()` removes that blind spot by tracking a real `std::vector<float>` per node -- one entry per tensor element -- so a graph can finally be fed genuinely different values at every position, and fusion's correctness can be checked the way it actually matters: element by element.

### Background

`evaluateArrays()` mirrors `evaluate()`'s own structure almost exactly -- same topological walk, same per-`OpKind` dispatch, same `FusedStep` execution for `FusedElementwise`/`FusedReduction` -- with every scalar promoted to a buffer and every buffer read guarded by `% buffer.size()`. That modulo is what makes broadcasting work without any special-casing: a same-shape operand's buffer size equals the loop's own element count, so `index % size == index` and nothing changes; a scalar operand's buffer has size 1, so `index % 1 == 0` always, reading its one element on every iteration; and an operand whose shape is a right-aligned SUFFIX of the output's shape -- like Chapter 6's own `b:[4]` broadcasting against `a:[3,4]` -- gets read with a period equal to its own element count, which is exactly what row-major broadcasting requires. This is not a hack specific to the scalar case: it is a correct, general consequence of row-major layout whenever an operand's shape is either identical to the output's, a bare scalar, or a trailing suffix of it -- covering every broadcast this book has ever actually built. It is NOT correct for an interior size-1 dimension (a shape like `[3,1]` broadcasting against `[3,4]`, which needs a real per-dimension stride of zero along the broadcast axis, not one flat modulo) -- a scope limitation worth stating plainly, in the same spirit as Chapter 10's own `cseKey()` or Chapter 15's own `loopNestsCompatibleForFusion()`, rather than silently assumed away. None of this book's own test graphs have ever broadcast an interior dimension, so the limitation is real but has never actually been exercised.

```text
evaluate()  (Ch9-16)                              evaluateArrays()  (this section)
one float per node                                one array (vector of float) per node
inputValuesByName["a"] = 2.0                      inputArraysByName["a"] = {1,2,3,...,12}

UNIFORM inputs (a[i] = 2.0 for every i)   -->      evaluateArrays() collapses to exactly
                                                    what evaluate() already reported
                                                    (Part A -- continuity with Ch9-16)

NON-UNIFORM inputs (a[i] all different)   -->      a check evaluate() could never even
                                                    express -- fusion's real per-element
                                                    correctness, not just its aggregate
                                                    correctness (Part B and Part C)

BROADCAST-SAFE READ (every operand):  buffer[index % buffer.size()]
  same shape as output  -> size == count       -> index % count == index   (no effect)
  scalar operand         -> size == 1           -> index % 1 == 0           (always elem 0)
  trailing-suffix shape  -> size == suffix count -> cycles with the right period
```

```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "039_from_scalars_to_arrays_a_real_per_element_interpreter.cpp" -o "039_from_scalars_to_arrays_a_real_per_element_interpreter"
./"039_from_scalars_to_arrays_a_real_per_element_interpreter"
```

**Output:**

```text
=== Section 17.1: evaluateArrays() -- a real per-element interpreter ===

Original graph: 6 nodes. Fused graph: 4 nodes (t1 stays external -- shared; t2/t3/out
fuse into one FusedElementwise, external inputs t1 and a).

--- Part A: uniform inputs (every element the same) -- bridging to everything Ch9-16 already proved ---

evaluate(a=2, b=3).out (OLD, scalar) = 15
evaluateArrays(original graph, uniform a=2,b=3).out: all 12 elements = 15? yes
evaluateArrays(fused graph,    uniform a=2,b=3).out: all 12 elements = 15? yes
self-check: under uniform inputs, evaluateArrays() collapses to exactly what the OLD
scalar evaluate() always reported, for BOTH graphs (confirmed)

--- Part B: non-uniform inputs (every element genuinely different) ---

out[0]: hand-derived 22, evaluateArrays(original) = 22
out[5]: hand-derived 182, evaluateArrays(original) = 182
out[11]: hand-derived 676, evaluateArrays(original) = 676
self-check: hand-derived positions match evaluateArrays() on the ORIGINAL graph (confirmed)

self-check: with genuinely DIFFERENT values at every one of the 12 positions, the
ORIGINAL graph and the FUSED graph agree at EVERY position, not just in aggregate (confirmed)

--- Part C: a reduction (Sum), non-uniform input, checked by direct arithmetic ---

x = [-3, 5, -1, 2, 0, 4]  ->  relu(x) = [0, 5, 0, 2, 0, 4]  ->  sum = 11 (hand-derived)
evaluateArrays(original, unfused).s = 11
evaluateArrays(fused: rt1+s -> one FusedReduction).s = 11
self-check: both agree with the hand-derived sum (confirmed)
```


!!! note "evaluateArrays() adds no new SEMANTICS -- only a new PRECISION"
    Every formula `evaluateArrays()` computes -- Add, Mul, ReLU, Sum, and every `FusedStep` dispatch -- is exactly what `evaluate()` and Chapters 13-14's own passes already defined. Nothing about WHAT fusion computes changes in this section. What changes is the GRANULARITY at which that computation gets checked: from one aggregate number per node to one number per element, closing a real gap in what this book's own correctness checks could see, not a gap in what CUDA Hammer's own passes actually did.

## 17.2 Generating Real Loop Code: A Textual C++ Backend

### Intuition

`evaluateArrays()` computes the right answer, but it computes it by directly *executing* C++ that this book wrote once, ahead of time, for every possible node shape. A real compiler doesn't get to do that: it has to produce NEW code, specific to a given node's own loop nest and step program, that some OTHER process compiles and runs. `generateLoopFunction()` is CUDA Hammer's first attempt at that -- and the honest way to build it is to notice that `evaluateArrays()`'s own per-element loop, for a `FusedElementwise` node, is *already* almost exactly the shape of the C++ this section needs to emit: a loop over a flat index, a straight-line sequence of `stepN = ...` assignments, a final write to `out[flat]`. Section 17.2 doesn't invent a new computational model; it takes the model Section 17.1 already executes directly and emits it as TEXT instead -- a string that, when compiled and run by a real `g++` invocation, computes the exact same numbers.

### Background

`lowerNode()` gives every non-leaf node the same shape for codegen purposes -- a list of `FusedStep`s plus a `LoopNest` -- whether the node arrived already fused (`FusedElementwise`/`FusedReduction`, whose own `fusedSteps` are used as-is) or is a plain, unfused `Add`/`Mul`/`ReLU`/`Sum` (synthesized into a single-step list on the fly, so `generateLoopFunction()` never needs to know which case it is looking at). `generateLoopFunction()` then walks the node's own `LoopNest` (Chapter 15) to emit nested `for` loops with a row-major flat-index expression, and its own `FusedStep` body to emit the per-element (or, for a reduction, per-reduce-step) program inside the innermost loop -- external-input reads go through the same `% extCounts[i]` broadcast-safe indexing Section 17.1 already established, now baked into the GENERATED code's own text rather than executed directly. The generated function is spliced into a full standalone `.cpp` program (buffer setup, a call to the generated function, printed output), written to disk, compiled with a plain `g++` shell-out, and run -- no real JIT machinery yet. That gap is deliberate: Chapter 20, "A JIT Backend: Compiling and Loading Generated Code at Runtime," builds the real thing, loading generated code into THIS process's own address space with `dlopen` instead of launching a separate binary. Proving generated code is correct doesn't yet require avoiding a process launch, so this section doesn't try to.

```text
ONE NODE'S LoopNest + FusedStep body  --generateLoopFunction()-->  REAL C++ TEXT

  LoopNest: [dim0:3, dim1:4]                    loop i0 from 0 up to 3 (exclusive)
  FusedStep body (ext0=t1, ext1=a):                loop i1 from 0 up to 4 (exclusive) {
    step0 = Mul(ext0, ext1)                          flat = 0 + i0*4 + i1*1
    step1 = ReLU(ext0)                                step0 = ext[0][flat % extCounts[0]]
    step2 = Add(step0, step1)                                 * ext[1][flat % extCounts[1]]
    (step2 is this node's own output)                 step1 = max(0.0f, ext[0][flat % extCounts[0]])
                                                        step2 = step0 + step1
                                                        out[flat] = step2
                                                      }

  ...then: write to a .cpp file, `g++ -std=c++17 -O2`, run the binary, read its
  printed stdout back, and diff it against evaluateArrays()'s own computed array
```

```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "040_generating_real_loop_code_a_textual_c++_backend.cpp" -o "040_generating_real_loop_code_a_textual_c++_backend"
./"040_generating_real_loop_code_a_textual_c++_backend"
```

**Output:**

```text
=== Section 17.2: generateLoopFunction() -- real, compiled, executed loop code ===

--- Target 1: t1 = Add(a, b), a plain (unfused) node -- a:[3,4], b:[4] broadcasts ---

Generated C++ for t1:

void compute_t1(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out) {
    for (long long i0 = 0; i0 < 3; ++i0) {
        for (long long i1 = 0; i1 < 4; ++i1) {
            float step0 = ext[0][(0 + i0 * 4 + i1 * 1) % extCounts[0]] + ext[1][(0 + i0 * 4 + i1 * 1) % extCounts[1]];
            out[0 + i0 * 4 + i1 * 1] = step0;
        }
    }
}

evaluateArrays().t1   = 11.000000 22.000000 33.000000 44.000000 15.000000 26.000000 37.000000 48.000000 19.000000 30.000000 41.000000 52.000000 
generated+compiled+run.t1 = 11.000000 22.000000 33.000000 44.000000 15.000000 26.000000 37.000000 48.000000 19.000000 30.000000 41.000000 52.000000 
self-check: generated code matches evaluateArrays(), broadcasting included (confirmed)

--- Target 2: out (FusedElementwise: ext0=t1, ext1=a; 3 internal steps) ---

Generated C++ for out:

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out) {
    for (long long i0 = 0; i0 < 3; ++i0) {
        for (long long i1 = 0; i1 < 4; ++i1) {
            float step0 = ext[0][(0 + i0 * 4 + i1 * 1) % extCounts[0]] * ext[1][(0 + i0 * 4 + i1 * 1) % extCounts[1]];
            float step1 = std::max(0.0f, ext[0][(0 + i0 * 4 + i1 * 1) % extCounts[0]]);
            float step2 = step0 + step1;
            out[0 + i0 * 4 + i1 * 1] = step2;
        }
    }
}

evaluateArrays(fused).out = 22.000000 66.000000 132.000000 220.000000 90.000000 182.000000 296.000000 432.000000 190.000000 330.000000 492.000000 676.000000 
generated+compiled+run.out = 22.000000 66.000000 132.000000 220.000000 90.000000 182.000000 296.000000 432.000000 190.000000 330.000000 492.000000 676.000000 
self-check: generated code for the FUSED node matches evaluateArrays() (confirmed)
```


!!! warning "[COMMON TRAP] a single node's own generated function is not yet a working PROGRAM"
    `generateLoopFunction()` only emits the BODY -- a function taking already-populated buffers and writing an output buffer. Section 17.2's own compile-and-run harness still has to wrap that function in a complete `.cpp` file (includes, a `main()`, concrete input arrays, a print statement) before `g++` has anything it can build. Forgetting this and trying to compile the bare function text alone fails immediately with "no `main` function" -- an easy mistake to make when the generated FUNCTION is the part that feels like the interesting output, but the surrounding PROGRAM is just as necessary for anything to actually run.

## 17.3 Lowering a Whole Fused Graph

### Intuition

Section 17.2 proved `generateLoopFunction()` correct for one node in isolation, with its inputs handed to it as ready-made arrays. A real compiler has to do more: given an entire GRAPH, it has to generate one function per node, then generate the GLUE that calls them in the right order, threading each node's own output buffer into whichever later node reads it. This section builds that glue -- `generateFullProgram()` -- and points it at a graph this book has already used to make a specific claim: Chapter 16's own 10-node capstone, run through `boundedReductionFusionPass()` twice, once uncapped and once with `maxChainLength=3`, producing two graphs with different node counts, different loop nests, and different generated functions. Chapter 16 proved those two structures agree on `evaluate()`'s own single representative number. This section proves something new: that two ACTUAL, SEPARATELY COMPILED, SEPARATELY EXECUTED programs -- built from those two different structures -- report the exact same real number too.

### Background

`generateFullProgram()` walks the fused graph in topological order exactly once for structure (emitting one `generateLoopFunction()`-produced function per non-leaf node) and once more for the driver: a `main()` that fills every `Input` node's own buffer with concrete, non-uniform literal values and every `Const` node's own buffer with its single literal, then calls each generated function in turn, passing pointers into whichever buffers that node's own `inputs` list names -- exactly the same buffer-threading `evaluateArrays()` already does through `buffersById`, just emitted as a sequence of C++ statements instead of executed as a loop. Every node's own final buffer gets printed, so the compiled program's own stdout can be parsed back into a `name -> array` map and diffed, buffer by buffer, against `evaluateArrays()` run on the SAME fused graph with the SAME concrete inputs -- the strongest correctness check this book has built yet, because it no longer checks one representative number or even one node's own array: it checks every element of every node's own buffer, produced by code that was generated, written to disk, compiled by a real `g++` invocation, and actually executed as its own process.

```text
UNCAPPED (5 nodes)                          CAPPED, maxChainLength=3 (6 nodes)

  a       b                                   a       b
  |       |                                   |       |
  +---+---+                                   +---+---+
      |                                           |
  t1..t5 all fuse into                        t1,t2,t3 fuse (SizeCap
  ONE FusedElementwise (t5)                    boundary) into t3;
      |                                        t4,t5 fuse separately (t5)
  +---+---+                                        |
  |       |                                    +---+---+
  s       y2                                   |       |
  (FusedReduction)                             s       y2
  |       |                                    |       |
  +---+---+                                    +---+---+
      |                                            |
      y                                            y

  generateFullProgram() --> DIFFERENT .cpp text, DIFFERENT compiled binary,
  for each graph -- but both, run for real, report the SAME y
```

```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "041_lowering_a_whole_fused_graph.cpp" -o "041_lowering_a_whole_fused_graph"
./"041_lowering_a_whole_fused_graph"
```

**Output:**

```text
=== Section 17.3: lowering a WHOLE graph -- Chapter 16's own capstone, now real and executable ===

a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = 132.000000

--- UNCAPPED fusion ---

Nodes: 5. a b t5(FE,5st) s(FR,1st) y(FE,2st) 

self-check: generated+compiled+run program's EVERY buffer matches evaluateArrays()
on this same fused graph (confirmed)
generated program's y = 132.000000

self-check: matches evaluateArrays(original, unfused).y = 132.000000 (confirmed)

--- CAPPED (maxChainLength=3) fusion ---

Nodes: 6. a b t3(FE,3st) t5(FE,2st) s(FR,1st) y(FE,2st) 

self-check: generated+compiled+run program's EVERY buffer matches evaluateArrays()
on this same fused graph (confirmed)
generated program's y = 132.000000

self-check: matches evaluateArrays(original, unfused).y = 132.000000 (confirmed)

=== Two structurally different generated programs (different node counts, different
loop nests, different generated functions), same final number ===

uncapped-generated y = 132.000000
capped-generated   y = 132.000000
self-check: both generated, compiled, and EXECUTED programs agree exactly (confirmed)
```


!!! note "Part 4 opens: CUDA Hammer's IR now produces real, executable code"
    Every chapter before this one measured CUDA Hammer's own structure -- node counts, bytes moved, FLOPs, loop-nest shapes. This chapter is the first to produce something that runs: `evaluateArrays()` closed a real gap in what this book's own correctness checks could see (per-element, not just aggregate), and `generateLoopFunction()`/`generateFullProgram()` turned the fused IR into real C++ text, compiled by a real `g++` invocation and executed as its own process -- proving, for the first time with actually running code, that Chapter 16's own "different fusion structure, same math" claim holds. This backend is deliberately simple: single-threaded scalar loops, shapes passed at runtime rather than baked in as compile-time constants, and a plain shell-out instead of real JIT loading. Chapter 18, "Generating CUDA C++ From the Fused IR," and Chapter 19, "Generating Vectorized CPU Code," target the exact same `LoopNest`-driven lowering this chapter built, aimed at progressively more sophisticated backends; Chapter 20, "A JIT Backend," replaces this chapter's own shell-out-and-read-stdout harness with real in-process code loading.

## Chapter Summary

This chapter opened Part 4 by turning CUDA Hammer's fused IR into real, executable code for the first time. Section 17.1 built `evaluateArrays()`, a genuine per-element interpreter that finally replaces the one-float-per-node simplification every `evaluate()` call since Chapter 9 has relied on -- checked for continuity against the OLD scalar `evaluate()` under uniform inputs, and against genuinely non-uniform inputs (hand-derived at specific positions before running) that no earlier chapter's own checks could express, on both an elementwise graph (Chapter 6's own diamond) and a reduction graph. Section 17.2 built `generateLoopFunction()`, which turns one node's own `LoopNest` (Chapter 15) and `FusedStep` body (Chapters 13-14) into the TEXT of a real C++ function -- spliced into a standalone program, compiled with a plain `g++` shell-out, run, and diffed against `evaluateArrays()`, on both a plain broadcasting node and a fused multi-step node. Section 17.3 closed the chapter by lowering an ENTIRE graph -- Chapter 16's own 10-node capstone, run through `boundedReductionFusionPass()` both uncapped and capped at `maxChainLength=3` -- into two structurally different, separately compiled, separately executed programs that were proven, by running them for real, to report the exact same final number: `132.0` at `a=[1,-2,3,-4,5,-6,7,-8]`, `b=[2]`, for the original graph, the uncapped-generated program, and the capped-generated program alike.

## Self-Check Questions

1. This chapter claims every earlier "evaluate() agreement confirmed" self-check in this book only ever proved fusion correct in a "degenerate case." What is that case, and why does a uniform input make it impossible to distinguish from full per-element correctness?
2. `evaluateArrays()` reads every operand's buffer as `buffer[index % buffer.size()]`. For which THREE shapes of operand (relative to the output) is this provably correct, and what specific kind of broadcast would it get wrong?
3. `lowerNode()` synthesizes a single-step `FusedStep` list for a plain, unfused `Add`/`Mul`/`ReLU`/`Sum` node. Why does this matter for `generateLoopFunction()`'s own design -- what would have to change about `generateLoopFunction()` if `lowerNode()` didn't do this?
4. Section 17.2's own [COMMON TRAP] warns that a generated function alone isn't a working program. What, specifically, does the compile-and-run harness have to add before `g++` can build anything?
5. Section 17.3's own `generateFullProgram()` walks the fused graph's topological order twice. What is each walk for, and why can't they be combined into one pass?
6. The uncapped and capped generated programs in Section 17.3 have different node counts, different loop nests, and different generated C++ text. What is the ONE thing this section proves stays identical between them, and by what mechanism (not just "they should," but what was actually checked)?
7. This chapter explicitly defers real JIT loading to Chapter 20. What does this chapter's own compile-and-run harness actually do instead, and why is that gap described as deliberate rather than a shortcut?
8. Chapters 18 and 19 are described as targeting "the exact same `LoopNest`-driven lowering this chapter built." What, specifically, carries over unchanged, and what is expected to change?

## Where We Go Next

Part 4 continues from here with two different backends built on the exact same `LoopNest`-driven lowering this chapter established: Chapter 18, "Generating CUDA C++ From the Fused IR," targets real device code (compiled via `nvcc`, verified against a host-side reference, but -- like the sibling CUDA books -- never executed, since this toolchain has no physical GPU on either side); Chapter 19, "Generating Vectorized CPU Code," targets real SIMD intrinsics that this toolchain CAN genuinely compile and run on two different real architectures (AVX2/FMA on the cloud sandbox, NEON on the device), closing the loop this chapter opened with actual hardware-specific execution. Chapter 20, "A JIT Backend," then replaces this chapter's own shell-out-and-read-stdout harness with real in-process code loading via `dlopen`. Apply the Chapter 5-17 depth-level standard (more prose, more diagrams before code) throughout.

## Worked Solutions

1. The degenerate case is a UNIFORM tensor -- every element holding the same value, which is all `evaluate()`'s own single-float-per-node model could ever express. Under a uniform input, a fused node and its unfused equivalent could in principle disagree at SOME positions and agree at others while still reporting the exact same aggregate scalar, simply because every position happens to compute the same thing under uniformity -- the scalar check has no way to tell "genuinely correct at every position" apart from "correct in aggregate, by coincidence of uniform inputs."
2. It is correct when the operand's shape is identical to the output's (buffer size equals loop count, so `index % size == index`, no effect), when the operand is a bare scalar (size 1, so `index % 1 == 0` always), and when the operand's shape is a right-aligned SUFFIX of the output's shape (the buffer cycles with a period equal to its own element count, matching row-major layout). It would get an INTERIOR size-1 dimension wrong -- a shape like `[3,1]` broadcasting against `[3,4]`, which needs a real per-dimension stride of zero along the broadcast axis rather than one flat modulo.
3. `lowerNode()`'s synthesis means `generateLoopFunction()` only ever has to handle ONE shape of input -- a `std::vector<FusedStep>` plus a `LoopNest` -- regardless of whether the node arrived already fused or not. Without it, `generateLoopFunction()` would need two separate code paths: one reading a real `fusedSteps` list, and another handling a plain node's single Add/Mul/ReLU/Sum operation directly, duplicating the entire loop-nest-and-step-emission logic for what is structurally the same problem.
4. It has to add `#include` directives, a `main()` function, concrete literal arrays for every external input the generated function needs, the actual call to the generated function with those arrays' pointers and sizes, and a `printf` statement to report the result -- without all of this, the generated function's own text alone has no entry point, and `g++` fails immediately with "no `main` function."
5. The first walk emits STRUCTURE -- one `generateLoopFunction()`-produced C++ function per non-leaf node. The second walk emits the DRIVER -- buffer declarations, concrete input literals, and the actual sequence of function calls threading each node's own output into whichever later node reads it. They can't be combined into one pass because the driver's own function-call statements need to reference functions that must already be fully defined earlier in the generated file -- the driver code has to come out AFTER every function it calls, which a single interleaved walk emitting both at once cannot guarantee.
6. The one thing proven identical is the FINAL COMPUTED NUMBER (`y = 132.0`), not the code, the node count, or the loop-nest shape, all of which differ. The mechanism was not inspection or assertion: both generated programs' own `.cpp` text was written to disk, compiled with a real `g++` invocation, run as its own separate process, and its printed stdout parsed back and compared against the other's, with a floating-point tolerance -- an actual execution-level check, not a structural one.
7. Instead of real JIT loading, this chapter's own harness writes generated source to a `.cpp` file, shells out to `g++` to compile it into a separate binary, launches that binary as its own process via `popen`, and reads its printed stdout back to parse out the result. This is described as deliberate, not a shortcut, because proving generated code is CORRECT doesn't require avoiding a process launch -- that concern (avoiding the cost and complexity of compiling to a separate binary and launching it) is exactly what Chapter 20's own real JIT backend exists to address, once correctness is already established.
8. `lowerNode()`, `LoweredNode`, and the overall shape of `generateLoopFunction()` -- turning a `LoopNest` plus a `FusedStep` body into nested loops around a per-element or per-reduce-step program -- carry over unchanged, since that mapping is backend-agnostic. What changes is the TARGET LANGUAGE and EXECUTION MODEL inside the innermost loop: Chapter 18 emits CUDA C++ (`__global__` kernels, one CUDA thread per loop iteration instead of a serial `for`), and Chapter 19 emits vectorized CPU code (SIMD intrinsics processing several loop iterations at once), while the LoopNest-to-loop-structure mapping this chapter built stays the same.

---

**Sources cited in this chapter:**

None new. This chapter's own scalar-loop backend and shell-out-based compile-and-run harness are both original to this book, building directly on the LoopNest view Chapter 15 already established and citing no new external source.