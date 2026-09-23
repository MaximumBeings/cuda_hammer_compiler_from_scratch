# 14. Reduction Fusion

**What you will understand:** `reductionFusionPass()`, the sixth real optimization pass to share Chapter 8's exact `TransformPass` signature unmodified -- the first genuinely SHAPE-CHANGING operator this book's IR has ever needed (`OpKind::Sum`), why a shape change forces a fusion rule Chapter 13's own machinery never had to enforce, and how much of Chapter 13's own infrastructure (byte-counting, `inferShapes()`, the pass-manager contract) keeps working unmodified versus what genuinely has to bend.

**What you need to know first:** Chapter 13's entire IR and pass (`OpKind::FusedElementwise`, `FusedStep`, `resolveIntoGroup()`, `elementwiseFusionPass()`) -- this chapter extends all of it rather than starting over -- plus Chapter 6's `Shape`/`inferShapes()`, Chapter 8's `TransformPass`/`runPasses()`, and Chapter 9's `evaluate()`.

---

Every op Chapter 13's `elementwiseFusionPass()` ever had to fuse was SHAPE-PRESERVING: `Add`, `Mul`, and `ReLU` each produce exactly as many output elements as they consume, one for one. That one-for-one correspondence is what made the `FusedStep` "register, not memory" model correct in the first place -- a `PriorStep` operand could always be read as "the same element, computed one step earlier," because every step in a fused body shared that same element-for-element relationship with the node's own external inputs. A reduction breaks that correspondence on purpose: `OpKind::Sum` takes N elements and produces exactly 1. In one sense that makes it an IDEAL fusion root -- it can absorb an entire upstream elementwise chain and still read each of its own external inputs from memory exactly once, same as any `FusedElementwise` root. In another sense, it is exactly the kind of operator Chapter 13's own machinery was never built to handle safely: once a value has been reduced, there is no longer a per-element correspondence for anything downstream to treat it as "one more element" the way `Add`/`Mul`/`ReLU` could always chain onto each other. This chapter builds the new IR that represents a reduction, the one new rule that keeps fusing it safe, and proves -- by direct, provable comparison -- that the result is a strict generalization of Chapter 13's own pass, not a separate set of rules bolted on beside it.

```text
WHAT CHANGES, ELEMENTWISE-ONLY vs. WITH A REDUCTION:

  Chapter 13's world (Add/Mul/ReLU only):        This chapter's world (+ Sum):

    every op:  N elements in -> N elements out      Sum:  N elements in -> 1 element out
    a PriorStep operand always means                a value ALREADY reduced can never be
    "the same element, one step earlier"             read as "one more element" downstream --
                                                       there is no element-for-element
    -> single-consumer elementwise chains             correspondence left to preserve
       can ALWAYS inline into their consumer
                                                    -> a reduction's own output is ALWAYS
                                                       external to whatever consumes it,
                                                       no matter how many consumers it has
```

## 14.1 A Shape-Changing Operator: Sum, FusedReduction, and reductionFusionPass()

### Intuition

Picture a factory line where every station hands off exactly one part to the next: a part comes in, gets painted, moves on, gets a label, moves on -- one part, the whole way through, and any station could reasonably be combined with its neighbor into a single combined station since each only ever needs "the one part currently in front of it." A reduction is a fundamentally different kind of station: it needs an entire BIN of parts delivered before it can produce its own single output (a weight, a count, a total). It can absolutely be the LAST station on a combined line -- feed it the bin as the line's parts arrive, and it never needs to set anything back down on a shelf in between. But nothing downstream of it can be handed "one part" the way every other station could, because there is no longer a bin -- there's one number. `OpKind::Sum` is this book's version of that reduction station, and `reductionFusionPass()` is what decides, correctly, when it's safe to combine it with the stations feeding it.

### Background

Chapter 13's `resolveIntoGroup()` decided whether to inline a node into its consumer's fused body using two conditions: is it `Input`/`Const` (always external -- a leaf), and does it have exactly one consumer (otherwise, sharing would force duplication or disagreement, Section 13.2's own subject). This chapter adds exactly one more condition, and it is the entire mechanism by which reduction fusion stays safe: **a `Sum` node's own output is always external, regardless of consumer count.** Every other line of `resolveIntoGroup()` is untouched.

That single added condition has a consequence for the pass's own top-level walk, too. Section 13.1's `elementwiseFusionPass()` skipped any `Add`/`Mul`/`ReLU` node with exactly one consumer, trusting that its one consumer would build it via `resolveIntoGroup()`. Applying that same shortcut to `Sum` would be a real bug: a single-consumer `Sum` node would be skipped, and because the new guard above now refuses to let ANY consumer inline it, nobody would ever build it at all -- it would silently vanish from the output graph. `reductionFusionPass()` therefore gives `Sum` its own top-level rule: **it always materializes as its own root**, whatever its consumer count.

A `FusedReduction` node looks structurally identical to a `FusedElementwise` node -- the same `inputs` list of external reads, the same `std::vector<FusedStep>` internal program -- with one difference: its own LAST step is always a `Sum`, and `FusedStep` itself gains one new field, `reduceElementCount`, meaningful only for that step. That count has to come from somewhere: Chapter 6's `inferShapes()` gains its own first new branch since Chapter 6 itself (every op Chapters 7 through 13 added was either shape-preserving or didn't need shape information at all) -- `Sum` maps its input's shape to `Shape{}`, an empty-dims scalar, exactly one element (the empty product, no special case needed). `reductionFusionPass()` looks up that count once, at fusion time, and bakes it directly into the `Sum` step -- so `evaluate()` never has to reconstruct shape information for a value already living inside a fused body's own internal program.

That evaluate()-side detail matters for a second reason, too. This book's `evaluate()` has, since Chapter 9, stood in for a WHOLE tensor with one representative scalar -- always defensible before now, because every op was shape-preserving, so which element that scalar stood for was never ambiguous or load-bearing. `Sum` is the first op whose correct evaluation genuinely depends on HOW MANY elements that one representative scalar stands for. And there's a second, more structural reason to be careful here: Files 027-029's own step dispatch used a bare `else` to mean `Mul`, which was harmless when `Add`/`Mul`/`ReLU` were the only step kinds that could ever appear -- but that same bare `else` would silently misread a `Sum` step as a `Mul` with one missing operand, reading past the end of `step.operands`. This chapter's own `evaluate()` checks every step kind by name instead, closing that hazard off explicitly.

```text
A FusedReduction NODE'S OWN SHAPE (compare Chapter 13's FusedElementwise):

  Node {
    op = FusedReduction
    inputs = [ external value A, external value B, ... ]   -- read from memory ONCE each
    fusedSteps = [
      step0 = SomeOp(operand, operand)     Add/Mul/ReLU, same as FusedElementwise
      ...                                     -- still shape-preserving, still per-element
      stepN-1 = SomeOp(operand, operand)
      stepN   = Sum(operand)               -- ALWAYS the LAST step, ALWAYS shape-changing
                                               carries its own reduceElementCount, baked in
                                               by the pass -- nothing after this step exists,
                                               because there is no per-element value left
    ]
  }
```

```cpp
// Chapter 14: Reduction Fusion
// 030_reduction_a_shape_changing_op_and_why_it_cant_be_inlined_like_the_rest.cpp
//
// Section 14.1 -- every op Chapter 13's elementwiseFusionPass() ever had
// to fuse was SHAPE-PRESERVING: Add, Mul, and ReLU all produce exactly as
// many output elements as they consume, one for one. That is precisely
// what made the FusedStep "register, not memory" model correct -- a
// PriorStep operand could always be read as "the same element, computed
// one step earlier," because every step in the body shared the same
// element-for-element correspondence with the node's own external inputs.
//
// A reduction breaks that correspondence on purpose. OpKind::Sum takes N
// elements and produces exactly 1. That makes it, in one sense, an IDEAL
// fusion root -- it can absorb an entire upstream elementwise chain and
// read each of ITS OWN external inputs from memory exactly once, same as
// any FusedElementwise root. But in another sense it is exactly the
// operator Chapter 13's own machinery was never built to handle: nothing
// downstream can treat a reduction's own output as "one more element" to
// chain further elementwise steps onto, because there is no longer a
// per-element correspondence to preserve. This section builds the new IR
// (OpKind::Sum, OpKind::FusedReduction, and one new field on FusedStep)
// and a new pass, reductionFusionPass(), that respects this boundary --
// while proving, by direct comparison, that it is a strict GENERALIZATION
// of Chapter 13's own pass, not a separate set of rules.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 030_reduction_a_shape_changing_op_and_why_it_cant_be_inlined_like_the_rest.cpp -o 030_reduction_a_shape_changing_op_and_why_it_cant_be_inlined_like_the_rest
// Run:     ./030_reduction_a_shape_changing_op_and_why_it_cant_be_inlined_like_the_rest
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind (from Chapter 13, OpKind extended again) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

// Sum and FusedReduction are this chapter's own two additions -- every
// other kind is exactly Chapter 13's own enum, unchanged.
enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input:            return "Input";
        case OpKind::Const:            return "Const";
        case OpKind::Add:              return "Add";
        case OpKind::Mul:              return "Mul";
        case OpKind::ReLU:             return "ReLU";
        case OpKind::Sum:              return "Sum";
        case OpKind::FusedElementwise: return "FusedElementwise";
        default:                       return "FusedReduction";
    }
}

// ==================== FusedStep (from Chapter 13, extended with ONE new field) ====================

enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand {
    OperandKind kind;
    int index;
};

struct FusedStep {
    OpKind op;                             // Add, Mul, ReLU, or (new) Sum
    std::vector<FusedOperand> operands;    // 1 operand for ReLU and Sum, 2 for Add/Mul
    long long reduceElementCount = 1;      // meaningful ONLY when op == OpKind::Sum: how many
                                            // elements this step's own operand represents. Baked
                                            // in by the PASS at fusion time (Section 14.1 below),
                                            // so evaluate() never has to reconstruct shape
                                            // information for a value already living inside a
                                            // fused body's own internal program.
};

// ==================== Node / Graph (from Chapter 13, Graph gets one new method) ====================

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;          // EXTERNAL inputs when op is one of the two Fused* kinds
    float constValue = 0.0f;            // only meaningful when op == OpKind::Const
    std::vector<FusedStep> fusedSteps;  // only meaningful when op == FusedElementwise or FusedReduction
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
    // New this chapter -- structurally identical to addFusedElementwise(),
    // just tagged with the other OpKind, so printGraph()/evaluate() can
    // tell "this fused body ends in a reduction" from "this one doesn't"
    // without inspecting fusedSteps.back().op every time.
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

// ==================== Shape / broadcastShapes / inferShapes (from Chapter 6, extended) ====================

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
    return n;  // an EMPTY dims vector (a scalar) is the empty product: 1 element, no special case needed
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
// Extended with exactly ONE new branch -- Sum -- the first SHAPE-CHANGING
// op this function has ever had to handle. Every op through Chapter 13
// either took its shape straight from declaredShapes (Input/Const) or
// preserved its operand's own shape (Add/Mul broadcast, ReLU unchanged).
// Sum always produces a scalar, regardless of its operand's own shape.
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
            shapes[id] = Shape{};  // zero dims: a scalar, exactly one element
        }
    }
    return shapes;
}

// ==================== Section 14.1: resolveIntoGroup() (from Chapter 13, ONE new condition) ====================
//
// A node qualifies to be INLINED into whichever single node consumes it
// exactly when: it is Add/Mul/ReLU (not Input/Const, not already fused,
// and -- new this chapter -- NOT itself a reduction), AND it has EXACTLY
// ONE consumer in the ORIGINAL graph. The one new condition below is the
// entire mechanism by which this chapter's own core question --
// "which fusions are still safe once a shape can change?" -- gets
// answered: a Sum node's own output is ALWAYS external to anyone who
// would otherwise inline it, no matter how many consumers it has,
// because there is no longer a per-element correspondence for a PriorStep
// operand to preserve.
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
    // Chapter 13's own two conditions, PLUS one new one: a reduction's
    // own output is never inlined as if it were just another same-shape
    // elementwise step.
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

    // p is Add/Mul/ReLU with exactly one consumer -- inline it.
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

// Chapter 13's own elementwiseFusionPass(), reused completely UNCHANGED --
// not one line different from File 027/028/029's own version, other than
// calling the (now slightly extended) resolveIntoGroup() above. Kept here
// so this section can prove, directly, that reductionFusionPass() below
// is a strict generalization of it.
static FusionResult elementwiseFusionPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("elementwiseFusionPass: graph is not acyclic");

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

// ==================== Section 14.1: reductionFusionPass() ====================
//
// Add/Mul/ReLU handling is byte-for-byte Chapter 13's own logic. The only
// genuinely new branch is OpKind::Sum, and it needs its own top-level
// rule, not just resolveIntoGroup()'s new guard: Chapter 13's own
// "consumers == 1 -> skip, my one consumer will build me" shortcut does
// NOT generalize to Sum. If it did, a single-consumer reduction would be
// silently skipped here -- and, because resolveIntoGroup()'s own new
// guard now refuses to inline a Sum node into anyone else's body either,
// nobody would EVER build it. It would simply vanish from the output
// graph. A reduction must always materialize as its OWN root, regardless
// of how many consumers it has.
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

            // Unlike Section 13.1's "one step -> emit a plain node"
            // shortcut, a reduction is ALWAYS wrapped as FusedReduction,
            // even when nothing was inlined: a plain OpKind::Sum node has
            // nowhere to carry its own reduceElementCount, and evaluate()
            // needs that count to produce a real number. This is a
            // deliberate, explained departure from Section 13.1's own
            // rule, not an oversight.
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }

        // Add/Mul/ReLU -- exactly Chapter 13's own elementwiseFusionPass(), unchanged.
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

// ==================== Debug printer (from Chapter 13, extended for FusedReduction) ====================
//
// Returns a std::string rather than printing directly, so this section's
// own backward-compatibility self-check (main(), below) can compare two
// graphs' printed forms for exact equality, not just eyeball them.
static std::string formatGraph(const Graph& g) {
    std::string out;
    char buf[256];
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise && n->op != OpKind::FusedReduction) {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            out += buf;
            if (n->op == OpKind::Const) {
                snprintf(buf, sizeof(buf), "%g", n->constValue);
                out += buf;
            } else {
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    if (i) out += ", ";
                    snprintf(buf, sizeof(buf), "%%%d", n->inputs[i].nodeId);
                    out += buf;
                }
            }
            out += ")\n";
        } else {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(external inputs: ", n->id, n->debugName.c_str(),
                     opKindStr(n->op).c_str());
            out += buf;
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                snprintf(buf, sizeof(buf), "ext%zu=%%%d", i, n->inputs[i].nodeId);
                out += buf;
            }
            out += ")\n";
            for (size_t s = 0; s < n->fusedSteps.size(); ++s) {
                const FusedStep& step = n->fusedSteps[s];
                snprintf(buf, sizeof(buf), "      step%zu = %s(", s, opKindStr(step.op).c_str());
                out += buf;
                for (size_t i = 0; i < step.operands.size(); ++i) {
                    if (i) out += ", ";
                    const FusedOperand& o = step.operands[i];
                    snprintf(buf, sizeof(buf), "%s%d", o.kind == OperandKind::ExternalInput ? "ext" : "step", o.index);
                    out += buf;
                }
                out += ")";
                if (step.op == OpKind::Sum) {
                    snprintf(buf, sizeof(buf), "  [reduces %lld elements]", step.reduceElementCount);
                    out += buf;
                }
                out += (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output\n" : "\n";
            }
        }
    }
    return out;
}
static void printGraph(const Graph& g) { printf("%s", formatGraph(g).c_str()); }

// ==================== evaluate() (from Chapter 13, extended for Sum/FusedReduction) ====================
//
// This book's evaluate() has, since Chapter 9, stood in for a WHOLE
// tensor with a single representative scalar -- defensible because every
// op through Chapter 13 was shape-preserving, so which element that
// scalar represented was never ambiguous or load-bearing. Sum is the
// first op whose correct evaluation genuinely depends on HOW MANY
// elements that one representative scalar stands for -- the first time
// this book's evaluate() has needed shape information at all. elementCounts
// supplies it, but ONLY for a plain, unfused Sum node -- inside a
// FusedReduction, the count was already baked into the step itself
// (reduceElementCount) when the pass ran, so no shape lookup is needed
// there at all.
//
// This version's own step dispatch is also written defensively where
// Chapter 13's was not: Files 027-029 used a bare `else` to mean Mul,
// which was harmless there since Add/Mul/ReLU were the only step kinds
// that could ever appear. That same bare `else` would have silently
// misread a Sum step as a Mul with one missing operand -- exactly the
// kind of undefined-behavior-shaped hazard resolveIntoGroup()'s own new
// guard (above) exists to make structurally impossible. This version
// checks every step kind by name instead, closing that hazard off
// explicitly rather than relying on it never coming up.
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
        } else {  // FusedElementwise or FusedReduction: run the internal program in a local register array
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

int main() {
    printf("=== Section 14.1: OpKind::Sum, FusedReduction, and reductionFusionPass() ===\n\n");

    // x = Input([4]); t1 = ReLU(x); s = Sum(t1) -- s is the graph's own
    // designated output. t1 has exactly ONE consumer (s), same shape of
    // situation as every Section 13.1 test case -- nothing here forces a
    // SHARING boundary. The new question is different: can t1
    // (shape-preserving) still inline into s (shape-CHANGING) the same
    // way it would inline into a plain Add/Mul/ReLU consumer?
    Graph g;
    Value x  = g.addInput("x");
    Value t1 = g.addUnary(OpKind::ReLU, x, "t1");
    Value s  = g.addUnary(OpKind::Sum, t1, "s");
    (void)s;

    std::map<int, Shape> declared = {{x.nodeId, Shape{{4}}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    printf("Original graph (%zu nodes), x declared as shape [4]:\n", g.size());
    printGraph(g);

    FusionResult result = reductionFusionPass(g, elementCounts);
    const Graph& fused = result.graph;

    printf("\nAfter reductionFusionPass() (%zu nodes):\n", fused.size());
    printGraph(fused);

    bool nodeCountDropped = (fused.size() == 2 && g.size() == 3);
    printf("\nself-check: node count dropped from %zu to %zu -- t1 inlines into s's own\n", g.size(), fused.size());
    printf("FusedReduction body even though s CHANGES shape (%s)\n", nodeCountDropped ? "confirmed" : "MISMATCH");

    const Node* fusedNode = nullptr;
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::FusedReduction) fusedNode = n.get();
    }
    bool fusedNodeExists = (fusedNode != nullptr);
    bool oneExternalInput = fusedNodeExists && fusedNode->inputs.size() == 1;
    bool twoSteps = fusedNodeExists && fusedNode->fusedSteps.size() == 2;
    bool lastStepIsSum = fusedNodeExists && !fusedNode->fusedSteps.empty() &&
                          fusedNode->fusedSteps.back().op == OpKind::Sum;
    printf("self-check: exactly one FusedReduction node was created, with 1 external input (x)\n");
    printf("and 2 internal steps (relu, sum), the LAST of which is the Sum (%s)\n",
           (fusedNodeExists && oneExternalInput && twoSteps && lastStepIsSum) ? "confirmed" : "MISMATCH");

    std::map<std::string, float> in1 = {{"x", 2.0f}};
    float orig1 = evaluate(g, in1, elementCounts).at("s");
    float fused1 = evaluate(fused, in1, {}).at("s");  // fused graph needs NO elementCounts --
                                                        // reduceElementCount is already baked into the step
    printf("\nself-check: evaluate(original, x=2).s = %g, evaluate(fused, x=2).s = %g (%s)\n",
           orig1, fused1, (orig1 == fused1) ? "confirmed" : "MISMATCH");

    std::map<std::string, float> in2 = {{"x", -3.0f}};
    float orig2 = evaluate(g, in2, elementCounts).at("s");
    float fused2 = evaluate(fused, in2, {}).at("s");
    printf("self-check: evaluate(original, x=-3).s = %g, evaluate(fused, x=-3).s = %g (%s)\n",
           orig2, fused2, (orig2 == fused2) ? "confirmed" : "MISMATCH");

    printf("\n=== Backward compatibility: reductionFusionPass() on a graph with NO reduction ===\n\n");

    // Chapter 13's own File 027 test graph, rebuilt verbatim -- no Sum
    // node anywhere in it.
    Graph chain;
    Value cx   = chain.addInput("x");
    Value cc2  = chain.addConst(2.0f, "c2");
    Value ccN3 = chain.addConst(-3.0f, "cN3");
    Value ct1  = chain.addBinary(OpKind::Mul, cx, cc2, "t1");
    Value ct2  = chain.addBinary(OpKind::Add, ct1, ccN3, "t2");
    Value ct3  = chain.addUnary(OpKind::ReLU, ct2, "t3");
    (void)ct3;

    Graph viaElementwise = elementwiseFusionPass(chain).graph;
    Graph viaReduction = reductionFusionPass(chain, {}).graph;  // elementCounts never consulted -- no Sum node exists

    printf("elementwiseFusionPass() output:\n%s", formatGraph(viaElementwise).c_str());
    printf("\nreductionFusionPass() output on the SAME Sum-free graph:\n%s", formatGraph(viaReduction).c_str());

    bool sameStructure = (formatGraph(viaElementwise) == formatGraph(viaReduction));
    printf("\nself-check: on a graph with no Sum node at all, reductionFusionPass() produces output\n");
    printf("TEXTUALLY IDENTICAL to Chapter 13's own elementwiseFusionPass() -- a strict\n");
    printf("generalization, not a separate pass with separate rules (%s)\n", sameStructure ? "confirmed" : "MISMATCH");

    bool allOk = nodeCountDropped && fusedNodeExists && oneExternalInput && twoSteps && lastStepIsSum &&
                 (orig1 == fused1) && (orig2 == fused2) && sameStructure;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 030_reduction_a_shape_changing_op_and_why_it_cant_be_inlined_like_the_rest.cpp -o 030_reduction_a_shape_changing_op_and_why_it_cant_be_inlined_like_the_rest
./030_reduction_a_shape_changing_op_and_why_it_cant_be_inlined_like_the_rest
```

**Output:**

```text
=== Section 14.1: OpKind::Sum, FusedReduction, and reductionFusionPass() ===

Original graph (3 nodes), x declared as shape [4]:
  %0 x = Input()
  %1 t1 = ReLU(%0)
  %2 s = Sum(%1)

After reductionFusionPass() (2 nodes):
  %0 x = Input()
  %1 s = FusedReduction(external inputs: ext0=%0)
      step0 = ReLU(ext0)
      step1 = Sum(step0)  [reduces 4 elements]  <- this node's own output

self-check: node count dropped from 3 to 2 -- t1 inlines into s's own
FusedReduction body even though s CHANGES shape (confirmed)
self-check: exactly one FusedReduction node was created, with 1 external input (x)
and 2 internal steps (relu, sum), the LAST of which is the Sum (confirmed)

self-check: evaluate(original, x=2).s = 8, evaluate(fused, x=2).s = 8 (confirmed)
self-check: evaluate(original, x=-3).s = 0, evaluate(fused, x=-3).s = 0 (confirmed)

=== Backward compatibility: reductionFusionPass() on a graph with NO reduction ===

elementwiseFusionPass() output:
  %0 x = Input()
  %1 c2 = Const(2)
  %2 cN3 = Const(-3)
  %3 t3 = FusedElementwise(external inputs: ext0=%0, ext1=%1, ext2=%2)
      step0 = Mul(ext0, ext1)
      step1 = Add(step0, ext2)
      step2 = ReLU(step1)  <- this node's own output

reductionFusionPass() output on the SAME Sum-free graph:
  %0 x = Input()
  %1 c2 = Const(2)
  %2 cN3 = Const(-3)
  %3 t3 = FusedElementwise(external inputs: ext0=%0, ext1=%1, ext2=%2)
      step0 = Mul(ext0, ext1)
      step1 = Add(step0, ext2)
      step2 = ReLU(step1)  <- this node's own output

self-check: on a graph with no Sum node at all, reductionFusionPass() produces output
TEXTUALLY IDENTICAL to Chapter 13's own elementwiseFusionPass() -- a strict
generalization, not a separate pass with separate rules (confirmed)
```

!!! note "Why reductionFusionPass() is a separate pass, not a modified elementwiseFusionPass()"
    This chapter keeps Chapter 13's own `elementwiseFusionPass()` byte-for-byte intact in this file (reused verbatim, as the backward-compatibility check above just proved) rather than editing it in place, so that any code elsewhere in this book still relying on it keeps working unmodified. `reductionFusionPass()` is a genuine superset -- its Add/Mul/ReLU branch is identical -- available as an upgrade wherever a graph might contain a reduction, not a replacement forced on graphs that never will.

## 14.2 The Reduction Boundary: What a Reduction's Output Forces Downstream

### Intuition

Section 13.2 showed that a SHARED value -- read by more than one consumer -- forces a fusion boundary, because duplicating its computation into two different kernels risks the two copies disagreeing. This section shows a second, completely different reason a node can be forced to stand alone: even with exactly ONE consumer, a reduction's own output can never be inlined into that consumer, because there is no per-element correspondence left for a `PriorStep` operand to preserve. Two different reasons, two different rules -- and, on a graph where both apply to different nodes at once, both are respected correctly, with no interaction bugs between them.

### Background

`resolveIntoGroup()`'s new guard (Section 14.1) is what enforces the second rule; the sharing rule from Chapter 13 is untouched. This section builds two graphs to show both rules operating, separately and together. Graph A isolates the NEW rule alone: a reduction's result feeds exactly one consumer, and that consumer still cannot inline it -- purely because the reduction changed shape, with no sharing anywhere in sight. Graph B puts both rules to work in the same graph at once: one value is shared (Chapter 13's own rule keeps it external), and a different value is a reduction's own output (this chapter's new rule keeps IT external too) -- and, because every node in that particular graph ends up a boundary for one reason or another, the pass correctly produces NO fusion at all. That is not a missed opportunity; it is the honest, correct answer for a graph whose own structure offers no safe fusion.

```text
GRAPH A (one reason to stand alone)         GRAPH B (two DIFFERENT reasons, same graph)

  x            a                              x              w
  |            |                              |              |
  ReLU (t1)    |    -- 1 consumer: s          ReLU (t1) ------+-- 2 consumers: s AND y2
  |            |       inlines into s          |              |    (SHARING -- Ch13's rule)
  Sum (s) -----+                               Sum (s)         |
  |            |    -- 1 consumer: y           |     -- 1 consumer: y
  |            |       CANNOT inline           |        CANNOT inline (REDUCTION -- Ch14's rule)
  Add (y)                                      Add (y)     Mul (y2)

  result: t1 fuses INTO s;                     result: NOTHING fuses -- t1 stays a boundary
  s stands alone (shape changed);              for ONE reason, s stands alone for a DIFFERENT
  y stays plain (s is external)                reason, y and y2 both stay plain
```

```cpp
// Chapter 14: Reduction Fusion
// 031_the_reduction_boundary_what_a_reductions_output_forces_downstream.cpp
//
// Section 14.2 -- Section 14.1 built the MECHANISM (resolveIntoGroup()'s
// new guard, reductionFusionPass()'s own always-materialize-Sum rule).
// This section proves what that mechanism actually FORCES, on two graphs
// that together cover both ways a node can end up standing alone in this
// chapter's model:
//
//   Graph A: a reduction's OWN result is consumed further downstream, by
//   something that would otherwise be free to fuse. Its single consumer
//   cannot inline it -- not because of sharing (Chapter 13's own original
//   rule), but purely because the reduction CHANGED SHAPE.
//
//   Graph B: a value feeds BOTH a reduction AND something else -- Chapter
//   13's original sharing rule and this chapter's new shape-changing rule
//   apply to two DIFFERENT nodes in the SAME graph at once, and both are
//   still correctly respected together, with zero interaction bugs
//   between them.
//
// Same pass, unmodified between the two graphs -- two different real
// outcomes, because the graphs' own structures differ. Exactly Section
// 13.2's own framing, one shape-changing operator later.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 031_the_reduction_boundary_what_a_reductions_output_forces_downstream.cpp -o 031_the_reduction_boundary_what_a_reductions_output_forces_downstream
// Run:     ./031_the_reduction_boundary_what_a_reductions_output_forces_downstream
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from File 030, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input:            return "Input";
        case OpKind::Const:            return "Const";
        case OpKind::Add:              return "Add";
        case OpKind::Mul:              return "Mul";
        case OpKind::ReLU:             return "ReLU";
        case OpKind::Sum:              return "Sum";
        case OpKind::FusedElementwise: return "FusedElementwise";
        default:                       return "FusedReduction";
    }
}

enum class OperandKind { ExternalInput, PriorStep };
struct FusedOperand { OperandKind kind; int index; };
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
        for (const auto& n : g.nodes())
            for (const Value& in : n->inputs)
                if (in.nodeId == id)
                    if (--inDegree[n->id] == 0) ready.push_back(n->id);
    }
    result.ok = (result.order.size() == g.size());
    return result;
}

// ==================== Shape / broadcastShapes / inferShapes (from File 030, unchanged) ====================

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
        if (n->op == OpKind::Input || n->op == OpKind::Const) shapes[id] = declaredShapes.at(id);
        else if (n->op == OpKind::Add || n->op == OpKind::Mul)
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        else if (n->op == OpKind::ReLU) shapes[id] = shapes.at(n->inputs[0].nodeId);
        else shapes[id] = Shape{};  // Sum
    }
    return shapes;
}

// ==================== resolveIntoGroup() / reductionFusionPass() (from File 030, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized,
                                      std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId,
                                      std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
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

// ==================== printGraph() / evaluate() (from File 030, unchanged) ====================

static std::string formatGraph(const Graph& g) {
    std::string out;
    char buf[256];
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise && n->op != OpKind::FusedReduction) {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            out += buf;
            if (n->op == OpKind::Const) {
                snprintf(buf, sizeof(buf), "%g", n->constValue);
                out += buf;
            } else {
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    if (i) out += ", ";
                    snprintf(buf, sizeof(buf), "%%%d", n->inputs[i].nodeId);
                    out += buf;
                }
            }
            out += ")\n";
        } else {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(external inputs: ", n->id, n->debugName.c_str(),
                     opKindStr(n->op).c_str());
            out += buf;
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                snprintf(buf, sizeof(buf), "ext%zu=%%%d", i, n->inputs[i].nodeId);
                out += buf;
            }
            out += ")\n";
            for (size_t s = 0; s < n->fusedSteps.size(); ++s) {
                const FusedStep& step = n->fusedSteps[s];
                snprintf(buf, sizeof(buf), "      step%zu = %s(", s, opKindStr(step.op).c_str());
                out += buf;
                for (size_t i = 0; i < step.operands.size(); ++i) {
                    if (i) out += ", ";
                    const FusedOperand& o = step.operands[i];
                    snprintf(buf, sizeof(buf), "%s%d", o.kind == OperandKind::ExternalInput ? "ext" : "step", o.index);
                    out += buf;
                }
                out += ")";
                if (step.op == OpKind::Sum) {
                    snprintf(buf, sizeof(buf), "  [reduces %lld elements]", step.reduceElementCount);
                    out += buf;
                }
                out += (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output\n" : "\n";
            }
        }
    }
    return out;
}
static void printGraph(const Graph& g) { printf("%s", formatGraph(g).c_str()); }

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName,
                                              const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");
    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) v = inputValuesByName.at(n->debugName);
        else if (n->op == OpKind::Const) v = n->constValue;
        else if (n->op == OpKind::Add) v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::Mul) v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::ReLU) v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        else if (n->op == OpKind::Sum) v = valuesById.at(n->inputs[0].nodeId) * static_cast<float>(elementCounts.at(n->inputs[0].nodeId));
        else {
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
                else sv = read(step.operands[0]) * static_cast<float>(step.reduceElementCount);
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

int main() {
    printf("=== Section 14.2: what a reduction's OWN output forces downstream ===\n\n");

    printf("--- Graph A: s = Sum(relu(x)) feeds y = Add(s, a) -- s has exactly ONE consumer ---\n\n");

    // x, a = Input; t1 = ReLU(x) [1 consumer: s]; s = Sum(t1) [1 consumer: y];
    // y = Add(s, a) [0 consumers -- the graph's own output].
    Graph gA;
    Value ax = gA.addInput("x");
    Value aa = gA.addInput("a");
    Value at1 = gA.addUnary(OpKind::ReLU, ax, "t1");
    Value as = gA.addUnary(OpKind::Sum, at1, "s");
    Value ay = gA.addBinary(OpKind::Add, as, aa, "y");
    (void)ay;

    std::map<int, Shape> declaredA = {{ax.nodeId, Shape{{4}}}, {aa.nodeId, Shape{}}};
    std::map<int, Shape> shapesA = inferShapes(gA, declaredA);
    std::map<int, long long> countsA;
    for (const auto& kv : shapesA) countsA[kv.first] = numElements(kv.second);

    printf("Original graph (%zu nodes):\n", gA.size());
    printGraph(gA);

    FusionResult resultA = reductionFusionPass(gA, countsA);
    const Graph& fusedA = resultA.graph;
    printf("\nAfter reductionFusionPass() (%zu nodes):\n", fusedA.size());
    printGraph(fusedA);

    bool aNodeCountDropped = (gA.size() == 5 && fusedA.size() == 4);
    const Node* aFusedNode = nullptr;
    for (const auto& n : fusedA.nodes()) if (n->op == OpKind::FusedReduction) aFusedNode = n.get();
    bool aReductionAbsorbedT1 = aFusedNode && aFusedNode->fusedSteps.size() == 2;
    const Node* aYNode = nullptr;
    for (const auto& n : fusedA.nodes()) if (n->debugName == "y") aYNode = n.get();
    bool aYStaysPlain = aYNode && aYNode->op == OpKind::Add && aYNode->fusedSteps.empty();
    printf("\nself-check: 5 nodes -> 4 -- t1 absorbed into s's own FusedReduction body (2 steps),\n");
    printf("s itself stands alone as a boundary (its consumer y cannot inline it, purely because\n");
    printf("s CHANGED SHAPE, not because of sharing -- s has only ONE consumer here) (%s)\n",
           (aNodeCountDropped && aReductionAbsorbedT1 && aYStaysPlain) ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inA = {{"x", 2.0f}, {"a", 1.0f}};
    float origA = evaluate(gA, inA, countsA).at("y");
    float fusedAOut = evaluate(fusedA, inA, {}).at("y");
    printf("self-check: evaluate(original, x=2,a=1).y = %g, evaluate(fused, ...).y = %g (%s)\n",
           origA, fusedAOut, (origA == fusedAOut) ? "confirmed" : "MISMATCH");

    printf("\n--- Graph B: t1 feeds BOTH a reduction and something else -- two DIFFERENT reasons\n");
    printf("    to stay a boundary, in the SAME graph, at the SAME time ---\n\n");

    // x, w = Input; t1 = ReLU(x) [2 consumers: s AND y2]; s = Sum(t1)
    // [1 consumer: y]; y = Add(s, w) [0 consumers]; y2 = Mul(t1, w)
    // [0 consumers]. t1 stays a boundary because it is SHARED (Chapter
    // 13's own original rule); s stays a boundary because it is a
    // REDUCTION (this chapter's new rule) -- even though s itself has
    // only one consumer, same as Graph A's s.
    Graph gB;
    Value bx = gB.addInput("x");
    Value bw = gB.addInput("w");
    Value bt1 = gB.addUnary(OpKind::ReLU, bx, "t1");
    Value bs = gB.addUnary(OpKind::Sum, bt1, "s");
    Value by = gB.addBinary(OpKind::Add, bs, bw, "y");
    Value by2 = gB.addBinary(OpKind::Mul, bt1, bw, "y2");
    (void)by; (void)by2;

    std::map<int, Shape> declaredB = {{bx.nodeId, Shape{{4}}}, {bw.nodeId, Shape{}}};
    std::map<int, Shape> shapesB = inferShapes(gB, declaredB);
    std::map<int, long long> countsB;
    for (const auto& kv : shapesB) countsB[kv.first] = numElements(kv.second);

    printf("Original graph (%zu nodes):\n", gB.size());
    printGraph(gB);

    FusionResult resultB = reductionFusionPass(gB, countsB);
    const Graph& fusedB = resultB.graph;
    printf("\nAfter reductionFusionPass() (%zu nodes):\n", fusedB.size());
    printGraph(fusedB);

    bool bNodeCountUnchanged = (gB.size() == 6 && fusedB.size() == 6);
    const Node* bT1Node = nullptr;
    const Node* bSNode = nullptr;
    for (const auto& n : fusedB.nodes()) {
        if (n->debugName == "t1") bT1Node = n.get();
        if (n->debugName == "s") bSNode = n.get();
    }
    bool bT1StaysPlain = bT1Node && bT1Node->op == OpKind::ReLU && bT1Node->fusedSteps.empty();
    bool bSIsTrivialFusedReduction = bSNode && bSNode->op == OpKind::FusedReduction && bSNode->fusedSteps.size() == 1;
    printf("\nself-check: 6 nodes -> 6 -- NO fusion happens at all here, and that is the CORRECT\n");
    printf("answer: t1 stays plain (shared -- 2 consumers), s stays its own (trivial, 1-step)\n");
    printf("FusedReduction (shape-changing -- always wrapped even alone, per Section 14.1's own\n");
    printf("stated rule), y and y2 both stay plain (each only 1 step once their own shared/reduced\n");
    printf("operand is external) (%s)\n", (bNodeCountUnchanged && bT1StaysPlain && bSIsTrivialFusedReduction) ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inB = {{"x", 2.0f}, {"w", 3.0f}};
    float origBY = evaluate(gB, inB, countsB).at("y");
    float fusedBY = evaluate(fusedB, inB, {}).at("y");
    float origBY2 = evaluate(gB, inB, countsB).at("y2");
    float fusedBY2 = evaluate(fusedB, inB, {}).at("y2");
    printf("self-check: evaluate(original, x=2,w=3).y = %g, evaluate(fused, ...).y = %g (%s)\n",
           origBY, fusedBY, (origBY == fusedBY) ? "confirmed" : "MISMATCH");
    printf("self-check: evaluate(original, x=2,w=3).y2 = %g, evaluate(fused, ...).y2 = %g (%s)\n",
           origBY2, fusedBY2, (origBY2 == fusedBY2) ? "confirmed" : "MISMATCH");

    bool allOk = aNodeCountDropped && aReductionAbsorbedT1 && aYStaysPlain && (origA == fusedAOut) &&
                 bNodeCountUnchanged && bT1StaysPlain && bSIsTrivialFusedReduction &&
                 (origBY == fusedBY) && (origBY2 == fusedBY2);
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 031_the_reduction_boundary_what_a_reductions_output_forces_downstream.cpp -o 031_the_reduction_boundary_what_a_reductions_output_forces_downstream
./031_the_reduction_boundary_what_a_reductions_output_forces_downstream
```

**Output:**

```text
=== Section 14.2: what a reduction's OWN output forces downstream ===

--- Graph A: s = Sum(relu(x)) feeds y = Add(s, a) -- s has exactly ONE consumer ---

Original graph (5 nodes):
  %0 x = Input()
  %1 a = Input()
  %2 t1 = ReLU(%0)
  %3 s = Sum(%2)
  %4 y = Add(%3, %1)

After reductionFusionPass() (4 nodes):
  %0 x = Input()
  %1 a = Input()
  %2 s = FusedReduction(external inputs: ext0=%0)
      step0 = ReLU(ext0)
      step1 = Sum(step0)  [reduces 4 elements]  <- this node's own output
  %3 y = Add(%2, %1)

self-check: 5 nodes -> 4 -- t1 absorbed into s's own FusedReduction body (2 steps),
s itself stands alone as a boundary (its consumer y cannot inline it, purely because
s CHANGED SHAPE, not because of sharing -- s has only ONE consumer here) (confirmed)
self-check: evaluate(original, x=2,a=1).y = 9, evaluate(fused, ...).y = 9 (confirmed)

--- Graph B: t1 feeds BOTH a reduction and something else -- two DIFFERENT reasons
    to stay a boundary, in the SAME graph, at the SAME time ---

Original graph (6 nodes):
  %0 x = Input()
  %1 w = Input()
  %2 t1 = ReLU(%0)
  %3 s = Sum(%2)
  %4 y = Add(%3, %1)
  %5 y2 = Mul(%2, %1)

After reductionFusionPass() (6 nodes):
  %0 x = Input()
  %1 w = Input()
  %2 t1 = ReLU(%0)
  %3 s = FusedReduction(external inputs: ext0=%2)
      step0 = Sum(ext0)  [reduces 4 elements]  <- this node's own output
  %4 y2 = Mul(%2, %1)
  %5 y = Add(%3, %1)

self-check: 6 nodes -> 6 -- NO fusion happens at all here, and that is the CORRECT
answer: t1 stays plain (shared -- 2 consumers), s stays its own (trivial, 1-step)
FusedReduction (shape-changing -- always wrapped even alone, per Section 14.1's own
stated rule), y and y2 both stay plain (each only 1 step once their own shared/reduced
operand is external) (confirmed)
self-check: evaluate(original, x=2,w=3).y = 11, evaluate(fused, ...).y = 11 (confirmed)
self-check: evaluate(original, x=2,w=3).y2 = 6, evaluate(fused, ...).y2 = 6 (confirmed)
```

!!! note "A graph that fuses NOTHING is a correct output, not a failure"
    Graph B's own pass run produces 6 nodes both before and after -- every single node is a boundary for one reason or another. It would be a mistake to read that as the pass "not working." The pass's job is to fuse exactly what is SAFE to fuse; a graph whose own structure happens to put a boundary in front of every node is a graph that offers no safe fusion, and reporting that honestly (rather than forcing a fusion that would be unsafe) is the correct behavior, the same discipline Chapter 10's own conservative `add(a,b)` vs. `add(b,a)` treatment already established.

## 14.3 Real Savings: Reduction Fusion Through Chapter 8's PassManager, Measured in Bytes

### Intuition

Sections 14.1 and 14.2 established WHEN reduction fusion is safe. This section closes the chapter the way Chapter 13's own Section 13.3 did: plugging the real pass into Chapter 8's real, unmodified `PassManager`, and measuring REAL bytes moved on a real graph -- this time a clean chain ending in a reduction, with no sharing anywhere, so nothing stops it from fusing all the way down to a single kernel.

### Background

Two real wrinkles surface here that Section 13.3 never had to face. First: `reductionFusionPass()` needs `elementCounts` (to bake `reduceElementCount` into each `Sum` step), but Chapter 8's `TransformPass` contract is exactly `Graph(const Graph&)` -- no room for a second argument. The fix is a closure that captures one specific graph's own precomputed element counts, rather than changing that contract -- a real, concrete instance of a pass gaining a genuine new external dependency the interface was never designed to carry. Second: Chapter 12's own `totalFlops()` used the convention "one FLOP per output element," an exact fit for `Add`/`Mul`/`ReLU`, where output element count IS operation count -- but wrong for `Sum`, where reducing N elements to 1 takes N-1 additions, not 1. Using the old convention unmodified would silently claim that summing a million elements costs one FLOP. This section's own `totalFlops()` fixes that with one new branch.

What does NOT need to change is the more interesting result: Chapter 13's own `bytesMoved()` function -- which reads every non-leaf node's own `inputs` and adds its own single output -- works on `FusedReduction` nodes with zero modifications, because Section 14.1's one-line `inferShapes()` extension (`Sum -> Shape{}`) already gives a `FusedReduction` node's own output the correct element count (1) through the exact same `representativeOldId` side channel Chapter 13 already built. Good original design pays dividends the moment the next feature arrives.

```text
BEFORE vs. AFTER, ON A CLEAN CHAIN ENDING IN A REDUCTION (a=[3,4], b=[4], no sharing):

  BEFORE (3 separate kernels -- t1=Add, t2=ReLU, s=Sum):     260 bytes
  AFTER reductionFusionPass() (t1/t2/s fuse into ONE
         FusedReduction kernel -- nothing stops full fusion,
         since nothing in this chain is shared):              68 bytes

              68 vs 260   (no idealized-vs-real gap to explain this time --
                            nothing here is shared, so nothing forces a
                            boundary, and the pass reaches the maximum
                            savings this graph's own shape allows)
```

```cpp
// Chapter 14: Reduction Fusion
// 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes.cpp
//
// Section 14.3 -- closes the chapter by plugging reductionFusionPass()
// into Chapter 8's own unmodified runPasses(), and measuring REAL bytes
// moved, before and after, on a clean chain that ends in a reduction:
// a=Input([3,4]), b=Input([4]), t1=Add(a,b), t2=ReLU(t1), s=Sum(t2) --
// no sharing anywhere in this graph, so nothing stops it from fusing all
// the way down to one kernel.
//
// One real wrinkle Chapter 13's own elementwiseFusionTransform() never
// had to face: reductionFusionPass() needs elementCounts (to bake
// reduceElementCount into each Sum step, Section 14.1), but Chapter 8's
// TransformPass contract is exactly Graph(const Graph&) -- no room for a
// second argument. This section's own wrapper captures a specific
// graph's precomputed element counts in a closure instead of changing
// that contract, a real, concrete instance of a pass gaining a genuine
// new external dependency Chapter 8's interface was never designed to
// carry.
//
// This section also has to fix Chapter 12's own totalFlops() convention
// ("one FLOP per output element"), which quietly assumed every op's
// element-for-element correspondence -- exactly true for Add/Mul/ReLU,
// but wrong for Sum: reducing N elements to 1 takes N-1 additions, not 1.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes.cpp -o 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes
// Run:     ./032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from File 030/031, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input:            return "Input";
        case OpKind::Const:            return "Const";
        case OpKind::Add:              return "Add";
        case OpKind::Mul:              return "Mul";
        case OpKind::ReLU:             return "ReLU";
        case OpKind::Sum:              return "Sum";
        case OpKind::FusedElementwise: return "FusedElementwise";
        default:                       return "FusedReduction";
    }
}

enum class OperandKind { ExternalInput, PriorStep };
struct FusedOperand { OperandKind kind; int index; };
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
        for (const auto& n : g.nodes())
            for (const Value& in : n->inputs)
                if (in.nodeId == id)
                    if (--inDegree[n->id] == 0) ready.push_back(n->id);
    }
    result.ok = (result.order.size() == g.size());
    return result;
}

// ==================== Shape / broadcastShapes / inferShapes (from File 030/031, unchanged) ====================

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
        if (n->op == OpKind::Input || n->op == OpKind::Const) shapes[id] = declaredShapes.at(id);
        else if (n->op == OpKind::Add || n->op == OpKind::Mul)
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        else if (n->op == OpKind::ReLU) shapes[id] = shapes.at(n->inputs[0].nodeId);
        else shapes[id] = Shape{};  // Sum
    }
    return shapes;
}

// ==================== resolveIntoGroup() / reductionFusionPass() (from File 030/031, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized,
                                      std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId,
                                      std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) ||
                           (p->op == OpKind::Sum) ||
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

// ==================== printGraph() / evaluate() (from File 030/031, unchanged) ====================

static std::string formatGraph(const Graph& g) {
    std::string out;
    char buf[256];
    for (const auto& n : g.nodes()) {
        if (n->op != OpKind::FusedElementwise && n->op != OpKind::FusedReduction) {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(", n->id, n->debugName.c_str(), opKindStr(n->op).c_str());
            out += buf;
            if (n->op == OpKind::Const) {
                snprintf(buf, sizeof(buf), "%g", n->constValue);
                out += buf;
            } else {
                for (size_t i = 0; i < n->inputs.size(); ++i) {
                    if (i) out += ", ";
                    snprintf(buf, sizeof(buf), "%%%d", n->inputs[i].nodeId);
                    out += buf;
                }
            }
            out += ")\n";
        } else {
            snprintf(buf, sizeof(buf), "  %%%d %s = %s(external inputs: ", n->id, n->debugName.c_str(),
                     opKindStr(n->op).c_str());
            out += buf;
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                snprintf(buf, sizeof(buf), "ext%zu=%%%d", i, n->inputs[i].nodeId);
                out += buf;
            }
            out += ")\n";
            for (size_t s = 0; s < n->fusedSteps.size(); ++s) {
                const FusedStep& step = n->fusedSteps[s];
                snprintf(buf, sizeof(buf), "      step%zu = %s(", s, opKindStr(step.op).c_str());
                out += buf;
                for (size_t i = 0; i < step.operands.size(); ++i) {
                    if (i) out += ", ";
                    const FusedOperand& o = step.operands[i];
                    snprintf(buf, sizeof(buf), "%s%d", o.kind == OperandKind::ExternalInput ? "ext" : "step", o.index);
                    out += buf;
                }
                out += ")";
                if (step.op == OpKind::Sum) {
                    snprintf(buf, sizeof(buf), "  [reduces %lld elements]", step.reduceElementCount);
                    out += buf;
                }
                out += (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output\n" : "\n";
            }
        }
    }
    return out;
}
static void printGraph(const Graph& g) { printf("%s", formatGraph(g).c_str()); }

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName,
                                              const std::map<int, long long>& elementCounts) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("evaluate: graph is not acyclic");
    std::map<int, float> valuesById;
    std::map<std::string, float> valuesByName;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        float v;
        if (n->op == OpKind::Input) v = inputValuesByName.at(n->debugName);
        else if (n->op == OpKind::Const) v = n->constValue;
        else if (n->op == OpKind::Add) v = valuesById.at(n->inputs[0].nodeId) + valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::Mul) v = valuesById.at(n->inputs[0].nodeId) * valuesById.at(n->inputs[1].nodeId);
        else if (n->op == OpKind::ReLU) v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        else if (n->op == OpKind::Sum) v = valuesById.at(n->inputs[0].nodeId) * static_cast<float>(elementCounts.at(n->inputs[0].nodeId));
        else {
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
                else sv = read(step.operands[0]) * static_cast<float>(step.reduceElementCount);
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== Chapter 8's TransformPass / runPasses() (unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;
struct NamedPass { std::string name; TransformPass pass; };

static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass (%zu nodes) ---\n", initial.size());
    printGraph(initial);

    Graph current = passes[0].pass(initial);
    {
        TopoResult topo = topologicalSort(current);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + passes[0].name + "' produced a graph that is not acyclic");
        }
    }
    printf("\n--- after pass '%s' (%zu nodes) ---\n", passes[0].name.c_str(), current.size());
    printGraph(current);

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& np = passes[i];
        Graph next = np.pass(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + np.name + "' produced a graph that is not acyclic");
        }
        printf("\n--- after pass '%s' (%zu nodes) ---\n", np.name.c_str(), next.size());
        printGraph(next);
        current = std::move(next);
    }
    return current;
}

// ==================== Section 14.3: bytes moved (from Chapter 13's File 029, unchanged) ====================
//
// Works uniformly for ANY node this chapter's IR can build -- including
// FusedReduction -- with ZERO changes needed: a leaf moves nothing; every
// other node reads its own `inputs` (external reads) and writes its own
// single output. A FusedReduction's own output element count is exactly
// 1 (Section 14.1's inferShapes() extension: Sum -> Shape{}), which this
// function picks up automatically through the SAME elementCounts/
// representativeOldId side channel Chapter 13 already built -- Section
// 14.1's one-line shape extension is the ENTIRE reason Chapter 13's own
// byte-counting apparatus needs no changes at all to handle reduction
// fusion too.
static constexpr double kBytesPerElement = 4.0;

static long long bytesMoved(const Graph& g, const std::map<int, long long>& elementCounts) {
    long long totalElements = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        for (const Value& in : n->inputs) totalElements += elementCounts.at(in.nodeId);
        totalElements += elementCounts.at(n->id);
    }
    return static_cast<long long>(static_cast<double>(totalElements) * kBytesPerElement);
}

// ==================== Section 14.3: FLOPs, corrected for a reduction ====================
//
// Chapter 12's own totalFlops() used "one FLOP per output element" -- an
// exact fit for Add/Mul/ReLU, where output element count IS operation
// count. Sum breaks that: reducing N elements to 1 takes N-1 additions,
// not 1. Using the old convention unmodified would claim summing a
// million elements costs a single FLOP -- silently, not even an error.
// This function fixes it with one new branch, on the ORIGINAL (unfused)
// graph only -- total FLOPs is identical whether fused or not, the same
// claim Chapters 12 and 13 already made; fusion changes WHERE and HOW,
// never HOW MUCH.
static long long totalFlops(const Graph& g, const std::map<int, Shape>& shapes) {
    long long total = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        if (n->op == OpKind::Sum) {
            total += numElements(shapes.at(n->inputs[0].nodeId)) - 1;  // N elements -> N-1 additions
        } else {
            total += numElements(shapes.at(n->id));  // Add/Mul/ReLU: one FLOP per output element
        }
    }
    return total;
}

int main() {
    printf("=== Section 14.3: reductionFusionPass() through Chapter 8's own runPasses() ===\n\n");

    // a=[3,4], b=[4] (broadcast); t1=Add(a,b); t2=ReLU(t1); s=Sum(t2).
    // NO sharing anywhere -- every intermediate node has exactly one
    // consumer, so nothing stops this chain from fusing all the way down.
    Graph chain;
    Value a = chain.addInput("a");
    Value b = chain.addInput("b");
    Value t1 = chain.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = chain.addUnary(OpKind::ReLU, t1, "t2");
    Value s = chain.addUnary(OpKind::Sum, t2, "s");
    (void)s;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> originalShapes = inferShapes(chain, declared);
    std::map<int, long long> originalElementCounts;
    for (const auto& kv : originalShapes) originalElementCounts[kv.first] = numElements(kv.second);

    // Chapter 8's TransformPass contract is exactly Graph(const Graph&) --
    // no room for reductionFusionPass()'s own second argument. The fix is
    // a closure that captures this SPECIFIC graph's own precomputed
    // element counts, rather than changing that contract.
    TransformPass reductionFusionTransform = [originalElementCounts](const Graph& g) {
        return reductionFusionPass(g, originalElementCounts).graph;
    };

    Graph afterPasses = runPasses(chain, {{"reductionFusionPass", reductionFusionTransform}});

    bool passManagerAccepted = (afterPasses.size() == 3);
    printf("\nself-check: runPasses() accepted reductionFusionTransform with ZERO changes to\n");
    printf("Chapter 8's own PassManager interface -- the closure carries the shape dependency\n");
    printf("reductionFusionPass() needs, so TransformPass itself stays exactly Graph(const Graph&) (%s)\n",
           passManagerAccepted ? "confirmed" : "MISMATCH");

    printf("\n=== Real bytes moved: before vs. after ===\n\n");

    FusionResult fusionResult = reductionFusionPass(chain, originalElementCounts);
    const Graph& fused = fusionResult.graph;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fusionResult.representativeOldId.at(n->id);
        fusedElementCounts[n->id] = originalElementCounts.at(oldId);
    }

    long long bytesBefore = bytesMoved(chain, originalElementCounts);
    long long bytesAfter = bytesMoved(fused, fusedElementCounts);

    printf("Bytes moved, BEFORE fusion (3 separate kernels -- t1, t2, s):   %lld\n", bytesBefore);
    printf("Bytes moved, AFTER reductionFusionPass() (t1/t2/s fuse into\n");
    printf("             ONE kernel -- no sharing anywhere in this chain): %lld\n", bytesAfter);

    bool realSavings = (bytesAfter < bytesBefore);
    double reduction = 1.0 - static_cast<double>(bytesAfter) / static_cast<double>(bytesBefore);
    printf("\nself-check: bytes moved strictly decreased (%lld -> %lld, a %.1f%% reduction) (%s)\n",
           bytesBefore, bytesAfter, reduction * 100.0, realSavings ? "confirmed" : "MISMATCH");

    long long flops = totalFlops(chain, originalShapes);
    double aiBefore = static_cast<double>(flops) / static_cast<double>(bytesBefore);
    double aiAfter = static_cast<double>(flops) / static_cast<double>(bytesAfter);
    static constexpr double kRidgePointFlopsPerByte = 12.5402;  // same A100 figure as Chapter 12's own 12.2
    printf("\nTotal FLOPs (identical either way -- fusion doesn't change the arithmetic, only where\n");
    printf("it happens; Sum's own N-1-additions correction from this section is what makes this\n");
    printf("number honest): %lld\n", flops);
    printf("AI_before = %lld / %lld bytes = %.6f FLOPs/byte\n", flops, bytesBefore, aiBefore);
    printf("AI_after  = %lld / %lld bytes = %.6f FLOPs/byte\n", flops, bytesAfter, aiAfter);
    printf("(ridge point, same A100 numbers as Chapter 12's own Section 12.2: %.4f FLOPs/byte --\n",
           kRidgePointFlopsPerByte);
    printf(" still memory-bound on both sides at this toy graph's tiny scale, consistent with\n");
    printf(" Chapter 12's own scale caveat)\n");
    bool aiImproved = aiAfter > aiBefore;

    printf("\n=== Correctness: evaluate() agreement, before vs. after ===\n\n");
    std::map<std::string, float> in = {{"a", 2.0f}, {"b", 3.0f}};
    float origOut = evaluate(chain, in, originalElementCounts).at("s");
    float fusedOut = evaluate(fused, in, {}).at("s");
    bool sameAnswer = (origOut == fusedOut);
    printf("self-check: evaluate(original, a=2,b=3).s = %g, evaluate(fused, ...).s = %g (%s)\n",
           origOut, fusedOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = passManagerAccepted && realSavings && aiImproved && sameAnswer;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes.cpp -o 032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes
./032_reduction_fusion_through_chapter_8s_pass_manager_measured_in_bytes
```

**Output:**

```text
=== Section 14.3: reductionFusionPass() through Chapter 8's own runPasses() ===

--- before any pass (5 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 t1 = Add(%0, %1)
  %3 t2 = ReLU(%2)
  %4 s = Sum(%3)

--- after pass 'reductionFusionPass' (3 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 s = FusedReduction(external inputs: ext0=%0, ext1=%1)
      step0 = Add(ext0, ext1)
      step1 = ReLU(step0)
      step2 = Sum(step1)  [reduces 12 elements]  <- this node's own output

self-check: runPasses() accepted reductionFusionTransform with ZERO changes to
Chapter 8's own PassManager interface -- the closure carries the shape dependency
reductionFusionPass() needs, so TransformPass itself stays exactly Graph(const Graph&) (confirmed)

=== Real bytes moved: before vs. after ===

Bytes moved, BEFORE fusion (3 separate kernels -- t1, t2, s):   260
Bytes moved, AFTER reductionFusionPass() (t1/t2/s fuse into
             ONE kernel -- no sharing anywhere in this chain): 68

self-check: bytes moved strictly decreased (260 -> 68, a 73.8% reduction) (confirmed)

Total FLOPs (identical either way -- fusion doesn't change the arithmetic, only where
it happens; Sum's own N-1-additions correction from this section is what makes this
number honest): 35
AI_before = 35 / 260 bytes = 0.134615 FLOPs/byte
AI_after  = 35 / 68 bytes = 0.514706 FLOPs/byte
(ridge point, same A100 numbers as Chapter 12's own Section 12.2: 12.5402 FLOPs/byte --
 still memory-bound on both sides at this toy graph's tiny scale, consistent with
 Chapter 12's own scale caveat)

=== Correctness: evaluate() agreement, before vs. after ===

self-check: evaluate(original, a=2,b=3).s = 60, evaluate(fused, ...).s = 60 (confirmed)
```

!!! warning "[COMMON TRAP] Reusing an old FLOP-counting convention without checking it still fits"
    Chapter 12's "one FLOP per output element" convention was never stated as universal -- it was correct for every op that existed when it was written, which happened to make it look universal. The moment a shape-CHANGING op like `Sum` exists, the convention's own hidden assumption (output element count equals operation count) breaks, silently: no compile error, no thrown exception, just a wrong number that looks perfectly plausible. The fix here (N-1 additions, not 1) is small, but the more durable lesson is the habit: any formula this book carries forward gets re-checked against the new operator, not assumed to still apply.

## Chapter Summary

This chapter extended Chapter 13's own fusion machinery to `OpKind::Sum`, the first genuinely SHAPE-CHANGING operator this book's IR has needed. Section 14.1 built the new IR (`OpKind::FusedReduction`, `FusedStep`'s new `reduceElementCount` field) and `reductionFusionPass()`, whose single new rule -- a reduction's own output is always external, regardless of consumer count -- required both a new guard inside `resolveIntoGroup()` and a new top-level branch in the pass's own walk, and proved, by direct textual comparison, that the result is a strict generalization of Chapter 13's own pass rather than a separate set of rules. Section 14.2 showed what that rule forces on two graphs: one where a reduction's single consumer still can't inline it (a boundary caused purely by the shape change), and one where a shared value and a reduction's own output are BOTH boundaries in the same graph at once, correctly producing no fusion at all -- the honest answer for a graph whose structure offers none. Section 14.3 closed the chapter by plugging the pass into Chapter 8's own unmodified `runPasses()` via a closure (working around `TransformPass`'s own signature having no room for the shape dependency this pass needs) and measuring real savings on a clean, unshared chain: 260 bytes before fusion, 68 after -- while also catching and fixing a real, silent gap in Chapter 12's own FLOP-counting convention, which never accounted for a reduction's true arithmetic cost until this chapter checked.

## Self-Check Questions

1. Every op Chapter 13's `elementwiseFusionPass()` fused was shape-preserving. What specifically breaks about the `PriorStep` "register" model once an op can change shape, and why does that make `Sum` unsafe to inline the way `Add`/`Mul`/`ReLU` could?
2. `resolveIntoGroup()` gains exactly one new condition this chapter. Why isn't that condition, by itself, enough to keep reduction fusion safe -- what second change did the pass's own top-level walk also need, and what would go wrong without it?
3. Section 14.1 always wraps a `Sum` node's result in a `FusedReduction`, even when nothing was inlined into it (a "trivial" one-step body) -- a deliberate departure from Section 13.1's own "one step, emit a plain node" rule. Why was that departure necessary here specifically?
4. In Section 14.2's Graph B, `t1` and `s` are both boundaries, but for two different reasons. What is each reason, and would removing ONE of them (but not the other) change the fused result?
5. Section 14.3's `reductionFusionTransform()` is a closure rather than a plain function pointer, unlike Chapter 13's own `elementwiseFusionTransform()`. What does it need to capture, and why couldn't Chapter 8's own `TransformPass` signature be extended instead?
6. Why does Chapter 13's own `bytesMoved()` function require zero changes to correctly count bytes for a `FusedReduction` node, when it was never written with reductions in mind?
7. What specifically was wrong with applying Chapter 12's "one FLOP per output element" convention to a `Sum` node unmodified, and why does this kind of bug tend to stay silent rather than announce itself?
8. Section 14.1's backward-compatibility check runs `reductionFusionPass()` on a graph with no `Sum` node at all and compares its output, as text, against `elementwiseFusionPass()`'s own output on the same graph. What would it mean, concretely, if that check had failed?

## Where We Go Next

This chapter fused a reduction with the elementwise chain feeding it -- but only ever ONE fused kernel deep, and only ever a chain, never a loop. Chapter 15, "Loop Fusion and Tiling," takes up what happens once CUDA Hammer's own generated code has to reason about actual iteration -- fusing operations that each need their own loop over a tensor's elements into a single shared loop nest, and the tiling decisions that come with controlling how much of a tensor's own data a single fused kernel invocation touches at once. Apply the Chapter 5-14 depth-level standard (more prose, more diagrams before code) there too.

## Worked Solutions

1. The `PriorStep` model relies on every step in a fused body sharing the SAME element-for-element correspondence with the node's own external inputs -- "the same element, computed one step earlier." A shape-changing op breaks that correspondence outright: `Sum` takes N elements and produces exactly 1, so there is no longer "the same element" for anything AFTER it to read as a `PriorStep`. Inlining `Sum` mid-body (rather than as the body's own final step) would produce a step whose own output no longer corresponds to any single element of the body's inputs, making every later step's own per-element reasoning meaningless.
2. The new condition (`p->op == OpKind::Sum` forces external) only prevents a reduction's output from being inlined INTO something else's body -- it says nothing about how the reduction's OWN root gets built. Without a matching change to the pass's own top-level walk, a single-consumer `Sum` node would still hit the "consumers == 1, skip, my one consumer will build me" shortcut inherited from `elementwiseFusionPass()` -- and since the new guard now refuses to let that one consumer inline it, nobody would ever materialize it. It would silently vanish from the output graph, not error out.
3. A one-step `FusedReduction` still needs to carry `reduceElementCount` on its own `Sum` step, so `evaluate()` can produce a correct number without re-deriving shape information. A plain `OpKind::Sum` node has nowhere to store that count -- only a `FusedStep` has the field. Emitting a plain node here (as Section 13.1's rule would suggest) would silently lose the information `evaluate()` needs.
4. `t1` is a boundary because it is SHARED -- it has two consumers (`s` and `y2`), and Chapter 13's own sharing rule (unchanged) refuses to inline any node with more than one consumer. `s` is a boundary because it is a REDUCTION -- even though `s` itself has only ONE consumer (`y`), this chapter's new rule refuses to inline a `Sum` node's output into anyone else's body regardless of consumer count. Removing `t1`'s second consumer (so only `s` reads it) would let `t1` inline into `s`'s own `FusedReduction` body, same as Graph A -- but `s` would STILL stand alone as its own root, because the reduction rule doesn't depend on `t1`'s sharing at all. The two rules are independent.
5. It needs to capture the SPECIFIC graph's own precomputed `elementCounts` map (from `inferShapes()` on that graph), because `reductionFusionPass()` needs it to bake `reduceElementCount` into each `Sum` step, and `TransformPass` itself is exactly `Graph(const Graph&)` -- no second parameter. Extending that signature would break every other pass already using it (five of them, by the end of Chapter 13) for a dependency only reduction fusion actually has; a closure lets exactly one pass carry exactly the extra state it needs, without changing a contract four other passes already rely on staying fixed.
6. `bytesMoved()` was written generically: for any non-leaf node, read every one of its own `inputs`' element counts and add its own single output's element count -- it never inspects what KIND of node it's counting. A `FusedReduction` node's own output element count is exactly 1, which falls out automatically from Section 14.1's one-line `inferShapes()` extension (`Sum -> Shape{}`) flowing through the same `representativeOldId` side channel Chapter 13 already built for exactly this purpose. The function needed no changes because its own generality already covered a case it was never written with in mind.
7. The convention silently assumed output element count equals operation count -- exactly true for `Add`/`Mul`/`ReLU`, where each output element genuinely costs one operation, but false for `Sum`, where N elements collapse to 1 output element via N-1 additions. Using it unmodified for `Sum` would compute FLOPs as 1, regardless of how many elements were actually summed -- a wrong number with no error, no crash, and no obviously implausible output to catch it by eye, which is exactly why this kind of convention needs to be re-checked against each new operator rather than assumed to still apply.
8. A failure would mean `reductionFusionPass()`'s own Add/Mul/ReLU handling had DIVERGED from `elementwiseFusionPass()`'s, even on a graph that never touches the new `Sum`-specific code paths at all -- concretely, it would mean this chapter's claim of being a strict generalization was false, and some graph existed where adding reduction-fusion support to a pass changed how it handled ordinary elementwise fusion too, which should never happen given the two passes share the exact same Add/Mul/ReLU branch verbatim.

---

**Sources cited in this chapter:**

None new. `OpKind::Sum` and `OpKind::FusedReduction` are original to this book, extending Chapter 13's own `OpKind::FusedElementwise` design in the same spirit already cited to XLA's own real fusion computations in Chapter 3.
