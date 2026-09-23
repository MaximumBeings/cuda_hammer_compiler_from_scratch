# 16. Fusion Boundaries: What Can't Be Fused, and Why

**What you will understand:** `classifyNode()`, a diagnostic taxonomy that re-derives every boundary condition Chapters 13-15 already enforce and exposes it as an explicit, inspectable reason rather than a side effect of deciding whether to inline; a genuinely different KIND of boundary -- a cost-based size cap, `computeSizeCapBoundaries()`, that exists even where fusing further would still be correct; and `boundedReductionFusionPass()`, which closes Part 3 by combining every boundary this Part has built into one pass, measured honestly, cost and all, through Chapter 8's still-unmodified `PassManager`.

**What you need to know first:** Chapter 13's `resolveIntoGroup()`/`elementwiseFusionPass()`, Chapter 14's `OpKind::Sum`/`FusedReduction`/`reductionFusionPass()`, Chapter 15's loop-nest view (referenced, not extended, in this chapter), and Chapter 8's `TransformPass`/`runPasses()`.

---

Three chapters, three boundaries, three separate reasons a fusion opportunity has had to stop: Chapter 13's sharing rule (a value with more than one consumer stays external, or duplication or disagreement follows), Chapter 14's reduction rule (a `Sum` node's own output is always external, because a shape change breaks the per-element correspondence the whole `FusedStep` model depends on), and Chapter 15's loop-nest view (a separate, later question about whether two nodes' own loops could still share a physical loop even when their programs stay apart). Every one of those rules was added the same way: one more condition inside `resolveIntoGroup()`'s own `mustBeExternal` check, explained in the prose around it, but never gathered into one place a reader -- or a future chapter's own code -- could ask "why, exactly, is this node a boundary" and get a direct answer. This chapter closes Part 3 by doing three things: building that missing diagnostic view, adding one genuinely new KIND of boundary this book has never needed before (a cost-based limit with nothing to do with correctness), and putting everything together on one graph, measured honestly, cost included, through the same `PassManager` every fusion pass in this book has used unmodified since Chapter 13.

```text
PART 3'S OWN BOUNDARIES, GATHERED IN ONE PLACE:

  Chapter 13 -- SHARED:      a value with 2+ consumers stays external
                              (inlining it would duplicate or disagree)

  Chapter 14 -- REDUCTION:   a Sum node's own output ALWAYS stays external
                              (a shape change breaks the per-element model)

  Chapter 15 -- loop-nest incompatibility: a SEPARATE, later question --
                              not a resolveIntoGroup() condition at all

  THIS CHAPTER adds:

  Section 16.1 -- explaining, not just enforcing: classifyNode() names
                  WHICH of the above applies to a given node, or several
                  at once, re-deriving existing logic rather than adding any

  Section 16.2 -- SizeCap: a cost-based boundary with NOTHING to do with
                  correctness -- fusing further would still be correct,
                  just costlier to generate and run as one kernel

  Section 16.3 -- boundedReductionFusionPass(): every boundary this Part
                  has built, on one graph, measured honestly through
                  Chapter 8's own unmodified PassManager
```

## 16.1 A Taxonomy of Fusion Boundaries: Explaining, Not Just Enforcing

### Intuition

Ask `elementwiseFusionPass()` or `reductionFusionPass()` whether a given node fuses, and you get a yes or no -- buried inside a boolean, `mustBeExternal`, computed once and then acted on immediately. That's exactly the right design for a PASS, whose only job is to build the right output graph. It's the wrong design for a QUESTION a person -- or a later chapter's own diagnostic tooling -- might actually want answered directly: "why didn't this fuse?" Chapter 14's own Graph B already ran into this informally, in prose: `t1` and `s` were both boundaries, the chapter explained, but for two completely different reasons, and working that out required reading the code and reasoning it through by hand. `classifyNode()` is that same reasoning, done once, as a reusable function, so the answer to "why" is a direct call away rather than a re-derivation every time.

### Background

`classifyNode()` checks exactly the conditions `resolveIntoGroup()` and each pass's own top-level walk already check -- Leaf (`Input`/`Const`, always external, a memory or literal boundary trivially, and the ONLY reason a leaf ever carries), Reduction (a `Sum` node's own output, unconditionally, whatever its consumer count), Shared (two or more consumers), and one category worth naming explicitly for the first time here: GraphRoot, for a node with ZERO consumers. Chapter 9's "last node is the graph's own designated output" convention usually makes that node unique, but nothing about fusion itself enforces that uniqueness -- Chapter 14's own Graph B, reused verbatim as this section's own second test graph, was never run through `deadCodeEliminationPass()` and genuinely has TWO independent zero-consumer sinks (`y` and `y2`), and `reductionFusionPass()` materializes both as their own roots without caring how many there are. `classifyNode()` returns every reason that applies, not just the first match an `if`/`else` chain happens to hit, because a node can carry more than one at once -- and the function's own correctness is checked the way this book always checks a claim like this: not by trusting the logic looks right, but by running the REAL pass on real graphs and confirming, node by node, that `classifyNode()`'s own predicted verdict (boundary or not) agrees with what the pass actually built.

```text
classifyNode() RETURNS EVERY REASON THAT APPLIES, NOT JUST ONE:

  Chapter 14's own Graph B (x=[4], w=scalar):

    x        w
    |        |
    ReLU(t1)-+---- 2 consumers: s AND y2 --------- reasons = [Shared]
    |        |
    Sum(s)   |     ---- 1 consumer: y, but Sum ---- reasons = [Reduction]
    |        |
    Add(y)  Mul(y2)
    ^        ^
    0 consumers each (this graph was never       reasons = [GraphRoot]
    run through deadCodeEliminationPass(),         (for BOTH y and y2)
    so it genuinely has two independent sinks)

  three nodes, three DIFFERENT reasons -- classifyNode() names each one
  directly instead of requiring a fresh re-derivation every time
```

```cpp
// Chapter 16: Fusion Boundaries
// 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing.cpp
//
// Section 16.1 -- Chapters 13, 14, and 15 each added exactly one new
// condition to resolveIntoGroup()'s own mustBeExternal check: Leaf/Shared
// (Chapter 13), Reduction (Chapter 14, independent of consumer count), and
// -- as a SEPARATE, later question, not a condition inside resolveIntoGroup()
// at all -- loop-nest incompatibility (Chapter 15). Nothing so far has ever
// asked, for a GIVEN node, "why, specifically, is this a boundary" as its
// own first-class question -- every earlier chapter only ever answered
// "boundary or not" as a side effect of deciding whether to inline. This
// section builds that missing piece: classifyNode(), a small diagnostic
// function that re-derives the SAME conditions resolveIntoGroup() already
// checks, exposed as an explicit, inspectable taxonomy -- and proves, by
// direct comparison against the real fusion passes' own actual behavior,
// that it agrees with them on every node of every test graph this book has
// built so far.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing.cpp -o 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing
// Run:     ./036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
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

// ==================== resolveIntoGroup / elementwiseFusionPass / reductionFusionPass (from Chapter 14, unchanged) ====================

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

// ==================== Section 16.1: BoundaryReason / classifyNode() ====================
//
// Every condition below is a re-derivation of something resolveIntoGroup()
// or reductionFusionPass()'s own top-level walk ALREADY checks -- nothing
// here is new fusion LOGIC, only a new, explicit, inspectable NAME for
// logic that has existed since Chapter 13. A node can carry more than one
// reason at once (Chapter 14's own Graph B already showed this informally:
// t1 and s were both boundaries, for two entirely different reasons) --
// classifyNode() returns every reason that applies, not just the first one
// found, so that overlap is visible rather than hidden behind whichever
// condition an if/else chain happened to check first.
enum class BoundaryReason { NotABoundary, Leaf, GraphRoot, Shared, Reduction };

static std::string boundaryReasonStr(BoundaryReason r) {
    switch (r) {
        case BoundaryReason::NotABoundary: return "NotABoundary";
        case BoundaryReason::Leaf:          return "Leaf";
        case BoundaryReason::GraphRoot:     return "GraphRoot";
        case BoundaryReason::Shared:        return "Shared";
        default:                            return "Reduction";
    }
}

// consumers.at(n->id) == 0 means nothing IN THIS GRAPH reads this node's
// own value. Chapter 9's "last node is the output" convention picks exactly
// ONE such node as the graph's own designated output when deadCodeEliminationPass()
// runs -- but no fusion pass in this book has ever required that there be
// only one. reductionFusionPass() (like elementwiseFusionPass() before it)
// materializes EVERY zero-consumer node as its own root, whatever their
// number -- a graph nobody has run DCE on can genuinely have more than one
// (Section 16.1's own Graph 2, below, actually does: `y` and `y2` are BOTH
// zero-consumer sinks, reused verbatim from Chapter 14's own Graph B).
// GraphRoot names that real, checked behavior precisely -- "a root fusion
// treats as its own" -- rather than overclaiming a uniqueness fusion itself
// was never told to enforce.
static std::vector<BoundaryReason> classifyNode(const Node* n, const std::map<int, int>& consumers) {
    std::vector<BoundaryReason> reasons;
    if (n->op == OpKind::Input || n->op == OpKind::Const) {
        reasons.push_back(BoundaryReason::Leaf);
        return reasons;  // a leaf's ONLY reason is being a leaf -- nothing else to check
    }
    if (n->op == OpKind::Sum) {
        reasons.push_back(BoundaryReason::Reduction);
    }
    int c = consumers.at(n->id);
    if (c == 0) {
        reasons.push_back(BoundaryReason::GraphRoot);
    } else if (c >= 2) {
        reasons.push_back(BoundaryReason::Shared);
    }
    if (reasons.empty()) {
        reasons.push_back(BoundaryReason::NotABoundary);
    }
    return reasons;
}

static bool isBoundary(const std::vector<BoundaryReason>& reasons) {
    return !(reasons.size() == 1 && reasons[0] == BoundaryReason::NotABoundary);
}

static std::string reasonsStr(const std::vector<BoundaryReason>& reasons) {
    std::string out;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i) out += "+";
        out += boundaryReasonStr(reasons[i]);
    }
    return out;
}

// Reuses reductionFusionPass()'s OWN consumer-counting logic (copy of the
// two-line loop every pass in this chapter builds) so classifyNode()'s
// predictions can be checked against the SAME consumers map the real pass
// actually used, not a separately-computed one that might disagree by
// accident.
static std::map<int, int> countConsumers(const Graph& g) {
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    return consumers;
}

// The actual proof: for every ORIGINAL node in a graph, classifyNode()'s
// own verdict (boundary or not) must agree with whether that node's own id
// shows up as a top-level materialized root in the FUSED graph's own
// representativeOldId map -- the REAL, executed outcome of running the
// actual pass, not a second opinion computed independently.
static bool verifyClassificationAgreesWithPass(const Graph& original, const FusionResult& result) {
    std::map<int, int> consumers = countConsumers(original);
    std::set<int> materializedOldIds;
    for (const auto& kv : result.representativeOldId) materializedOldIds.insert(kv.second);

    for (const auto& n : original.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumers);
        bool predictedBoundary = isBoundary(reasons);
        bool actuallyMaterializedAsOwnRoot = materializedOldIds.count(n->id) > 0;
        // A Leaf (Input/Const) is always "materialized" too (as itself, not
        // inlined into anything) -- both predictedBoundary and
        // actuallyMaterializedAsOwnRoot are true for a leaf, so this check
        // holds uniformly across every category without a special case.
        if (predictedBoundary != actuallyMaterializedAsOwnRoot) {
            printf("  MISMATCH at node %%%d (%s): classifyNode predicted boundary=%s, reasons=[%s],\n",
                   n->id, n->debugName.c_str(), predictedBoundary ? "true" : "false", reasonsStr(reasons).c_str());
            printf("    but the pass's own actual output says materialized-as-own-root=%s\n",
                   actuallyMaterializedAsOwnRoot ? "true" : "false");
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Section 16.1: classifyNode() -- re-deriving fusion's own logic as an explicit taxonomy ===\n\n");

    // ---- Graph 1: Chapter 13's own diamond ----
    printf("--- Graph 1: Chapter 13's own diamond (a=[3,4], b=[4]) ---\n\n");
    Graph diamond;
    Value a  = diamond.addInput("a");
    Value b  = diamond.addInput("b");
    Value t1 = diamond.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = diamond.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = diamond.addUnary(OpKind::ReLU, t1, "t3");
    Value dout = diamond.addBinary(OpKind::Add, t2, t3, "out");
    (void)dout;
    std::map<int, Shape> declaredD = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> shapesD = inferShapes(diamond, declaredD);
    std::map<int, long long> elementCountsD;
    for (const auto& kv : shapesD) elementCountsD[kv.first] = numElements(kv.second);
    FusionResult diamondResult = reductionFusionPass(diamond, elementCountsD);

    std::map<int, int> consumersD = countConsumers(diamond);
    for (const auto& n : diamond.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumersD);
        printf("  %%%d %-4s consumers=%d  reasons=[%s]\n", n->id, n->debugName.c_str(),
               consumersD.at(n->id), reasonsStr(reasons).c_str());
    }
    bool diamondOk = verifyClassificationAgreesWithPass(diamond, diamondResult);
    printf("\nself-check: classifyNode()'s own predicted boundary/not-boundary verdict agrees with\n");
    printf("reductionFusionPass()'s own actual output on every node of the diamond graph (%s)\n",
           diamondOk ? "confirmed" : "MISMATCH");

    // ---- Graph 2: Chapter 14's own Graph B (two DIFFERENT reasons, same graph) ----
    printf("\n--- Graph 2: Chapter 14's own Graph B (x=[4], w=scalar) ---\n\n");
    Graph gB;
    Value x  = gB.addInput("x");
    Value w  = gB.addInput("w");
    Value bt1 = gB.addUnary(OpKind::ReLU, x, "t1");
    Value bs  = gB.addUnary(OpKind::Sum, bt1, "s");
    Value by  = gB.addBinary(OpKind::Add, bs, w, "y");
    Value by2 = gB.addBinary(OpKind::Mul, bt1, w, "y2");
    (void)by; (void)by2;
    std::map<int, Shape> declaredB = {{x.nodeId, Shape{{4}}}, {w.nodeId, Shape{}}};
    std::map<int, Shape> shapesB = inferShapes(gB, declaredB);
    std::map<int, long long> elementCountsB;
    for (const auto& kv : shapesB) elementCountsB[kv.first] = numElements(kv.second);
    FusionResult gBResult = reductionFusionPass(gB, elementCountsB);

    std::map<int, int> consumersB = countConsumers(gB);
    for (const auto& n : gB.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumersB);
        printf("  %%%d %-4s consumers=%d  reasons=[%s]\n", n->id, n->debugName.c_str(),
               consumersB.at(n->id), reasonsStr(reasons).c_str());
    }
    bool gBOk = verifyClassificationAgreesWithPass(gB, gBResult);
    printf("\nself-check: t1 is classified Shared (2 consumers), s is classified Reduction (1\n");
    printf("consumer, but Sum) -- two DIFFERENT reasons on two DIFFERENT nodes. y AND y2 are BOTH\n");
    printf("classified GraphRoot: this graph, reused verbatim from Chapter 14's own File 031, was\n");
    printf("never run through deadCodeEliminationPass(), so it genuinely has TWO independent\n");
    printf("zero-consumer sinks, not one -- and reductionFusionPass() materializes both as their\n");
    printf("own roots without caring how many there are, exactly as GraphRoot's own definition\n");
    printf("above says it should. All four predictions agree with the pass's own actual output\n");
    printf("(which fuses nothing in this graph at all) (%s)\n", gBOk ? "confirmed" : "MISMATCH");

    bool t1IsShared = false, sIsReduction = false, yIsGraphRoot = false, y2IsGraphRoot = false;
    for (const auto& n : gB.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumersB);
        if (n->debugName == "t1")
            t1IsShared = std::find(reasons.begin(), reasons.end(), BoundaryReason::Shared) != reasons.end();
        if (n->debugName == "s")
            sIsReduction = std::find(reasons.begin(), reasons.end(), BoundaryReason::Reduction) != reasons.end();
        if (n->debugName == "y")
            yIsGraphRoot = std::find(reasons.begin(), reasons.end(), BoundaryReason::GraphRoot) != reasons.end();
        if (n->debugName == "y2")
            y2IsGraphRoot = std::find(reasons.begin(), reasons.end(), BoundaryReason::GraphRoot) != reasons.end();
    }
    printf("self-check: t1 carries Shared (%s), s carries Reduction (%s), y carries GraphRoot (%s),\n",
           t1IsShared ? "confirmed" : "MISMATCH", sIsReduction ? "confirmed" : "MISMATCH",
           yIsGraphRoot ? "confirmed" : "MISMATCH");
    printf("y2 carries GraphRoot too (%s)\n", y2IsGraphRoot ? "confirmed" : "MISMATCH");

    bool allOk = diamondOk && gBOk && t1IsShared && sIsReduction && yIsGraphRoot && y2IsGraphRoot;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing.cpp -o 036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing
./036_a_taxonomy_of_fusion_boundaries_explaining_not_just_enforcing
```

**Output:**

```text
=== Section 16.1: classifyNode() -- re-deriving fusion's own logic as an explicit taxonomy ===

--- Graph 1: Chapter 13's own diamond (a=[3,4], b=[4]) ---

  %0 a    consumers=2  reasons=[Leaf]
  %1 b    consumers=1  reasons=[Leaf]
  %2 t1   consumers=2  reasons=[Shared]
  %3 t2   consumers=1  reasons=[NotABoundary]
  %4 t3   consumers=1  reasons=[NotABoundary]
  %5 out  consumers=0  reasons=[GraphRoot]

self-check: classifyNode()'s own predicted boundary/not-boundary verdict agrees with
reductionFusionPass()'s own actual output on every node of the diamond graph (confirmed)

--- Graph 2: Chapter 14's own Graph B (x=[4], w=scalar) ---

  %0 x    consumers=1  reasons=[Leaf]
  %1 w    consumers=2  reasons=[Leaf]
  %2 t1   consumers=2  reasons=[Shared]
  %3 s    consumers=1  reasons=[Reduction]
  %4 y    consumers=0  reasons=[GraphRoot]
  %5 y2   consumers=0  reasons=[GraphRoot]

self-check: t1 is classified Shared (2 consumers), s is classified Reduction (1
consumer, but Sum) -- two DIFFERENT reasons on two DIFFERENT nodes. y AND y2 are BOTH
classified GraphRoot: this graph, reused verbatim from Chapter 14's own File 031, was
never run through deadCodeEliminationPass(), so it genuinely has TWO independent
zero-consumer sinks, not one -- and reductionFusionPass() materializes both as their
own roots without caring how many there are, exactly as GraphRoot's own definition
above says it should. All four predictions agree with the pass's own actual output
(which fuses nothing in this graph at all) (confirmed)
self-check: t1 carries Shared (confirmed), s carries Reduction (confirmed), y carries GraphRoot (confirmed),
y2 carries GraphRoot too (confirmed)
```

!!! note "classifyNode() adds no new fusion logic -- only a new name for logic that already existed"
    Every condition `classifyNode()` checks was already being checked somewhere in Chapters 13-14's own passes; this section restates none of it differently and adds nothing a pass didn't already decide. The value is entirely in making that decision INSPECTABLE on its own terms -- answerable in one function call rather than requiring a fresh trace through `resolveIntoGroup()`'s own recursion every time the question comes up. `verifyClassificationAgreesWithPass()` exists specifically to keep that promise honest: a diagnostic tool that could silently drift out of sync with the real passes it describes would be worse than no diagnostic at all.

## 16.2 A Cost-Based Boundary: Capping How Large One Fused Kernel Can Grow

### Intuition

Every boundary before this one exists to keep an answer CORRECT. This one exists even when fusing further would still be perfectly correct -- purely because an unbounded fused kernel has real costs no correctness check would ever catch: more live values held at once inside one generated loop body (register pressure), more code generated per kernel (instruction-cache pressure, longer compile times). Real compilers bound this in practice; CUDA Hammer's own version is a simple `maxChainLength` -- a cap on how many steps one `FusedElementwise` body may contain before the chain feeding it has to be cut somewhere.

### Background

The natural first instinct -- check "has this group already gotten too big" live, inside `resolveIntoGroup()`'s own recursion -- runs into a real structural problem worth stating plainly rather than working around silently: that recursion resolves a chain's DEEPEST node FIRST, because every node's own operands are resolved before its own step gets pushed. A check made at ENTRY to each recursive call would see an empty `steps` list for every node in a chain, no matter how long the chain actually is, and only discover the group had grown too large after everything was already inlined -- too late to do anything about it. Rather than threading extra, order-dependent state through that recursion (and rather than trying to patch around it with something fragile), this section computes the cap as a separate, simple PRE-PASS: `computeChainDepths()` walks the graph in topological order and assigns each single-consumer elementwise node a depth -- one more than whatever single-consumer elementwise predecessor feeds it, or `1` if it starts a fresh run -- and `computeSizeCapBoundaries()` marks every node whose own depth is an exact multiple of `maxChainLength` as a forced boundary, BEFORE fusion ever starts. `resolveIntoGroup()` then checks that precomputed set the exact same way it already checks Leaf, Sum, and Shared: as one more static, already-decided property of the node itself, not a dynamic one computed mid-recursion. The one other change this section needs is the same shape Chapter 14 needed for `Sum`: the top-level pass loop's own "consumers == 1, skip, my consumer will build me" shortcut has to ALSO check the size-cap set, or a node the cap marks as a boundary but whose consumer count is still 1 would be skipped here and then refused by `resolveIntoGroup()` when its consumer tries to inline it anyway -- silently vanishing from the output graph exactly the way an unguarded single-consumer `Sum` would have in Chapter 14.

```text
WHY A LIVE CHECK INSIDE resolveIntoGroup()'S OWN RECURSION DOESN'T WORK:

  chain: t1 -> t2 -> t3 -> t4  (each single-consumer, feeding the next)

  resolveIntoGroup() resolves the DEEPEST node FIRST:
    resolve(t4) calls resolve(t3) calls resolve(t2) calls resolve(t1)
    -- steps.size() is 0 at EVERY one of those calls, no matter how
       long the chain is, because nothing gets PUSHED until the
       recursion returns back UP

  a check made at entry to each call would never see steps.size()
  grow during the descent -- it would only discover the group was
  too big after t1, t2, and t3 had ALL already been inlined

  fix: decide the cap BEFORE fusion runs, as a separate pre-pass --
  computeSizeCapBoundaries() -- checked as one more static property,
  the same way Leaf/Sum/Shared already are
```

```cpp
// Chapter 16: Fusion Boundaries
// 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp
//
// Section 16.2 -- every boundary Chapters 13-15 ever built exists to
// preserve CORRECTNESS: sharing (Chapter 13), a shape change (Chapter 14),
// and loop-nest incompatibility (Chapter 15) are all reasons a fusion would
// either compute the wrong answer or couldn't be expressed as one loop nest
// at all. This section builds a genuinely different KIND of boundary --
// one that exists even though fusing further would still be perfectly
// CORRECT, purely because letting a single fused kernel's own body grow
// without limit has real practical costs real compilers have to bound:
// more live values held at once (register pressure), more code generated
// per kernel (instruction cache pressure, compile time). CUDA Hammer's own
// version of that idea is a simple one: maxChainLength, a cap on how many
// steps a single FusedElementwise body may contain before something has to
// give.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp -o 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow
// Run:     ./037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

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

// ==================== Debug printer (from Chapter 13, unchanged) ====================

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
                out += (s + 1 == n->fusedSteps.size()) ? "  <- this node's own output\n" : "\n";
            }
        }
    }
    return out;
}
static void printGraph(const Graph& g) { printf("%s", formatGraph(g).c_str()); }

// ==================== evaluate() (from Chapter 13, unchanged) ====================

static std::map<std::string, float> evaluate(const Graph& g, const std::map<std::string, float>& inputValuesByName) {
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
        } else {  // FusedElementwise
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
                else sv = read(step.operands[0]) * read(step.operands[1]);  // Mul
                stepVals.push_back(sv);
            }
            v = stepVals.back();
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== Chapter 13's own resolveIntoGroup() / elementwiseFusionPass() (unchanged, kept for comparison) ====================

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

// ==================== Section 16.2: a size-cap boundary, precomputed BEFORE fusion runs ====================
//
// A live check ("has the group I'm building already gotten too big") would
// have to be threaded through resolveIntoGroup()'s own recursion -- but
// that recursion resolves a chain's DEEPEST node FIRST (operands are always
// resolved before a node's own step is pushed), so a check made at entry to
// each recursive call would see an empty `steps` for every node in a chain,
// no matter how long the chain is, and only discover the group was too big
// after everything had already been inlined. Rather than threading extra,
// order-dependent state through that recursion, this section computes the
// cap as a SEPARATE, simple pre-pass -- a set of node ids forced external
// for a size reason, decided before fusion ever starts, checked by
// resolveIntoGroup() the exact same way it already checks Leaf/Shared/Reduction:
// as one more static, precomputed property of the node itself.
//
// chainDepth[n] counts how many single-consumer elementwise nodes deep n is
// within its own maximal run: 1 for a node whose relevant operand isn't
// itself such a node (the start of a new chain), or chainDepth[operand] + 1
// otherwise. A node whose own chainDepth is an exact multiple of
// maxChainLength is marked as a boundary -- the last member of one
// maxChainLength-sized segment, forcing the NEXT segment to start fresh.
static std::map<int, long long> computeChainDepths(const Graph& g, const std::map<int, int>& consumers,
                                                     const TopoResult& topo) {
    std::map<int, long long> depth;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        bool isChainCandidate = (n->op == OpKind::Add || n->op == OpKind::Mul || n->op == OpKind::ReLU) &&
                                 consumers.at(id) == 1;
        if (!isChainCandidate) {
            depth[id] = 0;  // Leaf, Shared, or a consumers!=1 node -- not part of any chain itself
            continue;
        }
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

// resolveIntoGroup(), reused verbatim except for ONE new condition -- the
// same shape every earlier chapter's own extension took: Chapter 14 added
// one condition for Reduction: here, sizeCapBoundaries.count(oldId) adds
// one condition for a size cap, checked as a simple, precomputed set
// lookup, exactly like every other condition in mustBeExternal.
static FusedOperand resolveIntoGroupCapped(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                            const std::set<int>& sizeCapBoundaries,
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
                           (consumers.at(oldId) != 1) ||
                           (sizeCapBoundaries.count(oldId) > 0);
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
        operands.push_back(resolveIntoGroupCapped(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
                                                    externalInputs, externalIndexByOldId, steps, stepIndexByOldId));
    }
    int myStepIndex = static_cast<int>(steps.size());
    steps.push_back(FusedStep{p->op, operands});
    stepIndexByOldId[oldId] = myStepIndex;
    return FusedOperand{OperandKind::PriorStep, myStepIndex};
}

// boundedFusionPass()'s own top-level walk needs the SAME companion change
// Chapter 14 needed for Sum: the "consumers==1, skip, my consumer will
// build me" shortcut is only safe when nothing ELSE also forces this node
// external. Without also checking sizeCapBoundaries here, a node the cap
// marks as a boundary but whose consumer count is still 1 would be skipped
// by this loop, then refused by resolveIntoGroupCapped() when its consumer
// tries to inline it anyway -- and, exactly like Chapter 14's own Sum
// oversight would have, it would vanish from the output graph instead of
// materializing as its own root.
static FusionResult boundedFusionPass(const Graph& g, long long maxChainLength) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("boundedFusionPass: graph is not acyclic");
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
        if (consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs) {
            operands.push_back(resolveIntoGroupCapped(in.nodeId, g, consumers, sizeCapBoundaries, materialized,
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
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

static long long maxFusedStepCount(const Graph& g) {
    long long best = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::FusedElementwise) best = std::max(best, static_cast<long long>(n->fusedSteps.size()));
    }
    return best;
}

int main() {
    printf("=== Section 16.2: boundedFusionPass() -- a boundary with nothing to do with correctness ===\n\n");

    // x = Input; c = Const(3); t1 = ReLU(x); t2 = Mul(t1,c); t3 = Add(t2,c);
    // t4 = Mul(t3,c). Every one of t1..t4 has exactly ONE consumer -- under
    // Chapter 13's own UNCAPPED elementwiseFusionPass(), this entire chain
    // fuses into ONE FusedElementwise node, 4 steps. Hand-derivation for
    // maxChainLength=2: chainDepth(t1)=1, chainDepth(t2)=2, chainDepth(t3)=3,
    // chainDepth(t4)=4. Depths divisible by 2: t2 and t4. t4 was ALREADY
    // going to be its own root regardless (zero consumers -- Section 16.1's
    // own GraphRoot category) -- t2 is the genuinely NEW boundary the cap
    // introduces, splitting the chain into [t1,t2] and [t3,t4].
    Graph chain;
    Value x  = chain.addInput("x");
    Value c  = chain.addConst(3.0f, "c");
    Value t1 = chain.addUnary(OpKind::ReLU, x, "t1");
    Value t2 = chain.addBinary(OpKind::Mul, t1, c, "t2");
    Value t3 = chain.addBinary(OpKind::Add, t2, c, "t3");
    Value t4 = chain.addBinary(OpKind::Mul, t3, c, "t4");
    (void)t4;

    printf("Original graph (%zu nodes):\n", chain.size());
    printGraph(chain);

    FusionResult uncapped = elementwiseFusionPass(chain);
    printf("\n--- UNCAPPED (Chapter 13's own elementwiseFusionPass(), no size limit) ---\n\n");
    printGraph(uncapped.graph);

    bool uncappedOk = (uncapped.graph.size() == 3 && maxFusedStepCount(uncapped.graph) == 4);
    printf("\nself-check: uncapped fusion produces ONE FusedElementwise node with all 4 steps,\n");
    printf("6 -> 3 nodes (%s)\n", uncappedOk ? "confirmed" : "MISMATCH");

    FusionResult capped = boundedFusionPass(chain, 2);
    printf("\n--- CAPPED (maxChainLength=2) ---\n\n");
    printGraph(capped.graph);

    bool cappedOk = (capped.graph.size() == 4 && maxFusedStepCount(capped.graph) == 2);
    printf("\nself-check: capped fusion produces TWO FusedElementwise nodes, each with AT MOST\n");
    printf("2 steps, 6 -> 4 nodes -- t2 becomes a NEW boundary the cap alone is responsible for\n");
    printf("(%s)\n", cappedOk ? "confirmed" : "MISMATCH");

    std::map<std::string, float> in = {{"x", 2.0f}};
    float origOut = evaluate(chain, in).at("t4");
    float uncappedOut = evaluate(uncapped.graph, in).at("t4");
    float cappedOut = evaluate(capped.graph, in).at("t4");
    bool sameAnswer = (origOut == uncappedOut) && (origOut == cappedOut);
    printf("\nself-check: evaluate(original, x=2).t4 = %g, evaluate(uncapped,...) = %g,\n",
           origOut, uncappedOut);
    printf("evaluate(capped,...) = %g -- all three agree (%s)\n", cappedOut, sameAnswer ? "confirmed" : "MISMATCH");

    // ---- Real bytes moved: the honest cost of imposing the cap ----
    printf("\n=== The cap's own real, measured cost ===\n\n");
    static constexpr double kBytesPerElement = 4.0;
    // x=[4], c=scalar (1 element); t1..t4 are all shape [4] (ReLU preserves,
    // Mul/Add broadcast [4] with a scalar unchanged).
    std::map<std::string, long long> elems = {{"x", 4}, {"c", 1}, {"t1", 4}, {"t2", 4}, {"t3", 4}, {"t4", 4}};

    auto bytesMovedByName = [&](const Graph& g, const std::map<int, std::string>& oldName) -> long long {
        long long total = 0;
        for (const auto& n : g.nodes()) {
            if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
            for (const Value& in : n->inputs) total += elems.at(oldName.at(in.nodeId));
            total += elems.at(oldName.at(n->id));
        }
        return total;
    };
    std::map<int, std::string> origNames, uncappedNames, cappedNames;
    for (const auto& n : chain.nodes()) origNames[n->id] = n->debugName;
    for (const auto& kv : uncapped.representativeOldId) uncappedNames[kv.first] = origNames.at(kv.second);
    for (const auto& kv : capped.representativeOldId) cappedNames[kv.first] = origNames.at(kv.second);

    long long unfusedElems = bytesMovedByName(chain, origNames);
    long long uncappedElems = bytesMovedByName(uncapped.graph, uncappedNames);
    long long cappedElems = bytesMovedByName(capped.graph, cappedNames);
    long long unfusedBytes = static_cast<long long>(static_cast<double>(unfusedElems) * kBytesPerElement);
    long long uncappedBytes = static_cast<long long>(static_cast<double>(uncappedElems) * kBytesPerElement);
    long long cappedBytes = static_cast<long long>(static_cast<double>(cappedElems) * kBytesPerElement);

    printf("Bytes moved, UNFUSED (4 separate kernels):              %lld\n", unfusedBytes);
    printf("Bytes moved, UNCAPPED fusion (1 kernel, all 4 steps):   %lld\n", uncappedBytes);
    printf("Bytes moved, CAPPED fusion (2 kernels, maxChainLength=2): %lld\n", cappedBytes);

    bool cappedCostsMoreThanUncapped = (cappedBytes > uncappedBytes);
    bool cappedStillBeatsUnfused = (cappedBytes < unfusedBytes);
    printf("\nself-check: capping costs real, measured bytes relative to uncapped fusion (%lld vs.\n",
           cappedBytes);
    printf("%lld) (%s), while still moving far less than staying fully unfused (%lld) (%s) --\n",
           uncappedBytes, cappedCostsMoreThanUncapped ? "confirmed" : "MISMATCH", unfusedBytes,
           cappedStillBeatsUnfused ? "confirmed" : "MISMATCH");
    printf("a real trade-off, not a free lunch: bounded kernel size costs some of fusion's own\n");
    printf("savings, in exchange for a cap real compilers impose for reasons this book's own\n");
    printf("byte-counting was never built to measure (register pressure, compile time).\n");

    bool allOk = uncappedOk && cappedOk && sameAnswer && cappedCostsMoreThanUncapped && cappedStillBeatsUnfused;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp -o 037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow
./037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow
```

**Output:**

```text
=== Section 16.2: boundedFusionPass() -- a boundary with nothing to do with correctness ===

Original graph (6 nodes):
  %0 x = Input()
  %1 c = Const(3)
  %2 t1 = ReLU(%0)
  %3 t2 = Mul(%2, %1)
  %4 t3 = Add(%3, %1)
  %5 t4 = Mul(%4, %1)

--- UNCAPPED (Chapter 13's own elementwiseFusionPass(), no size limit) ---

  %0 x = Input()
  %1 c = Const(3)
  %2 t4 = FusedElementwise(external inputs: ext0=%0, ext1=%1)
      step0 = ReLU(ext0)
      step1 = Mul(step0, ext1)
      step2 = Add(step1, ext1)
      step3 = Mul(step2, ext1)  <- this node's own output

self-check: uncapped fusion produces ONE FusedElementwise node with all 4 steps,
6 -> 3 nodes (confirmed)

--- CAPPED (maxChainLength=2) ---

  %0 x = Input()
  %1 c = Const(3)
  %2 t2 = FusedElementwise(external inputs: ext0=%0, ext1=%1)
      step0 = ReLU(ext0)
      step1 = Mul(step0, ext1)  <- this node's own output
  %3 t4 = FusedElementwise(external inputs: ext0=%2, ext1=%1)
      step0 = Add(ext0, ext1)
      step1 = Mul(step0, ext1)  <- this node's own output

self-check: capped fusion produces TWO FusedElementwise nodes, each with AT MOST
2 steps, 6 -> 4 nodes -- t2 becomes a NEW boundary the cap alone is responsible for
(confirmed)

self-check: evaluate(original, x=2).t4 = 27, evaluate(uncapped,...) = 27,
evaluate(capped,...) = 27 -- all three agree (confirmed)

=== The cap's own real, measured cost ===

Bytes moved, UNFUSED (4 separate kernels):              140
Bytes moved, UNCAPPED fusion (1 kernel, all 4 steps):   36
Bytes moved, CAPPED fusion (2 kernels, maxChainLength=2): 72

self-check: capping costs real, measured bytes relative to uncapped fusion (72 vs.
36) (confirmed), while still moving far less than staying fully unfused (140) (confirmed) --
a real trade-off, not a free lunch: bounded kernel size costs some of fusion's own
savings, in exchange for a cap real compilers impose for reasons this book's own
byte-counting was never built to measure (register pressure, compile time).
```

!!! warning "[COMMON TRAP] A size cap changes WHERE fusion stops, never WHETHER a graph is correct"
    It is tempting to think of `maxChainLength` as a quality knob -- turn it up for "better" fusion, down for "safer" fusion. It isn't either. Every structure `boundedReductionFusionPass()` produces, at any cap value, computes the exact same answer as the uncapped version and the original unfused graph -- Section 16.2's own `evaluate()` agreement check confirms this explicitly, and Section 16.3's does again on a larger graph. The cap changes ONE thing only: how many separate kernels the computation is split across, which changes bytes moved (measurably, honestly, in the wrong direction relative to uncapped fusion) and, in a real generated-code backend Part 4 will eventually build, how much register pressure and code size any ONE of those kernels carries. Treating the cap as a correctness dial -- rather than the pure cost/practicality trade-off it actually is -- would be a real misunderstanding of what Section 16.2 built.

## 16.3 Putting It Together: A Full Fusion Report Through Chapter 8's PassManager

### Intuition

Part 3 opened with Chapter 12 asking why fusion matters at all, then spent four chapters building the machinery to actually do it -- and, just as importantly, to know precisely when NOT to. This closing section is the capstone every Part-ending chapter in this book has used since Chapter 7 and Chapter 11: one graph, built to exercise everything the Part has introduced at once, run through the real pipeline, and measured honestly -- costs included, not just the wins.

### Background

`boundedReductionFusionPass()` combines this book's two independent extensions to Chapter 13's original `mustBeExternal` check -- Chapter 14's `Sum` condition and Section 16.2's `sizeCapBoundaries` condition -- into one pass, plugged into Chapter 8's still-completely-unmodified `runPasses()` exactly the way every fusion pass in this book has been. The test graph is built specifically to trigger every boundary category from Section 16.1's own taxonomy in a single structure: a chain of four single-consumer elementwise ops long enough to trigger a size cap of 3 (only the THIRD node in the chain lands on a multiple of 3 and gets cut; the fourth does not, and inlines into whatever comes next), a shared value forcing a `Shared` boundary, a `Sum` forcing a `Reduction` boundary, and the graph's own designated output forcing a `GraphRoot` boundary. Measured through `bytesMoved()` (Chapter 13's own function, unchanged, needing no modification to handle any of this), the honest three-way comparison is exactly what Section 16.2 promised: capped fusion (244 bytes) moves strictly more than uncapped fusion (176 bytes) -- a real, measured cost -- while still moving far less than staying fully unfused (508 bytes). Nothing about the cap makes the graph MORE efficient; it trades away some of fusion's own savings for a bound real compilers need for reasons this book's own byte-counting was never built to measure in the first place.

```text
ONE GRAPH, EVERY BOUNDARY CATEGORY AT ONCE (a=[8], b=scalar):

  a                                     b
  |                                     |
  ReLU(t1) - Mul(t2) - Add(t3) - Mul(t4) - Add(t5) -+-- 2 consumers:
   depth1     depth2    depth3    depth4              s AND y2 -- SHARED
                          ^
                    SizeCap (3 % 3 == 0) -- t4 is NOT
                    (depth 4, not a multiple of 3)

  t5 --- Sum(s) --------- 1 consumer: y, but Sum --- REDUCTION
  t5 --- Mul(y2) -------- 1 consumer: y
  Add(y) ----------------  0 consumers -------------- GraphRoot

  UNFUSED:            508 bytes  (8 separate kernels)
  UNCAPPED fusion:     176 bytes  (65.4% fewer -- the maximum this
                                    graph's own structure allows)
  CAPPED fusion (3):   244 bytes  (52.0% fewer -- strictly more than
                                    uncapped, strictly less than unfused --
                                    the cap's own real, honest cost)
```

```cpp
// Chapter 16: Fusion Boundaries
// 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager.cpp
//
// Section 16.3 -- the capstone for Part 3. A single graph, built to exercise
// every boundary this Part has ever introduced at once: a SHARED value
// (Chapter 13), a REDUCTION (Chapter 14), a long single-consumer CHAIN long
// enough to trigger Section 16.2's own size cap, and the graph's own
// GraphRoot (Section 16.1). boundedReductionFusionPass() combines Chapter
// 14's own Sum-handling with Section 16.2's own size-cap mechanism into one
// pass, plugged into Chapter 8's still-completely-unmodified runPasses()
// exactly the way every fusion pass in this book has been -- and a fusion
// REPORT, built from Section 16.1's own classifyNode() extended with one
// more reason, explains every boundary in the final structure by name.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager.cpp -o 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager
// Run:     ./038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <deque>
#include <functional>
#include <algorithm>
#include <stdexcept>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

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

// ==================== Debug printer / evaluate() (from Chapter 14, unchanged) ====================

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

// ==================== boundedReductionFusionPass(): Chapter 14's Sum rule + Section 16.2's size cap ====================

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
        if (!isChainCandidate) {
            depth[id] = 0;
            continue;
        }
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

// resolveIntoGroup(), carrying BOTH of this book's own two independent
// extensions to Chapter 13's original three-condition mustBeExternal: Sum
// (Chapter 14, a correctness boundary) and sizeCapBoundaries (Section 16.2,
// a cost boundary). Neither condition knows the other exists -- they are
// simply two more entries in the same OR chain, exactly as independent in
// code as they are in the reasons behind them.
static FusedOperand resolveIntoGroupBounded(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                             const std::set<int>& sizeCapBoundaries,
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
                           (consumers.at(oldId) != 1) ||
                           (sizeCapBoundaries.count(oldId) > 0);
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
            FusedOperand summedOperand = resolveIntoGroupBounded(n->inputs[0].nodeId, g, consumers, sizeCapBoundaries,
                                                                   materialized, externalInputs, externalIndexByOldId,
                                                                   steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v;
            result.representativeOldId[v.nodeId] = oldId;
            continue;
        }
        if (consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0) continue;
        std::vector<Value> externalInputs;
        std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps;
        std::map<int, int> stepIndexByOldId;
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
        materialized[oldId] = v;
        result.representativeOldId[v.nodeId] = oldId;
    }
    return result;
}

// ==================== Section 16.1's own BoundaryReason, extended with ONE more reason ====================

enum class BoundaryReason { NotABoundary, Leaf, GraphRoot, Shared, Reduction, SizeCap };

static std::string boundaryReasonStr(BoundaryReason r) {
    switch (r) {
        case BoundaryReason::NotABoundary: return "NotABoundary";
        case BoundaryReason::Leaf:          return "Leaf";
        case BoundaryReason::GraphRoot:     return "GraphRoot";
        case BoundaryReason::Shared:        return "Shared";
        case BoundaryReason::Reduction:     return "Reduction";
        default:                            return "SizeCap";
    }
}
static std::vector<BoundaryReason> classifyNode(const Node* n, const std::map<int, int>& consumers,
                                                 const std::set<int>& sizeCapBoundaries) {
    std::vector<BoundaryReason> reasons;
    if (n->op == OpKind::Input || n->op == OpKind::Const) {
        reasons.push_back(BoundaryReason::Leaf);
        return reasons;
    }
    if (n->op == OpKind::Sum) reasons.push_back(BoundaryReason::Reduction);
    int c = consumers.at(n->id);
    if (c == 0) reasons.push_back(BoundaryReason::GraphRoot);
    else if (c >= 2) reasons.push_back(BoundaryReason::Shared);
    if (sizeCapBoundaries.count(n->id) > 0) reasons.push_back(BoundaryReason::SizeCap);
    if (reasons.empty()) reasons.push_back(BoundaryReason::NotABoundary);
    return reasons;
}
static std::string reasonsStr(const std::vector<BoundaryReason>& reasons) {
    std::string out;
    for (size_t i = 0; i < reasons.size(); ++i) {
        if (i) out += "+";
        out += boundaryReasonStr(reasons[i]);
    }
    return out;
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
        if (!topo.ok) throw std::runtime_error("runPasses: pass '" + passes[0].name + "' produced a cyclic graph");
    }
    printf("\n--- after pass '%s' (%zu nodes) ---\n", passes[0].name.c_str(), current.size());
    printGraph(current);
    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& np = passes[i];
        Graph next = np.pass(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) throw std::runtime_error("runPasses: pass '" + np.name + "' produced a cyclic graph");
        printf("\n--- after pass '%s' (%zu nodes) ---\n", np.name.c_str(), next.size());
        printGraph(next);
        current = std::move(next);
    }
    return current;
}

// ==================== bytesMoved() (from Chapter 13/14, unchanged) ====================

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

int main() {
    printf("=== Section 16.3: one graph, every boundary this Part has ever built, at once ===\n\n");

    // a=[8], b=scalar. t1..t4: a single-consumer elementwise chain long
    // enough (depth 4) to trigger a size cap of 3. t5: SHARED (2 consumers:
    // s and y2). s: Sum(t5) -- REDUCTION. y2: Mul(t5,b), single consumer.
    // y: Add(s,y2) -- the graph's own designated output, GraphRoot (0
    // consumers). Hand-derivation for maxChainLength=3: chainDepth(t1..t4)
    // = 1,2,3,4 -- only t3 (depth 3) is a multiple of 3, so t3 is the ONE
    // new SizeCap boundary; t4 is not (depth 4, not a multiple of 3), so it
    // inlines into t5's own group instead.
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

    // ---- The fusion report: classifyNode() on the ORIGINAL graph ----
    printf("--- Fusion report (Section 16.1's classifyNode(), extended with SizeCap) ---\n\n");
    std::map<int, int> consumers;
    for (const auto& n : g.nodes()) consumers[n->id] = 0;
    for (const auto& n : g.nodes())
        for (const Value& in : n->inputs) consumers[in.nodeId]++;
    std::set<int> sizeCapBoundaries = computeSizeCapBoundaries(g, consumers, 3);
    for (const auto& n : g.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumers, sizeCapBoundaries);
        printf("  %%%d %-4s consumers=%d  reasons=[%s]\n", n->id, n->debugName.c_str(),
               consumers.at(n->id), reasonsStr(reasons).c_str());
    }
    bool t3IsSizeCap = false, t4IsNotSizeCap = true;
    for (const auto& n : g.nodes()) {
        std::vector<BoundaryReason> reasons = classifyNode(n.get(), consumers, sizeCapBoundaries);
        if (n->debugName == "t3")
            t3IsSizeCap = std::find(reasons.begin(), reasons.end(), BoundaryReason::SizeCap) != reasons.end();
        if (n->debugName == "t4" && std::find(reasons.begin(), reasons.end(), BoundaryReason::SizeCap) != reasons.end())
            t4IsNotSizeCap = false;
    }
    printf("\nself-check: t3 alone carries SizeCap (chainDepth 3, a multiple of maxChainLength=3);\n");
    printf("t4 (chainDepth 4) does not (%s)\n", (t3IsSizeCap && t4IsNotSizeCap) ? "confirmed" : "MISMATCH");

    // ---- UNFUSED bytes ----
    long long unfusedBytes = bytesMoved(g, elementCounts);

    // ---- UNCAPPED fusion (maxChainLength effectively disabled: pass a huge cap) ----
    printf("\n--- UNCAPPED (boundedReductionFusionPass with no effective size limit) ---\n\n");
    TransformPass uncappedTransform = [elementCounts](const Graph& gr) {
        return boundedReductionFusionPass(gr, elementCounts, 1000000).graph;
    };
    Graph uncapped = runPasses(g, {{"boundedReductionFusionPass(uncapped)", uncappedTransform}});
    bool uncappedStructureOk = (uncapped.size() == 5);
    printf("\nself-check: uncapped fusion reaches the same structure Chapter 14's own logic would\n");
    printf("produce -- t1..t5 all fuse into ONE FusedElementwise (5 steps), s is its own\n");
    printf("FusedReduction, y2/y fuse into one more FusedElementwise -- 10 -> 5 nodes (%s)\n",
           uncappedStructureOk ? "confirmed" : "MISMATCH");

    // ---- CAPPED fusion (maxChainLength=3) ----
    printf("\n--- CAPPED (maxChainLength=3) ---\n\n");
    TransformPass cappedTransform = [elementCounts](const Graph& gr) {
        return boundedReductionFusionPass(gr, elementCounts, 3).graph;
    };
    Graph capped = runPasses(g, {{"boundedReductionFusionPass(cap=3)", cappedTransform}});
    bool cappedStructureOk = (capped.size() == 6);
    printf("\nself-check: capped fusion introduces exactly ONE extra materialization point (t3),\n");
    printf("10 -> 6 nodes instead of 10 -> 5 (%s)\n", cappedStructureOk ? "confirmed" : "MISMATCH");

    // ---- Bytes moved: all three, side by side ----
    printf("\n=== Bytes moved: unfused vs. uncapped fusion vs. capped fusion ===\n\n");

    FusionResult uncappedResult = boundedReductionFusionPass(g, elementCounts, 1000000);
    FusionResult cappedResult = boundedReductionFusionPass(g, elementCounts, 3);
    std::map<int, long long> uncappedElementCounts, cappedElementCounts;
    for (const auto& n : uncappedResult.graph.nodes())
        uncappedElementCounts[n->id] = elementCounts.at(uncappedResult.representativeOldId.at(n->id));
    for (const auto& n : cappedResult.graph.nodes())
        cappedElementCounts[n->id] = elementCounts.at(cappedResult.representativeOldId.at(n->id));

    long long uncappedBytes = bytesMoved(uncappedResult.graph, uncappedElementCounts);
    long long cappedBytes = bytesMoved(cappedResult.graph, cappedElementCounts);

    printf("Bytes moved, UNFUSED (8 separate kernels):        %lld\n", unfusedBytes);
    printf("Bytes moved, UNCAPPED fusion (3 fused kernels):   %lld\n", uncappedBytes);
    printf("Bytes moved, CAPPED fusion, maxChainLength=3\n");
    printf("             (4 fused kernels):                  %lld\n", cappedBytes);

    bool bytesOrderingOk = (uncappedBytes < cappedBytes) && (cappedBytes < unfusedBytes);
    double uncappedReduction = 1.0 - static_cast<double>(uncappedBytes) / static_cast<double>(unfusedBytes);
    double cappedReduction = 1.0 - static_cast<double>(cappedBytes) / static_cast<double>(unfusedBytes);
    printf("\nuncapped fusion: %.1f%% fewer bytes than unfused. capped fusion: %.1f%% fewer bytes\n",
           uncappedReduction * 100.0, cappedReduction * 100.0);
    printf("than unfused -- still a large real saving, just not the maximum this graph's own\n");
    printf("structure would otherwise allow.\n");
    printf("\nself-check: uncapped < capped < unfused, strictly, on real measured bytes (%s)\n",
           bytesOrderingOk ? "confirmed" : "MISMATCH");

    // ---- Correctness: evaluate() agreement across all three ----
    printf("\n=== Correctness: evaluate() agreement across original, uncapped, and capped ===\n\n");
    std::map<std::string, float> in = {{"a", 2.0f}, {"b", 3.0f}};
    float origOut = evaluate(g, in, elementCounts).at("y");
    float uncappedOut = evaluate(uncapped, in, {}).at("y");
    float cappedOut = evaluate(capped, in, {}).at("y");
    bool sameAnswer = (origOut == uncappedOut) && (origOut == cappedOut);
    printf("evaluate(original, a=2,b=3).y = %g\n", origOut);
    printf("evaluate(uncapped, ...).y     = %g\n", uncappedOut);
    printf("evaluate(capped, ...).y       = %g\n", cappedOut);
    printf("self-check: all three agree (%s)\n", sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = t3IsSizeCap && t4IsNotSizeCap && uncappedStructureOk && cappedStructureOk &&
                 bytesOrderingOk && sameAnswer;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager.cpp -o 038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager
./038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager
```

**Output:**

```text
=== Section 16.3: one graph, every boundary this Part has ever built, at once ===

--- Fusion report (Section 16.1's classifyNode(), extended with SizeCap) ---

  %0 a    consumers=1  reasons=[Leaf]
  %1 b    consumers=5  reasons=[Leaf]
  %2 t1   consumers=1  reasons=[NotABoundary]
  %3 t2   consumers=1  reasons=[NotABoundary]
  %4 t3   consumers=1  reasons=[SizeCap]
  %5 t4   consumers=1  reasons=[NotABoundary]
  %6 t5   consumers=2  reasons=[Shared]
  %7 s    consumers=1  reasons=[Reduction]
  %8 y2   consumers=1  reasons=[NotABoundary]
  %9 y    consumers=0  reasons=[GraphRoot]

self-check: t3 alone carries SizeCap (chainDepth 3, a multiple of maxChainLength=3);
t4 (chainDepth 4) does not (confirmed)

--- UNCAPPED (boundedReductionFusionPass with no effective size limit) ---

--- before any pass (10 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 t1 = ReLU(%0)
  %3 t2 = Mul(%2, %1)
  %4 t3 = Add(%3, %1)
  %5 t4 = Mul(%4, %1)
  %6 t5 = Add(%5, %1)
  %7 s = Sum(%6)
  %8 y2 = Mul(%6, %1)
  %9 y = Add(%7, %8)

--- after pass 'boundedReductionFusionPass(uncapped)' (5 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 t5 = FusedElementwise(external inputs: ext0=%0, ext1=%1)
      step0 = ReLU(ext0)
      step1 = Mul(step0, ext1)
      step2 = Add(step1, ext1)
      step3 = Mul(step2, ext1)
      step4 = Add(step3, ext1)  <- this node's own output
  %3 s = FusedReduction(external inputs: ext0=%2)
      step0 = Sum(ext0)  [reduces 8 elements]  <- this node's own output
  %4 y = FusedElementwise(external inputs: ext0=%3, ext1=%2, ext2=%1)
      step0 = Mul(ext1, ext2)
      step1 = Add(ext0, step0)  <- this node's own output

self-check: uncapped fusion reaches the same structure Chapter 14's own logic would
produce -- t1..t5 all fuse into ONE FusedElementwise (5 steps), s is its own
FusedReduction, y2/y fuse into one more FusedElementwise -- 10 -> 5 nodes (confirmed)

--- CAPPED (maxChainLength=3) ---

--- before any pass (10 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 t1 = ReLU(%0)
  %3 t2 = Mul(%2, %1)
  %4 t3 = Add(%3, %1)
  %5 t4 = Mul(%4, %1)
  %6 t5 = Add(%5, %1)
  %7 s = Sum(%6)
  %8 y2 = Mul(%6, %1)
  %9 y = Add(%7, %8)

--- after pass 'boundedReductionFusionPass(cap=3)' (6 nodes) ---
  %0 a = Input()
  %1 b = Input()
  %2 t3 = FusedElementwise(external inputs: ext0=%0, ext1=%1)
      step0 = ReLU(ext0)
      step1 = Mul(step0, ext1)
      step2 = Add(step1, ext1)  <- this node's own output
  %3 t5 = FusedElementwise(external inputs: ext0=%2, ext1=%1)
      step0 = Mul(ext0, ext1)
      step1 = Add(step0, ext1)  <- this node's own output
  %4 s = FusedReduction(external inputs: ext0=%3)
      step0 = Sum(ext0)  [reduces 8 elements]  <- this node's own output
  %5 y = FusedElementwise(external inputs: ext0=%4, ext1=%3, ext2=%1)
      step0 = Mul(ext1, ext2)
      step1 = Add(ext0, step0)  <- this node's own output

self-check: capped fusion introduces exactly ONE extra materialization point (t3),
10 -> 6 nodes instead of 10 -> 5 (confirmed)

=== Bytes moved: unfused vs. uncapped fusion vs. capped fusion ===

Bytes moved, UNFUSED (8 separate kernels):        508
Bytes moved, UNCAPPED fusion (3 fused kernels):   176
Bytes moved, CAPPED fusion, maxChainLength=3
             (4 fused kernels):                  244

uncapped fusion: 65.4% fewer bytes than unfused. capped fusion: 52.0% fewer bytes
than unfused -- still a large real saving, just not the maximum this graph's own
structure would otherwise allow.

self-check: uncapped < capped < unfused, strictly, on real measured bytes (confirmed)

=== Correctness: evaluate() agreement across original, uncapped, and capped ===

evaluate(original, a=2,b=3).y = 330
evaluate(uncapped, ...).y     = 330
evaluate(capped, ...).y       = 330
self-check: all three agree (confirmed)
```

!!! note "Part 3, closed: what CUDA Hammer's fusion machinery can and cannot do"
    Four chapters, one unmodified `PassManager` contract, and a small, closed set of reasons a fusion opportunity has to stop: sharing (Chapter 13), a shape change (Chapter 14), loop-nest incompatibility (Chapter 15, a separate later question rather than a `resolveIntoGroup()` condition at all), and now a cost-based size limit (this chapter) -- the first boundary in this book with nothing to do with correctness. `classifyNode()` (Section 16.1) turns all of the correctness-driven ones into an inspectable, checked taxonomy rather than logic buried inside a pass's own control flow. None of this is the complete list a production tensor compiler would need -- memory layout mismatches, control-flow boundaries, and multi-output kernels are all real reasons XLA, TVM, and Triton (Chapter 3) have to stop fusing that CUDA Hammer's own single-output, layout-free IR has never had occasion to model -- and that's a stated, honest limitation, not a gap this chapter tries to paper over. Part 4 starts from here: CUDA Hammer's own fused IR, exactly as this Part left it, gets lowered to real loops for the first time.

## Chapter Summary

This chapter closed Part 3 by gathering, extending, and finally exercising every fusion boundary this book has built. Section 16.1 introduced `classifyNode()`, a diagnostic taxonomy re-deriving Chapters 13-14's own boundary conditions (Leaf, Shared, Reduction, and the newly-named GraphRoot for a zero-consumer sink) as an explicit, inspectable set of reasons rather than a side effect buried inside `resolveIntoGroup()`'s own control flow -- checked, not just asserted, by confirming its predictions against the real passes' own actual output on two different graphs, including one (Chapter 14's own Graph B) that genuinely has two independent GraphRoot sinks rather than the one Chapter 9's own convention usually assumes. Section 16.2 built this book's first boundary with nothing to do with correctness: a cost-based size cap, computed as a separate pre-pass (`computeSizeCapBoundaries()`) rather than threaded live through `resolveIntoGroup()`'s own bottom-up recursion, where such a live check provably cannot work -- and measured its own real, honest cost (72 vs. 36 bytes on a small isolated chain) rather than presenting it as a free improvement. Section 16.3 closed the chapter and the Part by combining both of this book's own independent `resolveIntoGroup()` extensions -- Chapter 14's `Sum` condition and Section 16.2's size cap -- into `boundedReductionFusionPass()`, plugged into Chapter 8's still-completely-unmodified `PassManager`, and measured on one graph exercising every boundary category at once: 508 bytes unfused, 176 with full (uncapped) fusion, 244 with a size cap of 3 -- a real, honest three-way comparison, evaluate()-agreement confirmed across all three.

## Self-Check Questions

1. `classifyNode()` is described as adding no new fusion LOGIC. What, precisely, does it add, and why does `verifyClassificationAgreesWithPass()` matter for that claim to be trustworthy rather than just plausible?
2. Section 16.1's own Graph 2 (Chapter 14's Graph B) has two nodes classified `GraphRoot` rather than one. Why does this graph have two, and what does that reveal about a claim this chapter deliberately did NOT make about `GraphRoot`?
3. Section 16.2 considered, and rejected, checking the size cap live inside `resolveIntoGroup()`'s own recursion. What specifically goes wrong with that approach, and why does it fail silently rather than obviously?
4. `computeSizeCapBoundaries()` marks a node as a boundary when its own chain depth is an exact multiple of `maxChainLength`. In Section 16.2's own isolated chain example (`t1`-`t4`, `maxChainLength=2`), which node gets marked, and why does `t4` NOT need the cap to already be forced external?
5. `boundedFusionPass()`'s own top-level walk needs a change beyond just `resolveIntoGroup()`'s new condition. What is that change, and what specific bug would appear without it -- and which earlier chapter needed the exact same KIND of fix, for a different reason?
6. Section 16.2's own [COMMON TRAP] warns against treating `maxChainLength` as a correctness dial. What evidence, specifically, backs up the claim that it isn't one?
7. Section 16.3's own capstone graph is deliberately built to trigger every category from Section 16.1's own taxonomy at once. Name each category and the specific node (or part of the graph) responsible for it.
8. This chapter's own closing note lists real fusion boundaries production compilers need that CUDA Hammer's own IR has never had to model. Why is naming these explicitly, rather than silently leaving them out, consistent with this book's own standing discipline?

## Where We Go Next

Part 3 is complete: fusion now has a real IR (Chapters 13-14), a real measured payoff (Chapter 12), an explicit loop-level view (Chapter 15), and a closed, checked taxonomy of exactly when it has to stop (this chapter). None of it has produced a single line of code that could actually run on real hardware yet -- every measurement in Part 3 has been counted bytes and FLOPs, not generated instructions. Part 4, "Code Generation," starts exactly there: Chapter 17, "Lowering CUDA Hammer's IR to Loops," takes the fused IR this Part produced -- FusedElementwise and FusedReduction nodes, their own LoopNests from Chapter 15 -- and turns it into real, executable loop structures for the first time. Apply the Chapter 5-16 depth-level standard (more prose, more diagrams before code) there too.

## Worked Solutions

1. It adds a NAME and a single, reusable ENTRY POINT for logic that Chapters 13-14 already computed as a side effect of deciding whether to inline a node -- nothing about WHICH nodes are boundaries, or why, changes. `verifyClassificationAgreesWithPass()` matters because a diagnostic function that merely LOOKS plausible but has quietly drifted out of sync with the real pass's own actual behavior would be actively misleading -- worse than having no diagnostic at all, since it would confidently give wrong answers. Checking it against the pass's own real, executed output on real graphs is what turns "this looks right" into "this is right."
2. This graph was never run through Chapter 9's own `deadCodeEliminationPass()`, so nothing has ever enforced that only ONE node has zero consumers -- `y` and `y2` are both independent, unconsumed sinks, and `reductionFusionPass()` (like `boundedReductionFusionPass()` later) materializes both as their own roots without caring how many there are. This reveals that `GraphRoot` does NOT mean "the graph's one true designated output" -- that uniqueness is a property Chapter 9's own convention assumes for LIVENESS purposes, not one fusion itself was ever told to enforce, and `classifyNode()` deliberately doesn't overclaim it.
3. `resolveIntoGroup()`'s own recursion resolves a chain's DEEPEST node FIRST, because every node's operands are resolved before its own step gets pushed. A check made at the START of each recursive call would see `steps.size() == 0` for every node on the way down a chain, no matter how deep -- it would never observe the group "getting big" during the descent, only discover it after everything had already been inlined on the way back up. It fails silently rather than obviously because the resulting fused group would simply be larger than the cap intended, with no crash, no wrong answer, and no signal that the cap was never actually enforced.
4. In the `t1`-`t4` example with `maxChainLength=2`: `chainDepth(t1)=1`, `chainDepth(t2)=2`, `chainDepth(t3)=3`, `chainDepth(t4)=4`. Both `t2` (depth 2) and `t4` (depth 4) are multiples of 2, so both get marked -- but `t4` was ALREADY going to be its own root regardless, because it has zero consumers (`GraphRoot`, Section 16.1's own category), independent of the cap entirely. `t2` is the one node where the cap is the ONLY reason it becomes a boundary -- without it, `t2` would have inlined into `t3`, which would have inlined into `t4`, producing one 4-step group instead of two 2-step groups.
5. The top-level pass loop's own "consumers == 1, skip" shortcut has to ALSO check the size-cap set (`consumers.at(oldId) == 1 && sizeCapBoundaries.count(oldId) == 0`), not just `resolveIntoGroup()`'s own new condition. Without it, a node the cap marks as a boundary but whose consumer count is still exactly 1 would be skipped by this loop (deferring to its one consumer to build it), and then REFUSED by `resolveIntoGroup()` when that consumer tries to inline it (since the cap condition now says external) -- with nothing left to materialize it at all, it would silently vanish from the output graph. Chapter 14 needed the exact same kind of fix for `Sum`: a new `resolveIntoGroup()` condition alone was not enough: the top-level walk needed a matching change too, or a single-consumer `Sum` node would have vanished the same way.
6. Every structure `boundedReductionFusionPass()` produces, at any `maxChainLength`, passes an `evaluate()` agreement check against the original, unfused graph -- Section 16.2's own isolated chain (`t4 = 27` at `x=2`, all three variants agreeing) and Section 16.3's own capstone graph (`y = 330` at `a=2,b=3`, all three variants agreeing) both confirm this directly, by actually running the computation and comparing real numbers, not by inspecting the code and asserting it should be fine.
7. `Leaf`: `a` and `b`, the two `Input` nodes. `SizeCap`: `t3` (chain depth 3, a multiple of `maxChainLength=3`). `Shared`: `t5` (two consumers, `s` and `y2`). `Reduction`: `s` (a `Sum` node, regardless of its own single consumer). `GraphRoot`: `y` (zero consumers, the graph's own designated output). `NotABoundary`: `t1`, `t2`, `t4`, and `y2` -- each inlines into its own single consumer.
8. This book's standing discipline, since at least Chapter 10's own commutativity discussion and Chapter 13's own single-consumer limitation, has been to state a real limitation explicitly rather than let a chapter's own scope quietly stand in for completeness it was never claiming. Memory layout, control flow, and multi-output kernels are real reasons production compilers refuse to fuse that CUDA Hammer's own IR has simply never needed to represent (no layouts, no control flow, one output per node) -- naming them keeps this chapter's own taxonomy honest about being a taxonomy of what CUDA HAMMER can currently reason about, not a claim that the list is exhaustive for tensor compilers in general.

---

**Sources cited in this chapter:**

None new. The size-cap boundary is original to this book, motivated by the same kind of practical, real-compiler concern (bounding a single fused kernel's own complexity) that XLA, TVM, and Triton -- surveyed in Chapter 3 -- all have to address in their own real implementations, without this chapter attributing any specific number or policy to any of them.
