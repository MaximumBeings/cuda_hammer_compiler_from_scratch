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
