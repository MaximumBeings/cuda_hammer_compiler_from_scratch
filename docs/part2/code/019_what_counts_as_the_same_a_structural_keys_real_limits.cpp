// Chapter 10: Common Subexpression Elimination for Tensor Graphs
// 019_what_counts_as_the_same_a_structural_keys_real_limits.cpp
//
// Section 10.2 -- Section 10.1's commonSubexpressionEliminationPass()
// merges two nodes exactly when they produce an IDENTICAL cseKey(). It
// is worth being precise about what "identical" means there, because it
// is narrower than "always computes the same number" -- and the gap
// between those two things is a real, honestly-stated limitation of
// this pass, in the same spirit as Section 9.2's explicit "the last
// node is the output" convention: a real simplification, not a hidden
// one, kept because it is simple and safe, not because it is complete.
//
// This file demonstrates BOTH directions of that gap with real,
// executed code -- not just an assertion in prose:
//
//   * a case cseKey() is TOO CONSERVATIVE about (Section 10.2, part 1):
//     add(a, b) and add(b, a) compute the exact same number for every
//     real input -- addition is commutative -- but cseKey() encodes
//     OPERAND ORDER (its key for add(a,b) is "add:a,b", not some
//     order-independent combination of a and b), so these two nodes get
//     DIFFERENT keys and are never merged. The pass still produces a
//     CORRECT answer either way -- it just misses a fusion opportunity
//     a smarter, order-aware key could have found.
//
//   * a case cseKey() gets RIGHT for a good reason (Section 10.2, part
//     2): two SEPARATE Input nodes that happen to share the same name
//     really do represent the identical runtime value under this book's
//     own evaluate() (Chapter 9), which looks an input up purely by
//     name -- so merging them is not a missed shortcut, it is simply
//     correct, and cseKey()'s name-based key for Input nodes gets this
//     right without any special-casing at all.
//
// Why not just fix the commutative case by sorting operand ids before
// building the key? Because that fix only works when BOTH the operation
// is actually commutative for every possible operand AND there is a
// stable way to choose a canonical order -- Add and Mul both qualify
// here, but a general tensor compiler has operations that are emphat-
// ically NOT commutative (subtraction, matrix multiplication, ReLU
// applied after something), and teaching cseKey() which ops are safe to
// reorder is real added complexity this chapter leaves as an open
// extension rather than folding in silently. Getting it wrong in the
// unsafe direction -- merging two nodes that are not actually
// equivalent -- would be a correctness bug, not just a missed
// optimization; this chapter's own commonSubexpressionEliminationPass()
// stays on the conservative side of that line on purpose.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 019_what_counts_as_the_same_a_structural_keys_real_limits.cpp -o 019_what_counts_as_the_same_a_structural_keys_real_limits
// Run:     ./019_what_counts_as_the_same_a_structural_keys_real_limits
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <functional>
#include <stdexcept>

// ==================== Value / Node / Graph (from Chapter 4) ====================

struct Value {
    int nodeId = -1;
    int outputIndex = 0;
};

enum class OpKind { Input, Const, Add, Mul, ReLU };

static std::string opKindStr(OpKind op) {
    switch (op) {
        case OpKind::Input: return "Input";
        case OpKind::Const: return "Const";
        case OpKind::Add:   return "Add";
        case OpKind::Mul:   return "Mul";
        default:            return "ReLU";
    }
}

struct Node {
    int id;
    OpKind op;
    std::string debugName;
    std::vector<Value> inputs;
    float constValue = 0.0f;
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

// ==================== evaluate() (from Chapter 9, unchanged) ====================

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
        } else { // ReLU
            v = std::max(0.0f, valuesById.at(n->inputs[0].nodeId));
        }
        valuesById[id] = v;
        valuesByName[n->debugName] = v;
    }
    return valuesByName;
}

// ==================== commonSubexpressionEliminationPass() (from Section 10.1, unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;

static std::string cseKey(const Node* n, const std::map<int, Value>& oldIdToNewValue) {
    char buf[128];
    switch (n->op) {
        case OpKind::Input:
            return "input:" + n->debugName;
        case OpKind::Const:
            snprintf(buf, sizeof(buf), "const:%.9g", n->constValue);
            return buf;
        case OpKind::ReLU: {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            snprintf(buf, sizeof(buf), "relu:%d", in.nodeId);
            return buf;
        }
        default: { // Add or Mul -- NOT order-independent: "add:X,Y" and
                   // "add:Y,X" are different strings, on purpose (see
                   // this file's own top-of-file comment).
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            snprintf(buf, sizeof(buf), "%s:%d,%d", n->op == OpKind::Add ? "add" : "mul", lhs.nodeId, rhs.nodeId);
            return buf;
        }
    }
}

static Graph commonSubexpressionEliminationPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("commonSubexpressionEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    std::map<std::string, Value> keyToExistingValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string key = cseKey(n, oldIdToNewValue);

        auto found = keyToExistingValue.find(key);
        if (found != keyToExistingValue.end()) {
            oldIdToNewValue[oldId] = found->second;
            continue;
        }

        Value newValue;
        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->inputs.size() == 1) {
            newValue = result.addUnary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId), n->debugName);
        } else {
            newValue = result.addBinary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId),
                                         oldIdToNewValue.at(n->inputs[1].nodeId), n->debugName);
        }
        keyToExistingValue[key] = newValue;
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

static void printGraph(const Graph& g) {
    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s)", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
        if (n->op == OpKind::Const) printf(" [value=%g]", n->constValue);
        printf("\n");
    }
}

int main() {
    printf("=== Section 10.2: what commonSubexpressionEliminationPass() does and does NOT recognize as duplicate ===\n\n");

    // Part 1: t1 = add(a, b), t2 = add(b, a) -- mathematically identical
    // for every real a and b, but cseKey() encodes operand ORDER, so
    // these get DIFFERENT keys and are NOT merged.
    Graph commutativeCase;
    Value a = commutativeCase.addInput("a");
    Value b = commutativeCase.addInput("b");
    Value t1 = commutativeCase.addBinary(OpKind::Add, a, b, "t1");  // add(a, b)
    Value t2 = commutativeCase.addBinary(OpKind::Add, b, a, "t2");  // add(b, a) -- operands swapped
    commutativeCase.addBinary(OpKind::Add, t1, t2, "out");

    printf("Part 1 -- add(a,b) and add(b,a), operand order swapped:\n");
    printGraph(commutativeCase);

    Graph commutativeCsed = commonSubexpressionEliminationPass(commutativeCase);
    printf("\nAfter commonSubexpressionEliminationPass():\n");
    printGraph(commutativeCsed);

    bool commutativeCaseUnmerged = (commutativeCsed.size() == commutativeCase.size());
    printf("\nself-check: add(a,b) and add(b,a) are NOT merged -- size stays %zu -> %zu (%s)\n",
           commutativeCase.size(), commutativeCsed.size(), commutativeCaseUnmerged ? "confirmed" : "MISMATCH");

    // Missing the merge does not make the ANSWER wrong -- just less
    // optimized than it could be. Confirm the result is still correct.
    std::map<std::string, float> commInputs = {{"a", 3.0f}, {"b", 9.0f}};
    float commOriginal = evaluate(commutativeCase, commInputs).at("out");
    float commCsed = evaluate(commutativeCsed, commInputs).at("out");
    bool commAnswerStillCorrect = (commOriginal == commCsed) && (commOriginal == 24.0f);  // (3+9)+(9+3) = 24
    printf("self-check: the un-merged result is still numerically correct, %g == %g == 24 (%s)\n",
           commOriginal, commCsed, commAnswerStillCorrect ? "confirmed" : "MISMATCH");

    // Part 2: two SEPARATE Input nodes that share the same name. Under
    // this book's own evaluate() (Chapter 9), a node with debugName "x"
    // is fed whatever inputValuesByName["x"] holds, regardless of WHICH
    // Input node asked -- so two Input("x") nodes really are the same
    // runtime value, and cseKey()'s plain name-based key for Input
    // merges them correctly, with no special-casing needed.
    Graph duplicateInputCase;
    Value x1 = duplicateInputCase.addInput("x");
    Value y  = duplicateInputCase.addInput("y");
    Value r1 = duplicateInputCase.addBinary(OpKind::Add, x1, y, "r1");
    Value x2 = duplicateInputCase.addInput("x");  // SAME name as x1, a separate node
    Value r2 = duplicateInputCase.addBinary(OpKind::Add, x2, y, "r2");
    duplicateInputCase.addBinary(OpKind::Add, r1, r2, "out");

    printf("\nPart 2 -- two separate Input nodes both named 'x':\n");
    printGraph(duplicateInputCase);

    Graph duplicateInputCsed = commonSubexpressionEliminationPass(duplicateInputCase);
    printf("\nAfter commonSubexpressionEliminationPass():\n");
    printGraph(duplicateInputCsed);

    // 6 nodes -> 4: x2 merges into x1 (same name), which cascades into
    // r2 merging into r1 (same operands once x2 == x1), exactly the
    // cascade Section 10.1's own Const test demonstrated.
    bool duplicateInputCascaded = (duplicateInputCsed.size() == 4);
    printf("\nself-check: the two same-named Input nodes merged, cascading into their consumers, %zu -> %zu (%s)\n",
           duplicateInputCase.size(), duplicateInputCsed.size(), duplicateInputCascaded ? "confirmed" : "MISMATCH");

    std::map<std::string, float> dupInputs = {{"x", 2.0f}, {"y", 10.0f}};
    float dupOriginal = evaluate(duplicateInputCase, dupInputs).at("out");
    float dupCsed = evaluate(duplicateInputCsed, dupInputs).at("out");
    bool dupAnswerCorrect = (dupOriginal == dupCsed) && (dupOriginal == 24.0f);  // (2+10)+(2+10) = 24
    printf("self-check: evaluate(original, x=2,y=10).out = %g, evaluate(deduped, ...).out = %g (%s)\n",
           dupOriginal, dupCsed, dupAnswerCorrect ? "confirmed" : "MISMATCH");

    bool allOk = commutativeCaseUnmerged && commAnswerStillCorrect && duplicateInputCascaded && dupAnswerCorrect;
    return allOk ? 0 : 1;
}
