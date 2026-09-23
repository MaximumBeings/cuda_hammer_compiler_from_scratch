# 10. Common Subexpression Elimination for Tensor Graphs

**What you will understand:** the book's third genuinely *optimizing* transform pass -- common subexpression elimination (CSE), which notices when two different nodes compute the exact same thing and collapses them into one -- exactly what "the exact same thing" means for a simple structural key, where that definition is deliberately conservative, and how constant folding can expose a CSE opportunity CSE could never have found on its own.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph` and `topologicalSort()`, Chapter 7's `printGraphAsSource()`, Chapter 8's `TransformPass` and `runPasses()`, and Chapter 9's `constantFoldPass()`, `deadCodeEliminationPass()`, and `evaluate()` -- this chapter adds one more pass to that same, unmodified pipeline, and reuses Chapter 9's own passes directly in Section 10.3.

---

Chapter 9 built two optimizing passes that each ask a narrow question about a single node in isolation: "is this computation's answer already fully known?" (constant folding), and "does anything still need this computation at all?" (dead code elimination). This chapter's pass asks a different kind of question, one that looks *across* the graph rather than at one node at a time: "has this exact computation already been done somewhere else?" A real tensor program can compute the same thing twice without anyone intending it -- the same normalization applied along two branches that happen to read the same two tensors, say -- and common subexpression elimination is the pass that notices the duplication and collapses it down to one computation, reused everywhere it was needed.

```text
WHAT THIS CHAPTER ADDS TO CHAPTERS 8 AND 9's OWN INFRASTRUCTURE:

  Chapter 8's passes:          Chapter 9's passes:              THIS chapter's pass:

    identityPass                 constantFoldPass                 commonSubexpressionEliminationPass
      (changes nothing)            (replaces a computation           (reuses an existing computation
    canonicalizeNodeNames          whose answer is already            instead of repeating it)
      (changes NAMES only)         known, with that answer)

                                  deadCodeEliminationPass
                                    (drops a node nothing
                                     actually depends on)

  all four have the EXACT signature Chapter 8 already defined --
  Graph(const Graph&) -- so all four plug straight into runPasses()
  with ZERO changes to Chapter 8's own PassManager.
```

## 10.1 Common Subexpression Elimination: Reusing What's Already Computed

### Intuition

Picture two people on the same team, working on different parts of a shared spreadsheet, who each independently need "revenue minus cost" for the same quarter. If neither one knows the other already computed it, they each write their own formula, and the spreadsheet now computes the identical subtraction twice -- more cells to maintain, more chances for the two copies to quietly drift apart if one gets edited later and the other doesn't. A good spreadsheet reviewer notices the duplication and points the second formula at the first one's cell instead: one computation, reused wherever the answer is needed. Common subexpression elimination is that reviewer, applied to a graph instead of a spreadsheet -- it looks for two nodes computing the identical thing and keeps only one of them, redirecting every consumer of the dropped node to the one that survives.

```text
TWO NODES THAT COMPUTE THE IDENTICAL THING, BEFORE AND AFTER CSE:

  before:                              after:

    a   b                                a   b
     \ /                                  \ /
    t1=add(a,b)     t3=add(a,b)          t1=add(a,b)
        |                |                   |        \
    t2=mul(t1,a)     t4=mul(t3,b)       t2=mul(t1,a)   t4=mul(t1,b)

  t1 and t3 both compute add(a,b) -- t3 is simply never built at all;
  t4 is rebuilt reading from t1 directly, and t3's own name disappears.
```

Chapter 9's `constantFoldPass()` and `deadCodeEliminationPass()` both worked by looking at ONE node's own inputs. CSE has to compare a node against every OTHER node that came before it in the same graph -- a genuinely different kind of check, though it turns out to reuse the exact same "rebuild in topological order, remap by id" skeleton every `TransformPass` in this book has used since Chapter 8.

### Background

`commonSubexpressionEliminationPass()` walks the graph in topological order, exactly like `constantFoldPass()` and `deadCodeEliminationPass()` before it. Before building each node, it first computes a **canonical key** describing exactly what that node computes: for an `Input`, its own name; for a `Const`, its literal value; for an `Add`, `Mul`, or `ReLU`, its operation plus the *already-rebuilt* ids of its own operands. If some earlier node in this same walk already produced an identical key, the new node is never built at all -- everything that would have consumed it is simply pointed at the node that already exists.

```text
cseKey() FOR EACH KIND OF NODE:

  Input   -->  "input:" + the node's own debugName
  Const   -->  "const:" + the node's own literal value
  ReLU    -->  "relu:"  + the NEW id of its (already-rebuilt) operand
  Add     -->  "add:"   + the NEW ids of its (already-rebuilt) operands, IN ORDER
  Mul     -->  "mul:"   + the NEW ids of its (already-rebuilt) operands, IN ORDER

  two nodes with an IDENTICAL key compute, by construction, the
  identical thing -- whichever one is seen SECOND is redundant.
```

The word "already-rebuilt" is doing real work in that table, for the same reason it mattered in `constantFoldPass()`: the key for `Add`/`Mul`/`ReLU` is built from `oldIdToNewValue`, the map from old ids to the *new* graph's own ids, populated as the walk proceeds. A merge discovered early in the walk is immediately visible to whatever gets keyed later -- so if two `Const` nodes merge first, and two `Add` nodes each consumed one of those `Const`s, those two `Add` nodes now have identical keys too, and merge in the very same pass. Checking the old graph's ids instead would miss this entirely: the old graph never changes, so a node that consumed one of two now-merged duplicates would still see them as two *different* ids, and the second-level merge would be invisible until the pass ran again.

```text
A CASCADE, DISCOVERED IN ONE SINGLE PASS:

  before CSE:                        walking in topological order...

    k1=const(7)   k2=const(7)          k1 -> new key "const:7" -> KEPT (first time)
       |              |                k2 -> new key "const:7" -> MERGED into k1
    r1=add(k1,m)  r2=add(k2,m)         r1 -> new key "add:k1,m" -> KEPT (first time)
       \              /                r2 -> new key "add:k1,m" (k2 already remapped
        out=add(r1,r2)                       to k1!) -> MERGED into r1

  after CSE:  k1=const(7)  ->  r1=add(k1,m)  ->  out=add(r1,r1)
```

```cpp
// Chapter 10: Common Subexpression Elimination for Tensor Graphs
// 018_common_subexpression_elimination_reuses_what_already_exists.cpp
//
// Section 10.1 -- common subexpression elimination (CSE), this book's
// THIRD TransformPass, plugging into the exact same Chapter 8 machinery
// Section 9.1's constantFoldPass() and Section 9.2's
// deadCodeEliminationPass() already plug into, with zero changes to
// that machinery required.
//
// Constant folding (9.1) asks "is this computation's answer already
// fully known?" Dead code elimination (9.2) asks "does anything still
// need this computation at all?" Common subexpression elimination asks
// a third, different question: "has this EXACT computation already been
// done somewhere else in this graph?" A real tensor program can easily
// compute the same thing twice without anyone intending it -- the same
// normalization applied along two different code paths that happen to
// read the same two tensors, say -- and CSE's job is to notice and
// collapse that duplication down to one computation, reused everywhere
// it was needed.
//
// "Exact computation," here, means something very specific: the SAME
// operation, applied to the SAME operands, where "the same operands"
// means the same already-deduplicated NODES in the graph being rebuilt
// -- not merely operands that would happen to evaluate to equal numbers
// at runtime. A Const(5) and an Input node that happens to receive 5.0
// at runtime are not "the same" to this pass; it has no way to know
// what an Input will be fed at runtime, and correctly does not try to
// guess. This file characterizes exactly what CSE built this way DOES
// catch. Section 10.2 characterizes what it deliberately does NOT.
//
// commonSubexpressionEliminationPass() rebuilds the graph in
// topological order, exactly like every earlier TransformPass in this
// book -- except before adding a new node, it computes a CANONICAL KEY
// describing exactly what that node computes, and checks whether some
// EARLIER node in this same walk already produced an identical key. If
// one did, the duplicate is never built at all: every later consumer is
// pointed at the node that already exists. This is the same "just don't
// add it to the new graph" idiom Section 9.2 used to drop nodes nothing
// needed -- here, a node is left out not because nothing needs it, but
// because something IDENTICAL already exists for everything to share.
//
// Because a node's key is built from the NEW graph's own (already
// deduplicated) ids -- via oldIdToNewValue, populated as the walk
// proceeds, the exact "check the new graph, not the old one" discipline
// constantFoldPass() used to fold an entire chain in a single pass --
// a merge discovered early in this walk is immediately visible to
// whatever gets keyed later. That lets a single CSE pass discover a
// CASCADE of merges: two duplicate leaves merging first can make two
// nodes that each consume one of them duplicates of each other too,
// discovered in the very same walk, with no need to run the pass twice.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 018_common_subexpression_elimination_reuses_what_already_exists.cpp -o 018_common_subexpression_elimination_reuses_what_already_exists
// Run:     ./018_common_subexpression_elimination_reuses_what_already_exists
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
//
// Keyed by node NAME, not id, for the exact reason Chapter 9's own
// evaluate() comment explains: a rebuilt graph's ids are not guaranteed
// to match the original's once a graph has more than one
// simultaneously-ready leaf, and this chapter's own passes are no
// exception to that rule.
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

static const Node* findNodeByName(const Graph& g, const std::string& name) {
    for (const auto& n : g.nodes()) {
        if (n->debugName == name) return n.get();
    }
    throw std::runtime_error("findNodeByName: no node named '" + name + "'");
}

// ============================== Section 10.1: commonSubexpressionEliminationPass() (new) ==============================

using TransformPass = std::function<Graph(const Graph&)>;

// The canonical key describing exactly what a node computes, in terms
// of the NEW (already-deduplicated) graph being rebuilt: an Input's key
// is its own name; a Const's key is its literal value; an Add/Mul/
// ReLU's key is its op plus the NEW ids of its own (already-processed)
// operands. Two nodes with an identical key compute, by construction,
// the identical thing -- one of them is redundant.
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
        default: { // Add or Mul
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
    std::map<std::string, Value> keyToExistingValue;  // canonical key -> the node already built for it

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string key = cseKey(n, oldIdToNewValue);

        auto found = keyToExistingValue.find(key);
        if (found != keyToExistingValue.end()) {
            // Something already built in THIS walk computes exactly this
            // -- reuse it. oldId is remapped, but nothing new is added.
            oldIdToNewValue[oldId] = found->second;
            continue;
        }

        // First time this exact key has been seen -- actually build it.
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
    printf("=== Section 10.1: commonSubexpressionEliminationPass(), reusing an already-computed value ===\n\n");

    // t1 and t3 compute the SAME thing -- add(a, b), in the SAME operand
    // order -- built independently, as two different names. t4 consumes
    // t3; after CSE, it should consume t1's node instead, and t3 itself
    // should not exist in the result at all.
    Graph withDuplicateAdd;
    Value a  = withDuplicateAdd.addInput("a");
    Value b  = withDuplicateAdd.addInput("b");
    Value t1 = withDuplicateAdd.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = withDuplicateAdd.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = withDuplicateAdd.addBinary(OpKind::Add, a, b, "t3");  // duplicate of t1
    Value t4 = withDuplicateAdd.addBinary(OpKind::Mul, t3, b, "t4");
    withDuplicateAdd.addBinary(OpKind::Add, t2, t4, "out");

    printf("Graph with one duplicate computation ('t3' duplicates 't1'):\n");
    printGraph(withDuplicateAdd);

    Graph deduped = commonSubexpressionEliminationPass(withDuplicateAdd);
    printf("\nAfter commonSubexpressionEliminationPass():\n");
    printGraph(deduped);

    bool oneNodeDropped = (deduped.size() == withDuplicateAdd.size() - 1);
    bool t3Gone = true;
    for (const auto& n : deduped.nodes()) if (n->debugName == "t3") t3Gone = false;
    printf("\nself-check: exactly one duplicate node was dropped, %zu -> %zu (%s)\n",
           withDuplicateAdd.size(), deduped.size(), oneNodeDropped ? "confirmed" : "MISMATCH");
    printf("self-check: 't3' does not appear anywhere in the deduplicated graph (%s)\n",
           t3Gone ? "confirmed" : "MISMATCH");

    const Node* t1InDeduped = findNodeByName(deduped, "t1");
    const Node* t4InDeduped = findNodeByName(deduped, "t4");
    bool t4NowReferencesT1 = t4InDeduped->inputs[0].nodeId == t1InDeduped->id;
    printf("self-check: 't4' now reads its first operand from the SAME node as 't1' (%s)\n",
           t4NowReferencesT1 ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs1 = {{"a", 3.0f}, {"b", 5.0f}};
    float originalOut1 = evaluate(withDuplicateAdd, inputs1).at("out");
    float dedupedOut1 = evaluate(deduped, inputs1).at("out");
    bool sameAnswer1 = (originalOut1 == dedupedOut1);
    printf("self-check: evaluate(original, a=3,b=5).out = %g, evaluate(deduped, a=3,b=5).out = %g (%s)\n",
           originalOut1, dedupedOut1, sameAnswer1 ? "confirmed" : "MISMATCH");

    // A second, independent case: two LITERALLY duplicate Const nodes.
    // Merging them is a plain instance of the same key mechanism (no
    // folding involved -- both are already Const when the pass sees
    // them) -- and it CASCADES: once the two Const(7) nodes collapse
    // into one, the two Add nodes that each consumed one of them turn
    // out to have an identical key too, and collapse in the same walk.
    Graph withDuplicateConst;
    Value k1 = withDuplicateConst.addConst(7.0f, "k1");
    Value m  = withDuplicateConst.addInput("m");
    Value r1 = withDuplicateConst.addBinary(OpKind::Add, k1, m, "r1");
    Value k2 = withDuplicateConst.addConst(7.0f, "k2");  // literal duplicate of k1's value
    Value r2 = withDuplicateConst.addBinary(OpKind::Add, k2, m, "r2");
    withDuplicateConst.addBinary(OpKind::Add, r1, r2, "out");

    printf("\nGraph with two literally-duplicate Const nodes ('k1' and 'k2', both 7):\n");
    printGraph(withDuplicateConst);

    Graph constDeduped = commonSubexpressionEliminationPass(withDuplicateConst);
    printf("\nAfter commonSubexpressionEliminationPass():\n");
    printGraph(constDeduped);

    // 6 nodes -> 4: k2 merges into k1 (same key), which makes r2's own
    // key collapse onto r1's, all discovered in this ONE pass.
    bool constCascadeCorrect = (constDeduped.size() == 4);
    printf("\nself-check: the Const merge CASCADED into an Add merge in the same pass, %zu -> %zu (%s)\n",
           withDuplicateConst.size(), constDeduped.size(), constCascadeCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs2 = {{"m", 4.5f}};
    float originalOut2 = evaluate(withDuplicateConst, inputs2).at("out");
    float dedupedOut2 = evaluate(constDeduped, inputs2).at("out");
    bool sameAnswer2 = (originalOut2 == dedupedOut2);
    printf("self-check: evaluate(original, m=4.5).out = %g, evaluate(deduped, m=4.5).out = %g (%s)\n",
           originalOut2, dedupedOut2, sameAnswer2 ? "confirmed" : "MISMATCH");

    // Control case: a graph with NO duplicate computations at all should
    // come back completely unchanged in size -- CSE must not misfire.
    Graph clean;
    Value ca = clean.addInput("a");
    Value cb = clean.addInput("b");
    Value ct1 = clean.addBinary(OpKind::Add, ca, cb, "t1");
    Value ct2 = clean.addBinary(OpKind::Mul, ct1, ca, "t2");
    Value ct3 = clean.addUnary(OpKind::ReLU, ct1, "t3");
    clean.addBinary(OpKind::Add, ct2, ct3, "out");
    Graph stillClean = commonSubexpressionEliminationPass(clean);
    bool nothingMerged = (stillClean.size() == clean.size());
    printf("\nself-check: a graph with NO duplicate computations comes back the same size, %zu -> %zu\n",
           clean.size(), stillClean.size());
    printf("(CSE does not misfire on already-distinct input) (%s)\n",
           nothingMerged ? "confirmed" : "MISMATCH");

    bool allOk = oneNodeDropped && t3Gone && t4NowReferencesT1 && sameAnswer1 &&
                 constCascadeCorrect && sameAnswer2 && nothingMerged;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 018_common_subexpression_elimination_reuses_what_already_exists.cpp -o 018_common_subexpression_elimination_reuses_what_already_exists
./018_common_subexpression_elimination_reuses_what_already_exists
```

**Output:**

```text
=== Section 10.1: commonSubexpressionEliminationPass(), reusing an already-computed value ===

Graph with one duplicate computation ('t3' duplicates 't1'):
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1)
  %3 = Mul(t2)
  %4 = Add(t3)
  %5 = Mul(t4)
  %6 = Add(out)

After commonSubexpressionEliminationPass():
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1)
  %3 = Mul(t2)
  %4 = Mul(t4)
  %5 = Add(out)

self-check: exactly one duplicate node was dropped, 7 -> 6 (confirmed)
self-check: 't3' does not appear anywhere in the deduplicated graph (confirmed)
self-check: 't4' now reads its first operand from the SAME node as 't1' (confirmed)
self-check: evaluate(original, a=3,b=5).out = 64, evaluate(deduped, a=3,b=5).out = 64 (confirmed)

Graph with two literally-duplicate Const nodes ('k1' and 'k2', both 7):
  %0 = Const(k1) [value=7]
  %1 = Input(m)
  %2 = Add(r1)
  %3 = Const(k2) [value=7]
  %4 = Add(r2)
  %5 = Add(out)

After commonSubexpressionEliminationPass():
  %0 = Const(k1) [value=7]
  %1 = Input(m)
  %2 = Add(r1)
  %3 = Add(out)

self-check: the Const merge CASCADED into an Add merge in the same pass, 6 -> 4 (confirmed)
self-check: evaluate(original, m=4.5).out = 23, evaluate(deduped, m=4.5).out = 23 (confirmed)

self-check: a graph with NO duplicate computations comes back the same size, 6 -> 6
(CSE does not misfire on already-distinct input) (confirmed)
```

!!! note "Why 't3' disappears entirely, rather than 't1' being renamed"
    Look closely at the first output block: the deduplicated graph keeps a node named `t1`, and no node named `t3` exists anywhere in it -- not "t3, renamed" or "t3, pointing at t1's value," just gone. `commonSubexpressionEliminationPass()` always keeps whichever node it reaches FIRST in topological order (here, `t1`, since it was created before `t3`) and never re-adds whatever comes later with a matching key. This is a real, if minor, information loss: if a later debugging session cared specifically about "the node the source program called `t3`," that name is no longer discoverable anywhere in the optimized graph. Chapter 9's `deadCodeEliminationPass()` made the identical trade-off when it dropped `junk`'s name entirely rather than keeping some trace of it -- an optimizing pass, by its nature, does not promise to preserve everything about the ORIGINAL program, only that the program it produces computes the same answers.

## 10.2 What Counts as "the Same": A Structural Key's Real Limits

### Intuition

A strict but literal-minded assistant, told to "flag any two paragraphs that say the same thing," might correctly flag two paragraphs that are character-for-character identical, but miss two paragraphs that say the identical thing in a different word order -- "the cat sat on the mat" and "on the mat sat the cat" mean the same thing to a person, but they are not the same STRING. `cseKey()` is exactly that literal-minded assistant: it merges two nodes only when they are identical in a very specific, narrow sense -- same operation, same operands, in the same order -- and it is worth being precise about exactly where that narrowness shows up, because the gap between "always computes the same number" and "has an identical key" is a real, honestly-stated limitation of this chapter's pass, not an oversight to quietly patch over.

```text
TWO THINGS THAT ARE MATHEMATICALLY EQUAL, BUT NOT THE SAME cseKey():

    a   b                    b   a
     \ /                      \ /
   t1=add(a,b)              t2=add(b,a)

   cseKey(t1) = "add:a,b"      cseKey(t2) = "add:b,a"

   DIFFERENT STRINGS -- even though a+b and b+a compute the identical
   number for every real a and b. cseKey() encodes OPERAND ORDER; it
   has no notion that Add happens to be commutative.
```

### Background

Fixing this by sorting an `Add`/`Mul` node's operand ids before building its key -- so `add(a,b)` and `add(b,a)` would key identically -- sounds like a small change, but it only works when TWO separate things are both true: the operation actually is commutative for every possible pair of operands, and there is a stable, safe way to decide a canonical order. `Add` and `Mul` both genuinely qualify, for this book's own plain IEEE-754 floating point arithmetic. A real tensor compiler's op set, though, is full of operations that are emphatically NOT commutative -- subtraction, matrix multiplication, a `ReLU` applied after something else -- and teaching `cseKey()` exactly which operations are safe to reorder, and which are not, is real added logic this chapter leaves as an open extension rather than folding in silently. Getting it wrong in the unsafe direction -- merging two nodes that only LOOK the same -- would be a correctness bug, not merely a missed optimization; `commonSubexpressionEliminationPass()` stays on the conservative side of that line on purpose, at the cost of occasionally leaving a real, valid merge on the table.

```text
CONSERVATIVE (this chapter's own choice)     VS     UNSAFE IF DONE CARELESSLY

  never merges add(a,b) with add(b,a)               sorts operand ids for EVERY op,
  -- misses a real optimization                     including ones that are NOT
                                                      actually commutative -- merges
  worst outcome: a slightly bigger graph             two nodes that compute DIFFERENT
  than necessary. Still CORRECT.                     answers. WRONG output. A real bug.
```

The same narrow, literal key mechanism gets a different case right for a genuinely good reason, not by luck. Two separately-created `Input` nodes that happen to share the same debugName are not a coincidence to worry about -- under this book's own `evaluate()` (Chapter 9), an `Input` node is looked up purely by its name, so two `Input` nodes both named `"x"` really do represent the identical runtime value, for every execution. `cseKey()`'s plain name-based key for `Input` merges them correctly with no special-casing at all -- it happens to be exactly the right rule for the right reason, in contrast to the `Add`/`Mul` case, where the same "compare by what's already written down" rule is too conservative.

```cpp
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
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 019_what_counts_as_the_same_a_structural_keys_real_limits.cpp -o 019_what_counts_as_the_same_a_structural_keys_real_limits
./019_what_counts_as_the_same_a_structural_keys_real_limits
```

**Output:**

```text
=== Section 10.2: what commonSubexpressionEliminationPass() does and does NOT recognize as duplicate ===

Part 1 -- add(a,b) and add(b,a), operand order swapped:
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1)
  %3 = Add(t2)
  %4 = Add(out)

After commonSubexpressionEliminationPass():
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1)
  %3 = Add(t2)
  %4 = Add(out)

self-check: add(a,b) and add(b,a) are NOT merged -- size stays 5 -> 5 (confirmed)
self-check: the un-merged result is still numerically correct, 24 == 24 == 24 (confirmed)

Part 2 -- two separate Input nodes both named 'x':
  %0 = Input(x)
  %1 = Input(y)
  %2 = Add(r1)
  %3 = Input(x)
  %4 = Add(r2)
  %5 = Add(out)

After commonSubexpressionEliminationPass():
  %0 = Input(x)
  %1 = Input(y)
  %2 = Add(r1)
  %3 = Add(out)

self-check: the two same-named Input nodes merged, cascading into their consumers, 6 -> 4 (confirmed)
self-check: evaluate(original, x=2,y=10).out = 24, evaluate(deduped, ...).out = 24 (confirmed)
```

!!! warning "[COMMON TRAP] Assuming a pass that misses an optimization must be buggy"
    Part 1's own output can look wrong at first glance -- `add(a,b)` and `add(b,a)` are sitting right next to each other, obviously computing the same thing to a human reader, and the pass does nothing about it. That is not a bug; it is `commonSubexpressionEliminationPass()` correctly staying inside the boundary of what it actually checks. A pass that misses a real optimization it was never designed to find is behaving exactly as documented. A pass that finds a "duplicate" that later turns out not to be one -- because some operation it merged operands for wasn't actually commutative -- is a genuinely different, far worse failure: a silent correctness bug, one that would need a completely separate investigation to even notice, since nothing about the graph's SHAPE would look wrong. When judging whether an optimization pass is "good enough," the question is never "does it find every possible optimization?" -- it is "does everything it finds hold up, and is what it misses an honest, documented limitation rather than a surprise?"

## 10.3 Folding, CSE, and Dead Code Elimination Together: A Three-Chapter Pipeline

### Intuition

Section 9.3's own closing warning left a question open on purpose: could a more elaborate graph need constant folding and dead code elimination to alternate more than once each, each pass exposing new work for the other? It raised that question but did not resolve it -- and this section does not resolve it either. What it does instead is show the SAME underlying question appearing from a completely different angle: not "can removing dead code expose a new constant to fold," but "can folding a constant expose two nodes that are now duplicates of each other, which common subexpression elimination could never have recognized on its own, no matter when it ran, without folding having happened first."

```text
THE SAME "COULD RUNNING THIS AGAIN FIND MORE" QUESTION, TWO ANGLES:

  Section 9.3's angle:                    THIS section's angle:

    fold  -->  creates DEAD CODE            fold  -->  makes two DIFFERENT-LOOKING
    dce   -->  cleans up what fold                      nodes become IDENTICAL
              exposed                       cse   -->  merges what fold exposed,
                                                         then CASCADES into MORE
                                                         merges in the same pass

  both are instances of: one pass's own output can hand a LATER pass
  work that pass could not have found by itself, running first.
```

### Background

The graph this section builds has two constants, `2` and `3`, added together and named `p` -- which folds, under `constantFoldPass()`, to `Const(5)`. A THIRD, independently created constant, `c3`, is given the literal value `5` directly -- the exact value `p` will eventually compute, but before folding runs, `p` is an `Add` node and `c3` is a `Const` node. `cseKey()` gives an `Add` node and a `Const` node completely different key prefixes regardless of what the `Add` would eventually evaluate to -- `commonSubexpressionEliminationPass()` has no arithmetic of its own, and cannot know an unfolded `Add` node's eventual value without actually computing it, which is precisely `constantFoldPass()`'s job, not CSE's.

```text
BEFORE FOLDING:  p and c3 have DIFFERENT keys, EVEN THOUGH THEY WILL
                  TURN OUT TO BE EQUAL -- CSE cannot see this yet.

    c1  c2                    c3
     \  /                     |
    p=add(c1,c2)          (already Const(5))

    cseKey(p)  = "add:c1,c2"     cseKey(c3) = "const:5"    -- DIFFERENT

  AFTER FOLDING:  p is now ALSO Const(5) -- SAME key as c3.

    p=const(5)             c3=const(5)

    cseKey(p)  = "const:5"       cseKey(c3) = "const:5"    -- IDENTICAL
                                                             -- NOW mergeable
```

Every downstream test in this book has proven "order matters" by actually running the alternative and measuring the result, not by asserting it, and this section keeps that discipline: before running the full pipeline, it runs `commonSubexpressionEliminationPass()` completely ALONE on the ORIGINAL, unfolded graph, and confirms directly that it finds nothing at all to merge -- ruling out the possibility that CSE could have found this particular reduction on its own, at any point, without folding running first.

Once folding and CSE have both run, `c1` and `c2` -- the two constants that fed the now-vanished `p` -- are no longer referenced by anything at all: `p` itself was never kept, since CSE decided `c3` (or `p`, depending on which one the topological walk reaches first) was the survivor of that merge, and the other one simply disappears the same way `t3` disappeared in Section 10.1. That leaves `c1` and `c2` as genuinely NEW dead code -- unreachable from the graph's own output -- that only `deadCodeEliminationPass()`, run as the third and final pass, can clean up.

```cpp
// Chapter 10: Common Subexpression Elimination for Tensor Graphs
// 020_folding_cse_and_dce_together_a_three_chapter_pipeline.cpp
//
// Section 10.3 -- constantFoldPass() (9.1), commonSubexpressionEliminationPass()
// (10.1), and deadCodeEliminationPass() (9.2), run back to back through
// Chapter 8's own, completely unmodified runPasses(). Three
// TransformPasses, from three different chapters, none of them aware
// the other two exist, composed through infrastructure that has not
// changed one line since Chapter 8 first wrote it.
//
// This file's real point is a specific instance of a question Section
// 9.3's own [COMMON TRAP] raised and deliberately left open: "could
// running this again find more?" There, the question was about folding
// and DCE alternating. Here it shows up from a genuinely different
// angle -- CAN FOLDING EXPOSE A CSE OPPORTUNITY CSE COULD NOT HAVE
// FOUND ON ITS OWN? The graph below is built so the answer is yes, and
// proves it the same way Section 9.3 proved its own order-matters
// claim: not by assuring the reader in prose, but by actually running
// commonSubexpressionEliminationPass() ALONE on the ORIGINAL (unfolded)
// graph and showing, by a real measurement, that it finds nothing.
//
// The graph: two constants, 2 and 3, get added together and named "p".
// A THIRD, separately-created constant, "c3", is independently given
// the literal value 5 -- which happens to be exactly what "p" computes,
// but "p" is an Add node and "c3" is a Const node BEFORE folding runs,
// so cseKey() (Section 10.1) gives them completely different keys:
// commonSubexpressionEliminationPass() has no arithmetic of its own,
// and cannot know an Add node's eventual VALUE without evaluating it.
// Only after constantFoldPass() turns "p" into an actual Const(5) node
// does it share a key with "c3" -- and only then can CSE recognize them
// as the same thing.
//
// Once folding and CSE have both run, "c1" and "c2" (the two constants
// that fed "p") are no longer referenced by anything -- "p" itself was
// replaced by the CSE merge, not kept -- leaving them as genuinely NEW
// dead code that only deadCodeEliminationPass() can clean up. The full
// pipeline is: 8 nodes -> (fold) 8 nodes -> (CSE) 6 nodes -> (DCE) 4
// nodes, checked at every step, with evaluate() confirming the exact
// same answer before and after all three passes.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 020_folding_cse_and_dce_together_a_three_chapter_pipeline.cpp -o 020_folding_cse_and_dce_together_a_three_chapter_pipeline
// Run:     ./020_folding_cse_and_dce_together_a_three_chapter_pipeline
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>
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

// ==================== printGraphAsSource() (from Chapter 7, unchanged) ====================

static std::string opKindToSourceKeyword(OpKind op) {
    switch (op) {
        case OpKind::Input: return "input";
        case OpKind::Const: return "const";
        case OpKind::Add:   return "add";
        case OpKind::Mul:   return "mul";
        default:            return "relu";
    }
}

static std::string printGraphAsSource(const Graph& g) {
    std::string out;
    for (const auto& n : g.nodes()) {
        out += n->debugName + " = " + opKindToSourceKeyword(n->op) + "(";
        if (n->op == OpKind::Const) {
            char buf[64];
            snprintf(buf, sizeof(buf), "%g", n->constValue);
            out += buf;
        } else {
            for (size_t i = 0; i < n->inputs.size(); ++i) {
                if (i) out += ", ";
                out += g.node(n->inputs[i].nodeId)->debugName;
            }
        }
        out += ")\n";
    }
    return out;
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

// ==================== constantFoldPass() (from Section 9.1, unchanged) ====================

using TransformPass = std::function<Graph(const Graph&)>;

static Graph constantFoldPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("constantFoldPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        Value newValue;

        if (n->op == OpKind::Input) {
            newValue = result.addInput(n->debugName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, n->debugName);
        } else if (n->op == OpKind::ReLU) {
            Value in = oldIdToNewValue.at(n->inputs[0].nodeId);
            const Node* inNode = result.node(in.nodeId);
            if (inNode->op == OpKind::Const) {
                newValue = result.addConst(std::max(0.0f, inNode->constValue), n->debugName);
            } else {
                newValue = result.addUnary(OpKind::ReLU, in, n->debugName);
            }
        } else { // Add or Mul
            Value lhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value rhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            const Node* lhsNode = result.node(lhs.nodeId);
            const Node* rhsNode = result.node(rhs.nodeId);
            if (lhsNode->op == OpKind::Const && rhsNode->op == OpKind::Const) {
                float folded = (n->op == OpKind::Add) ? (lhsNode->constValue + rhsNode->constValue)
                                                        : (lhsNode->constValue * rhsNode->constValue);
                newValue = result.addConst(folded, n->debugName);
            } else {
                newValue = result.addBinary(n->op, lhs, rhs, n->debugName);
            }
        }

        oldIdToNewValue[oldId] = newValue;
    }

    return result;
}

// ==================== commonSubexpressionEliminationPass() (from Section 10.1, unchanged) ====================

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
        default: { // Add or Mul
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

// ==================== deadCodeEliminationPass() (from Section 9.2, unchanged) ====================

static std::set<int> computeLiveNodeIds(const Graph& g, int rootId) {
    std::set<int> live;
    std::deque<int> worklist{rootId};
    while (!worklist.empty()) {
        int id = worklist.front();
        worklist.pop_front();
        if (live.count(id)) continue;
        live.insert(id);
        for (const Value& in : g.node(id)->inputs) {
            worklist.push_back(in.nodeId);
        }
    }
    return live;
}

static Graph deadCodeEliminationPass(const Graph& g) {
    if (g.size() == 0) throw std::runtime_error("deadCodeEliminationPass: an empty graph has no output node");
    int rootId = static_cast<int>(g.size()) - 1;
    std::set<int> live = computeLiveNodeIds(g, rootId);

    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("deadCodeEliminationPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
        if (!live.count(oldId)) continue;

        const Node* n = g.node(oldId);
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
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ==================== PassManager (from Section 8.3, unchanged) ====================

struct NamedPass {
    std::string name;
    TransformPass fn;
};

static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass (%zu nodes) ---\n%s\n", initial.size(), printGraphAsSource(initial).c_str());

    Graph current = passes[0].fn(initial);
    {
        TopoResult topo = topologicalSort(current);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + passes[0].name +
                                      "' produced an invalid (non-acyclic) graph");
        }
    }
    printf("--- after pass '%s' (%zu nodes) ---\n%s\n", passes[0].name.c_str(), current.size(),
           printGraphAsSource(current).c_str());

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& p = passes[i];
        Graph next = p.fn(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + p.name +
                                      "' produced an invalid (non-acyclic) graph");
        }
        printf("--- after pass '%s' (%zu nodes) ---\n%s\n", p.name.c_str(), next.size(),
               printGraphAsSource(next).c_str());
        current = std::move(next);
    }
    return current;
}

int main() {
    printf("=== Section 10.3: folding creates a CSE opportunity; CSE creates new dead code ===\n\n");

    // c1(2) + c2(3) -> "p", which folds to Const(5) -- the exact same
    // VALUE as "c3", an independently-created Const(5). Before folding,
    // "p" is an Add node and "c3" is a Const node: different cseKey()s,
    // no merge possible. "r1" and "r2" each depend on one of them plus
    // a real runtime input "x", so neither folds away entirely --
    // keeping "p" and "c3" both alive as real operands for CSE to see.
    Graph original;
    Value c1 = original.addConst(2.0f, "c1");
    Value c2 = original.addConst(3.0f, "c2");
    Value p  = original.addBinary(OpKind::Add, c1, c2, "p");   // 2 + 3 -- folds to Const(5)
    Value c3 = original.addConst(5.0f, "c3");                  // literal duplicate of p's eventual value
    Value x  = original.addInput("x");
    Value r1 = original.addBinary(OpKind::Add, p, x, "r1");    // p + x
    Value r2 = original.addBinary(OpKind::Add, c3, x, "r2");   // c3 + x
    original.addBinary(OpKind::Add, r1, r2, "out");            // r1 + r2

    // First, the control measurement: run CSE ALONE on the UNFOLDED
    // original graph. "p" (an Add) and "c3" (a Const) have different
    // keys regardless of what they'd eventually compute -- CSE has no
    // arithmetic of its own, so it cannot see they are equal in VALUE.
    Graph cseAloneOnOriginal = commonSubexpressionEliminationPass(original);
    bool cseAloneFindsNothing = (cseAloneOnOriginal.size() == original.size());
    printf(">>> Control: commonSubexpressionEliminationPass() ALONE, before any folding\n\n");
    printf("size: %zu -> %zu\n", original.size(), cseAloneOnOriginal.size());
    printf("self-check: CSE alone finds NOTHING to merge in the unfolded graph (%s)\n\n",
           cseAloneFindsNothing ? "confirmed" : "MISMATCH");

    printf(">>> Pipeline: [constantFoldPass, commonSubexpressionEliminationPass, deadCodeEliminationPass]\n\n");
    Graph finalResult = runPasses(original, {
        {"constantFoldPass", constantFoldPass},
        {"commonSubexpressionEliminationPass", commonSubexpressionEliminationPass},
        {"deadCodeEliminationPass", deadCodeEliminationPass},
    });

    bool finalSizeCorrect = (finalResult.size() == 4);
    printf("self-check: the full pipeline reduced the graph from %zu nodes to %zu nodes (%s)\n",
           original.size(), finalResult.size(), finalSizeCorrect ? "confirmed" : "MISMATCH");

    std::map<std::string, float> inputs = {{"x", 6.5f}};
    float originalOut = evaluate(original, inputs).at("out");
    float finalOut = evaluate(finalResult, inputs).at("out");
    bool sameAnswer = (originalOut == finalOut);
    printf("self-check: evaluate(original, x=6.5).out = %g, evaluate(final, x=6.5).out = %g (%s)\n",
           originalOut, finalOut, sameAnswer ? "confirmed" : "MISMATCH");

    bool allOk = cseAloneFindsNothing && finalSizeCorrect && sameAnswer;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 020_folding_cse_and_dce_together_a_three_chapter_pipeline.cpp -o 020_folding_cse_and_dce_together_a_three_chapter_pipeline
./020_folding_cse_and_dce_together_a_three_chapter_pipeline
```

**Output:**

```text
=== Section 10.3: folding creates a CSE opportunity; CSE creates new dead code ===

>>> Control: commonSubexpressionEliminationPass() ALONE, before any folding

size: 8 -> 8
self-check: CSE alone finds NOTHING to merge in the unfolded graph (confirmed)

>>> Pipeline: [constantFoldPass, commonSubexpressionEliminationPass, deadCodeEliminationPass]

--- before any pass (8 nodes) ---
c1 = const(2)
c2 = const(3)
p = add(c1, c2)
c3 = const(5)
x = input()
r1 = add(p, x)
r2 = add(c3, x)
out = add(r1, r2)

--- after pass 'constantFoldPass' (8 nodes) ---
c1 = const(2)
c2 = const(3)
c3 = const(5)
x = input()
p = const(5)
r2 = add(c3, x)
r1 = add(p, x)
out = add(r1, r2)

--- after pass 'commonSubexpressionEliminationPass' (6 nodes) ---
c1 = const(2)
c2 = const(3)
c3 = const(5)
x = input()
r2 = add(c3, x)
out = add(r2, r2)

--- after pass 'deadCodeEliminationPass' (4 nodes) ---
c3 = const(5)
x = input()
r2 = add(c3, x)
out = add(r2, r2)

self-check: the full pipeline reduced the graph from 8 nodes to 4 nodes (confirmed)
self-check: evaluate(original, x=6.5).out = 23, evaluate(final, x=6.5).out = 23 (confirmed)
```

!!! warning "[COMMON TRAP] Assuming three passes, run once each, must be the end of the story"
    The pipeline above genuinely needed all three passes, in this order, to reach the 4-node result -- that much this section proved by actual measurement, not assertion. It would be a mistake to generalize from that to "three passes, run once each in this order, is always enough." Nothing here rules out a MORE elaborate graph where dead code elimination's own removals expose a fresh constant-folding opportunity (Section 9.3's own open question), which in turn creates a fresh CSE opportunity, which in turn creates more dead code -- each pass, once again, only looking at what's directly in front of it, with no pass re-examining its own earlier work in light of what a LATER pass just did. This chapter does not resolve that open question any more than Chapter 9 did; it simply shows the identical shape of problem recurring with a third pass added to the mix, which is exactly the sort of accumulating evidence that eventually justifies a REAL pass manager running its sequence to a fixed point, rather than the fixed number of once-through passes `runPasses()` still runs today.

## Chapter Summary

This chapter added Part 2's third optimizing pass, `commonSubexpressionEliminationPass()`, plugging into Chapter 8's `TransformPass`/`PassManager` infrastructure with no changes to that infrastructure at all -- the third chapter in a row to do so. Section 10.1 built the pass itself: a canonical key describing exactly what each node computes, checked against a table of everything already built earlier in the same topological walk, letting duplicate `Add`/`Mul`/`ReLU` computations collapse into one and, because the key is built from the graph being rebuilt rather than the original, letting a merge discovered early in the walk cascade into further merges discovered later in the very same pass. Section 10.2 characterized the pass's real boundaries with actual executed evidence on both sides: `cseKey()` is too conservative to merge `add(a,b)` with `add(b,a)`, a real, deliberate, honestly-stated limitation kept to avoid the much worse alternative of merging operands for operations that are not actually commutative -- and, in contrast, correctly merges two separately-created `Input` nodes sharing a name, since this book's own `evaluate()` treats them as the identical runtime value already. Section 10.3 composed `constantFoldPass()`, `commonSubexpressionEliminationPass()`, and `deadCodeEliminationPass()` through Chapter 8's unmodified `runPasses()`, proving by direct measurement -- running CSE alone on the unfolded graph and confirming it finds nothing -- that constant folding can expose a structural-equality opportunity CSE could never have found on its own, and that the CSE merge which follows creates yet more dead code for the final pass to clean up, shrinking a real eight-node graph down to four while `evaluate()` confirms the answer never changed.

## Self-Check Questions

1. `cseKey()` is built using `oldIdToNewValue`, mapping OLD ids to the NEW graph's own ids, rather than reading an `Add`/`Mul` node's operand ids directly from the old graph. Why does that choice matter, and what specific behavior (demonstrated in Section 10.1's own Const test) would be lost if `cseKey()` read the old graph's ids instead?
2. In Section 10.1's duplicate-Const test, `k2` was dropped because it duplicated `k1`. But `r2` was ALSO dropped, even though `r2` itself was never a literal duplicate of anything at the moment it was created. Why did `r2` disappear too?
3. Why does `commonSubexpressionEliminationPass()` leave `add(a,b)` and `add(b,a)` unmerged, even though they always compute the same number? What would fixing this require, and why is that harder than simply sorting the two operand ids?
4. Section 10.2 also showed two separately-created `Input` nodes sharing a name getting merged correctly, with no special-casing. Why is that merge safe, in a way the `add(a,b)`/`add(b,a)` case is not?
5. In Section 10.3, `commonSubexpressionEliminationPass()` run ALONE on the unfolded original graph finds nothing to merge, even though `p` and `c3` both represent the value 5. Why can't CSE see that on its own?
6. After both `constantFoldPass()` and `commonSubexpressionEliminationPass()` have run in Section 10.3's pipeline, `c1` and `c2` become dead code. Were they already dead before CSE ran? Explain what specifically changed.
7. Section 9.3's own [COMMON TRAP] left open the question of whether running each pass once, in sequence, is always enough. Does Section 10.3's three-pass pipeline resolve that question, or does it restate it in a new form? Explain.
8. Sketch, in words (no code needed), a graph where running `deadCodeEliminationPass()` would expose a NEW common-subexpression opportunity that `commonSubexpressionEliminationPass()` could not have found before DCE ran.

## Where We Go Next

Part 2 now has three independently useful, independently tested optimizations -- constant folding, dead code elimination, and common subexpression elimination -- composed through infrastructure that has needed zero changes since Chapter 8 first built it, across three separate chapters. Chapter 11, "Algebraic Simplification," continues in the same pattern with a fourth `TransformPass`: rewriting a computation into an equivalent but cheaper one using algebraic identities (`x + 0` is always just `x`; `x * 1` is always just `x`), rather than merely reusing or removing what is already there.

## Worked Solutions

1. Building `cseKey()` from `oldIdToNewValue` means a merge decided EARLIER in the walk (say, `k2` collapsing into `k1`) is immediately reflected in the key of anything LATER in the walk that consumed the dropped node (`r2`, which consumed `k2`) -- `r2`'s key is computed using `k1`'s new id, not `k2`'s old one, so it correctly comes out identical to `r1`'s key. Reading the OLD graph's ids instead would mean `r1` and `r2` still looked like they read two DIFFERENT operands (the old `k1` and the old `k2`), even after those two operands had already been recognized as the same thing -- the cascade demonstrated in Section 10.1's own Const test would not happen in a single pass; it would need the pass to run a second time, checking the old graph's state after the first run, to catch what checking the new graph catches immediately.
2. Once `k2` is recognized as a duplicate of `k1` and remapped in `oldIdToNewValue`, `r2 = add(k2, m)`'s own `cseKey()` is computed using `k2`'s NEW mapped id -- which is `k1`'s id, not `k2`'s own old one. That makes `r2`'s key `"add:k1id,mid"`, the exact same string as `r1`'s key (`r1 = add(k1, m)`, using `k1`'s own id directly). `r2` was never a duplicate of anything AT THE MOMENT it was created -- it only became one once the operand it read was itself recognized as redundant, one step earlier in the very same walk.
3. `cseKey()` encodes operand order as part of the key string (`"add:a,b"` versus `"add:b,a"`), so two operands in a different order never produce an identical key, regardless of whether the underlying operation happens to be commutative. Fixing this by sorting operand ids before building the key would work for `Add` and `Mul`, which genuinely are commutative for this book's own floating-point arithmetic -- but a general tensor compiler has operations that are NOT commutative (subtraction, matrix multiplication, a `ReLU` chained after something), and `cseKey()` would need to know, for every operation it might ever see, whether reordering its operands is actually safe. Getting that wrong in the unsafe direction would merge two nodes that do not actually compute the same thing -- a real correctness bug, not merely a missed optimization -- which is exactly the risk staying conservative avoids.
4. Under this book's own `evaluate()` (Chapter 9), an `Input` node's value is looked up purely by its `debugName` -- `inputValuesByName.at(n->debugName)` -- with no reference at all to WHICH `Input` node is asking. Two separate `Input` nodes sharing the name `"x"` therefore genuinely represent the identical runtime value for every possible execution; merging them changes nothing about what the graph computes. `add(a,b)` and `add(b,a)`, by contrast, are only equal because addition happens to be commutative -- a property of the SPECIFIC operation involved, not a guarantee that holds for operations in general. The `Input` merge is safe because of what `evaluate()` itself guarantees; the `Add`/`Mul` non-merge is conservative because that same guarantee does not extend to every operation.
5. Before folding, `p` is an `Add` node (its `cseKey()` starts with `"add:"`, built from `c1`'s and `c2`'s own ids) and `c3` is a `Const` node (its `cseKey()` starts with `"const:"`, built from its own literal value) -- two completely different key PREFIXES, regardless of what `p` would eventually evaluate to if computed. `commonSubexpressionEliminationPass()` only ever compares the keys nodes ALREADY have, as they are already written in the graph; it has no arithmetic of its own and never evaluates an `Add` node to discover its value the way `constantFoldPass()` does. Only after `constantFoldPass()` has actually turned `p` into a real `Const(5)` node does `p`'s key become `"const:5"` -- the same as `c3`'s -- making the two nodes visibly identical to CSE for the first time.
6. `c1` and `c2` were NOT dead before CSE ran -- `constantFoldPass()` alone leaves the graph at a full 8 nodes, with `p` (now `Const(5)`) still present as its own distinct node, still nominally "produced from" `c1` and `c2` in the sense that `p` exists at all (even though its own new node no longer literally references them as inputs, since a folded `Const` node has no inputs at all). What actually made `c1`/`c2` unreachable from the graph's output was `commonSubexpressionEliminationPass()` deciding not to keep `p` as a distinct node once it recognized `p` and `c3` compute the same value -- once `p` itself was elided in favor of `c3`, nothing in the graph referenced `c1` or `c2` in any way at all, which is precisely what `deadCodeEliminationPass()`, running last, is built to find and remove.
7. It restates the question in a new form; it does not resolve it. Section 10.3's own pipeline genuinely needed three passes, run once each, in a specific order -- but that is a fact about THIS graph, proven by measurement, not a guarantee about every possible graph. Nothing in this chapter rules out an even more elaborate graph where `deadCodeEliminationPass()`'s own removals expose a fresh constant-folding opportunity, or a fresh CSE opportunity, that would need one of the earlier passes to run again to catch. The [COMMON TRAP] closing this section says so directly: this chapter adds a third real instance of the same open problem Chapter 9 first raised, rather than closing it.
8. One concrete shape: two nodes, `r1` and `r2`, that are each ALMOST identical computations except that one of them also depends on some third node, `junk`, that is otherwise completely unused elsewhere in the graph (so `junk` is dead from the very start) -- say `r1 = add(t, junk)` and `r2 = add(t, other)`, where `junk` and `other` are NOT currently equal, so `r1` and `r2` do not share a `cseKey()` and CSE finds no merge. If `deadCodeEliminationPass()` runs FIRST and removes `junk` entirely, and if removing `junk` also lets `other`'s OWN definition simplify down to something structurally identical to what `t` already is elsewhere in the graph (for instance, if `other` was only complicated because it was built to compensate for `junk`'s presence), THEN `r1` and `r2` could become genuine duplicates only after DCE has already run -- a merge CSE, run first, could never have found, mirroring the exact "folding creates a CSE opportunity" shape this section built, with DCE playing folding's role instead.

---

**Sources cited in this chapter:**

None. `commonSubexpressionEliminationPass()` and its combination with Chapter 9's own passes through Chapter 8's `runPasses()` are original designs for this book's own `Graph`, built entirely from vocabulary and machinery Chapters 4, 7, 8, and 9 already established. The general idea of common subexpression elimination is standard, decades-old compiler-construction vocabulary (not specific to any one real compiler this book would need to cite), consistent with how Chapter 8 treated the general analysis-pass/transform-pass distinction.
