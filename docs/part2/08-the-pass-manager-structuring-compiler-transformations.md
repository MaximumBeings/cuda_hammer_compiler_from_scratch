# 8. The Pass Manager: Structuring Compiler Transformations

**What you will understand:** the difference between an *analysis* pass (reads a `Graph`, produces some other result, never touches the graph) and a *transform* pass (reads a `Graph`, produces a *different* `Graph`); a real, working first transform pass; and a `PassManager` that runs a sequence of passes over a graph, printing a before/after dump of each one and checking the result is still a valid DAG after every single pass. This opens Part 2.

**What you need to know first:** Chapter 4's `Value`/`Node`/`Graph` and `topologicalSort()`, Chapter 5's `graphsStructurallyEqual()`, and Chapter 7's `printGraphAsSource()` -- this chapter reuses all three completely unchanged.

---

Chapter 6 built this book's first analysis pass, `inferShapes()`: it reads a `Graph`, walks it, and returns a brand-new `map<int, Shape>` -- and never once modifies the `Graph` it read. Chapter 7 built something with a similar shape, `printGraphAsSource()`: reads a `Graph`, returns a `std::string`, and again never touches the graph itself. Both chapters called their own function a "pass," borrowing Chapter 2's own general vocabulary, but neither chapter needed to be precise about what other *kind* of pass might exist, because neither one needed to change a graph. Part 2 does. Chapter 9's constant folding needs to replace a computed node with a simpler constant one; its dead code elimination needs to drop nodes nothing downstream uses at all. Neither of those is possible with only the kind of pass Chapters 6 and 7 already wrote -- this chapter builds the other kind, and the machinery to run a sequence of them safely.

```text
TWO KINDS OF PASS, BOTH ALREADY USED IN THIS BOOK, NOW MADE PRECISE:

  ANALYSIS pass (Chapters 6, 7):        TRANSFORM pass (this chapter, and
                                          every chapter in Part 2 after it):

    Graph  -->  inferShapes()             Graph  -->  (some transform)  -->  a
           -->  a map from node id                                           DIFFERENT
                to Shape                                                     Graph

    Graph  -->  printGraphAsSource()      the input Graph is READ, never
           -->  std::string                mutated -- a NEW Graph is built
                                            and returned instead.
    the input Graph is READ, never
    mutated, and the RESULT is not
    a Graph at all.
```

## 8.1 What a Pass Actually Is: Analysis Versus Transform

### Intuition

A proofreader and a copy editor both read the exact same manuscript, but they hand back fundamentally different things: a proofreader hands back a list of problems, leaving the manuscript itself untouched, while a copy editor hands back a *revised manuscript*, a genuinely different document built from the original. Both are doing real, valuable work over the same input; neither one's output is "wrong" for not looking like the other's. An analysis pass is this book's own proofreader -- `inferShapes()` reads a graph and reports something about it (a shape for every node) without changing a single thing about the graph itself. A transform pass is the copy editor -- it reads a graph and hands back a genuinely different graph, one this book has never seen before that call, built specifically because the original wasn't quite what was wanted.

### Background

The concrete decision this section has to make is not just "transform passes exist" -- it's *how* a transform pass is allowed to change a graph, and that decision is forced by something Chapter 4 already built (or rather, didn't build): `Graph` has never had a way to remove a node once `addInput`/`addConst`/`addUnary`/`addBinary` added it. That was a perfectly reasonable omission for Chapters 4 through 7 -- nothing in representing, parsing, validating, or printing a graph ever needed to delete anything -- but Chapter 9's dead code elimination is going to need to drop nodes nothing downstream depends on, and this book is not going to retrofit an in-place removal API onto `Graph` just to support one future chapter. Instead, this chapter fixes a `TransformPass`'s own signature once, for every pass Part 2 will ever write:

```text
WHY A TransformPass READS AN OLD Graph AND RETURNS A NEW ONE, RATHER
THAN MODIFYING ITS INPUT IN PLACE:

  Graph (from Chapter 4) can ADD nodes -- addInput/addConst/addUnary/
  addBinary -- but has NO way to REMOVE one once added.

  Chapter 9's own dead code elimination will need to drop nodes -- so
  "modify the graph you were given, in place" is not an option for
  every pass this book will ever want to write.

  the fix: a TransformPass is a function taking a "const Graph&" and
           returning a "Graph" (in code: using TransformPass =
           std::function of Graph from const Graph reference)

    every pass reads an EXISTING graph (never mutates it) and BUILDS
    AND RETURNS a brand new one, keeping only what it wants to keep.
    dropping a node this way is simply never adding it to the new
    graph in the first place -- no removal API is ever needed.
```

This is a real, deliberate architecture choice some real compiler IRs make too -- rebuilding rather than mutating in place trades a bit of extra copying for never needing a removal API at all -- and this is a general point about compiler architecture, not a specific claim needing its own citation, the same way Chapter 5's comparison to LLVM's own flat IR text format didn't need one. The concrete payoff shows up immediately: because a `TransformPass` never mutates what it was given, the caller's own "before" copy of a graph is always still completely intact after the pass runs -- which is exactly what lets Section 8.3's `PassManager` print a trustworthy "before" dump, run the pass, and print an "after" dump, knowing the "before" dump wasn't secretly describing something that got mutated out from under it half a function call ago.

!!! warning "[COMMON TRAP] Assuming a transform pass could just add a `remove()` method to `Graph` instead"
    It's tempting to think the more "obvious" fix is giving `Graph` a real node-removal method and letting a pass like dead code elimination just call it directly. That path is not free: removing a node the normal way would require finding and fixing up every OTHER node's `inputs` list that might reference it (or deciding removal is illegal while any live reference exists, which is its own bookkeeping problem), plus deciding what happens to node ids after a removal -- do they shift down, leaving gaps, or stay stable with holes in `nodes_`? The "rebuild a new graph, keep only what belongs" approach this chapter picks sidesteps every one of those questions: a node that shouldn't exist in the result is just never re-added, and the new graph's ids are simply reassigned in whatever order the pass walks the old one -- Chapter 9 will lean on exactly this simplicity.

## 8.2 A First Real Transform Pass: Canonicalizing Node Names

### Intuition

Two photographs of the exact same street corner, taken by two different photographers, can look completely different -- different angle, different time of day, different framing -- while depicting the literal same physical place. Comparing them side by side to answer "is this the same corner" is hard precisely because the *incidental* differences (lighting, framing) swamp the *structural* fact that matters (same buildings, same intersection). Two graphs built through this book's own API can end up in exactly this situation: the diamond graph built by hand in Chapter 4, parsed from text in Chapter 5, or handed back by some future pass, all compute the exact same thing, but a human (or a future pass, printing before/after dumps) comparing their *printed text* would see totally different node names and have to work to notice the underlying computation never changed at all.

```text
THE SAME COMPUTATION, TWO COMPLETELY DIFFERENT PRINTED TEXTS -- A REAL
PROBLEM FOR ANY FUTURE PASS THAT WANTS TO COMPARE "BEFORE" AND "AFTER"
DUMPS BY EYE:

  built by hand (Chapter 4's own            the exact same computation,
  human-chosen names):                       with different names chosen
                                              by someone else entirely:
    a = input()
    b = input()                                x = input()
    t1 = add(a, b)                             y = input()
    t2 = mul(t1, a)                            sum1 = add(x, y)
    t3 = relu(t1)                              prod1 = mul(sum1, x)
    out = add(t2, t3)                          relu1 = relu(sum1)
                                                result = add(prod1, relu1)

  graphsStructurallyEqual() (Chapter 5) already knows these two ARE the
  same graph -- it never even looks at debugName. But a human reading
  the printed TEXT would need to check that fact by hand, every time.
```

### Background

`canonicalizeNodeNames()` fixes exactly this problem by rewriting every node's `debugName` to a name determined entirely by its own position in the graph -- `"%" + id` -- so two structurally identical graphs, however they were originally named, print out identically after canonicalization. The function walks the input graph in Chapter 4's own `topologicalSort()` order (deliberately, not `g.nodes()` directly -- the same "don't rely on the coincidence" discipline Chapter 6's `inferShapes()` and Chapter 7's own printer both already established) and rebuilds a fresh `Graph`, one node at a time, through the ordinary `addInput`/`addConst`/`addUnary`/`addBinary` API -- the same API every other chapter has used, never anything more privileged.

```text
CANONICALIZING THE DIAMOND GRAPH, STEP BY STEP:

  old id  op     old name  new id  canonical name
  %0      Input  a         %0      %0
  %1      Input  b         %1      %1
  %2      Add    t1        %2      %2
  %3      Mul    t2        %3      %3
  %4      ReLU   t3        %4      %4
  %5      Add    out       %5      %5

  every canonical name is exactly "%" + that node's OWN new id -- the
  new graph's structure (which node points at which) is completely
  unchanged; only every debugName is replaced.
```

The one piece of bookkeeping this function cannot skip is remapping *references*: when node `%3` (old `t2`, `Mul(t1, a)`) gets rebuilt, its two inputs have to point at the NEW graph's own values for `t1` and `a` -- not their old ids, which is why `canonicalizeNodeNames()` keeps an explicit `map<int, Value> oldIdToNewValue` as it walks, updated after every node it rebuilds, and consulted every time a later node needs to reference an earlier one.

```text
WHY canonicalizeNodeNames() KEEPS AN EXPLICIT oldIdToNewValue MAP,
RATHER THAN ASSUMING NEW IDS EQUAL OLD IDS:

  for THIS book's own diamond graph, walking in topological order
  happens to assign new id 0 to old id 0, new id 1 to old id 1, and so
  on -- the new and old ids happen to coincide, exactly the same
  coincidence Chapter 6's own [COMMON TRAP] and Chapter 7's own note
  both already called out for THEIR own functions.

  canonicalizeNodeNames() does not rely on that coincidence either: it
  looks up every input through oldIdToNewValue.at(oldInputId), which
  gives the CORRECT new Value regardless of whether new and old ids
  happen to line up -- so this function keeps working correctly even
  for some future graph where they don't (a pass that only rebuilds
  SOME of a graph's nodes, skipping others, is exactly the kind of
  future case where they wouldn't).
```

File 013 below builds the diamond graph with Chapter 4's own human-chosen names, runs `canonicalizeNodeNames()`, and checks two genuinely different things: that every resulting name is exactly `"%" + id` (the surface-level claim), and, far more importantly, that the canonicalized graph is still `graphsStructurallyEqual()` to the original (the claim that actually matters -- the computation itself never changed, only its names did). A third check confirms the original graph's own names are completely untouched after the pass runs, proving `canonicalizeNodeNames()` really is read-only on its input, exactly as Section 8.1 requires of every `TransformPass`.

```cpp
// Chapter 8: The Pass Manager: Structuring Compiler Transformations
// 013_pass_interface_and_a_first_transform_pass.cpp
//
// Section 8.1 -- what a PASS actually is, made concrete for THIS book's
// own Graph -- and Section 8.2 -- the very first real transform pass,
// one deliberately simple enough not to overlap with Chapter 9's
// constant folding, Chapter 10's common subexpression elimination, or
// Chapter 11's algebraic simplification, while still being a genuine,
// useful rewrite rather than a placeholder.
//
// Chapter 2 already gave this book the word "pass": something that
// reads an IR and produces a result, without touching source text
// again. Chapters 6 and 7 already wrote two: inferShapes() (an ANALYSIS
// pass -- reads a Graph, produces a side table, never touches the
// Graph itself) and printGraphAsSource() (arguably a pass too, though
// its result is text rather than a data structure). This file adds the
// other kind: a TRANSFORM pass, one that reads a Graph and produces a
// DIFFERENT Graph.
//
// One design decision worth stating up front, because it shapes
// everything from here through Chapter 11: this book's own Graph (from
// Chapter 4) has never had a way to REMOVE a node once added -- addInput/
// addConst/addUnary/addBinary are strictly append-only. Chapter 9's dead
// code elimination is going to need to drop nodes; a real in-place
// removal API is more machinery than this book needs to add just to
// support that. Instead, a TransformPass in this book has the signature
// `Graph(const Graph&)` -- it reads an old graph and BUILDS AND RETURNS
// a brand new one, keeping only what it wants to keep. This "rebuild
// rather than mutate" style is a real, legitimate design some real
// compiler IRs use too (this is a general architectural fact, not a
// specific claim needing its own citation) -- it trades a bit of extra
// copying for never needing a removal API at all.
//
// This file's own first real transform pass, canonicalizeNodeNames(),
// renumbers every node's debugName to "%<id>" in topological order.
// That might look cosmetic, but it solves a real problem: two graphs
// that compute the exact same thing, built through different paths
// (by hand, parsed from text, or already passed through some earlier
// pass), can have completely different human-chosen names -- which
// means Chapter 7's own printGraphAsSource() would print them
// differently even though they are structurally identical. A canonical
// naming pass makes "print two graphs and diff the text" a meaningful
// way to compare them, which Chapter 9 onward will lean on constantly
// when showing a pass's own before/after effect.
//
// Reuses Chapter 4's Value/Node/Graph and topologicalSort(), and
// Chapter 5's graphsStructurallyEqual() (which compares op kind and
// input ids -- NEVER debugName -- making it exactly the right tool to
// confirm this pass changes names and nothing else).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 013_pass_interface_and_a_first_transform_pass.cpp -o 013_pass_interface_and_a_first_transform_pass
// Run:     ./013_pass_interface_and_a_first_transform_pass
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
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

// ==================== Structural equality (from Chapter 5, unchanged) ====================
//
// Deliberately compares op kind and input ids only -- NEVER debugName.
// That is exactly what makes it the right tool to check a renaming
// pass: two graphs this function calls equal may still have completely
// different names on every single node.
static bool graphsStructurallyEqual(const Graph& a, const Graph& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const Node* na = a.node(static_cast<int>(i));
        const Node* nb = b.node(static_cast<int>(i));
        if (na->op != nb->op) return false;
        if (na->inputs.size() != nb->inputs.size()) return false;
        for (size_t j = 0; j < na->inputs.size(); ++j) {
            if (na->inputs[j].nodeId != nb->inputs[j].nodeId) return false;
        }
    }
    return true;
}

// ============================== Section 8.1: the Pass interface (new) ==============================
//
// A TransformPass reads an existing Graph and returns a BRAND NEW one.
// It never mutates the Graph it was given -- the caller's own original
// graph is always still intact after a pass runs, which is exactly what
// lets Chapter 8's own PassManager (File 014) print a "before" dump,
// run the pass, and print an "after" dump without the "before" dump
// having become a lie partway through.
using TransformPass = std::function<Graph(const Graph&)>;

// ============================== Section 8.2: canonicalizeNodeNames() (new) ==============================
//
// Walks the input graph in topological order (reusing Chapter 4's own
// topologicalSort() explicitly, rather than assuming id order already
// matches it -- the same discipline Chapter 6's inferShapes() and
// Chapter 7's own [COMMON TRAP]-adjacent note both already established:
// don't rely on a coincidence a future pass might not preserve) and
// rebuilds every node into a fresh Graph with a canonical name, "%<id>"
// matching that node's OWN id in the new graph. A map from the OLD
// graph's node ids to the Values the NEW graph produced is threaded
// through so every input reference gets correctly remapped, even if the
// new graph's ids ever ended up different from the old ones.
static Graph canonicalizeNodeNames(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("canonicalizeNodeNames: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;

    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string canonicalName = "%" + std::to_string(result.size());
        Value newValue;

        if (n->op == OpKind::Input) {
            newValue = result.addInput(canonicalName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, canonicalName);
        } else if (n->inputs.size() == 1) {
            Value remappedIn = oldIdToNewValue.at(n->inputs[0].nodeId);
            newValue = result.addUnary(n->op, remappedIn, canonicalName);
        } else {
            Value remappedLhs = oldIdToNewValue.at(n->inputs[0].nodeId);
            Value remappedRhs = oldIdToNewValue.at(n->inputs[1].nodeId);
            newValue = result.addBinary(n->op, remappedLhs, remappedRhs, canonicalName);
        }

        oldIdToNewValue[oldId] = newValue;
    }

    return result;
}

int main() {
    printf("=== Section 8.2: canonicalizeNodeNames(), a real first transform pass ===\n\n");

    // The familiar diamond graph, with human-chosen names -- exactly
    // what Chapters 4 through 7 have all used.
    Graph original;
    Value a  = original.addInput("a");
    Value b  = original.addInput("b");
    Value t1 = original.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = original.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = original.addUnary(OpKind::ReLU, t1, "t3");
    original.addBinary(OpKind::Add, t2, t3, "out");

    printf("Original graph (human-chosen names):\n");
    for (const auto& n : original.nodes()) {
        printf("  %%%d = %s(%s)\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
    }

    Graph canonical = canonicalizeNodeNames(original);

    printf("\nAfter canonicalizeNodeNames() (canonical names):\n");
    for (const auto& n : canonical.nodes()) {
        printf("  %%%d = %s(%s)\n", n->id, opKindStr(n->op).c_str(), n->debugName.c_str());
    }

    // Every node's debugName should now be exactly "%" + its own id.
    bool everyNameCanonical = true;
    for (const auto& n : canonical.nodes()) {
        if (n->debugName != "%" + std::to_string(n->id)) everyNameCanonical = false;
    }
    printf("\nself-check: every node's debugName is exactly \"%%<its own id>\" (%s)\n",
           everyNameCanonical ? "confirmed" : "MISMATCH");

    // The real claim: canonicalization changed NAMES ONLY. Structural
    // equality (which never looks at debugName at all) must still hold.
    bool stillStructurallyEqual = graphsStructurallyEqual(original, canonical);
    printf("self-check: the canonicalized graph is still graphsStructurallyEqual() to the\n");
    printf("original -- only names changed, the computation itself did not (%s)\n",
           stillStructurallyEqual ? "confirmed" : "MISMATCH");

    // The original graph must be completely untouched -- a TransformPass
    // reads, it never mutates.
    bool originalNamesUntouched = (original.node(t1.nodeId)->debugName == "t1") &&
                                   (original.node(t2.nodeId)->debugName == "t2") &&
                                   (original.node(t3.nodeId)->debugName == "t3");
    printf("self-check: the ORIGINAL graph's own names are completely untouched --\n");
    printf("canonicalizeNodeNames() read from it but never mutated it (%s)\n",
           originalNamesUntouched ? "confirmed" : "MISMATCH");

    bool allOk = everyNameCanonical && stillStructurallyEqual && originalNamesUntouched;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 013_pass_interface_and_a_first_transform_pass.cpp -o 013_pass_interface_and_a_first_transform_pass
./013_pass_interface_and_a_first_transform_pass
```

**Output:**

```text
=== Section 8.2: canonicalizeNodeNames(), a real first transform pass ===

Original graph (human-chosen names):
  %0 = Input(a)
  %1 = Input(b)
  %2 = Add(t1)
  %3 = Mul(t2)
  %4 = ReLU(t3)
  %5 = Add(out)

After canonicalizeNodeNames() (canonical names):
  %0 = Input(%0)
  %1 = Input(%1)
  %2 = Add(%2)
  %3 = Mul(%3)
  %4 = ReLU(%4)
  %5 = Add(%5)

self-check: every node's debugName is exactly "%<its own id>" (confirmed)
self-check: the canonicalized graph is still graphsStructurallyEqual() to the
original -- only names changed, the computation itself did not (confirmed)
self-check: the ORIGINAL graph's own names are completely untouched --
canonicalizeNodeNames() read from it but never mutated it (confirmed)
```

!!! note "Why the printed output above still says \"Add(%2)\" rather than \"Add(%0, %1)\""
    This particular `main()` reuses the same quick debug-style print loop Chapters 4 through 6 each wrote their own version of -- it prints a node's *first* input inline next to its own name, for a fast human-readable glance, not a full reparseable dump. Section 8.3 switches over to Chapter 7's own `printGraphAsSource()` for its own before/after dumps specifically because THAT is the one built and proven to be a genuine, trustworthy serialization -- this distinction is exactly Section 8.1's own point about not confusing a debug dump with a real serialization, showing up again here in a chapter that is itself about passes.

## 8.3 The Pass Manager: Chaining Passes and Validating Between Them

### Intuition

A relay team's coach does not just fire the starting gun and hope -- between every single handoff, someone is watching to confirm the baton was actually passed cleanly before the next runner is allowed to take off. If a handoff goes wrong, everyone finds out at THAT exact handoff, not three legs later when the team crosses the finish line in the wrong order and nobody can say where it went wrong. A `PassManager` plays exactly this role for a sequence of transform passes: Part 2 and beyond will eventually want to run several passes back to back (constant fold, then eliminate dead code, then simplify, then eliminate dead code again), and trusting every single one blindly -- only checking the FINAL result, if at all -- means a bug in an early pass could silently corrupt everything after it, with no way to tell which pass actually caused the problem.

```text
WHAT A PassManager THAT ONLY CHECKED THE FINAL RESULT WOULD RISK:

  Graph --> pass A --> pass B (has a bug!) --> pass C --> FINAL CHECK

  if the final check fails, WHICH pass is at fault? A, B, or C? The
  bug happened at B, but by the time anyone looks, C has already run
  on B's already-broken output -- the evidence of WHERE things went
  wrong is buried under whatever C did to the mess it was handed.

  THIS chapter's own PassManager checks after EVERY pass instead:

  Graph --> pass A --> CHECK --> pass B (has a bug!) --> CHECK (FAILS,
            HERE, immediately -- naming pass B specifically, before
            pass C ever gets a chance to run on broken input at all)
```

### Background

`runPasses()` takes an initial graph and an ordered list of `NamedPass`es (a name paired with a `TransformPass`), and threads the graph through them one at a time. After every single pass -- not just at the end -- it does two things: it calls Chapter 4's own `topologicalSort()` on the result and throws immediately, naming that exact pass, if the result is not a valid DAG; and it calls Chapter 7's own `printGraphAsSource()` to print the graph's new state, so a human watching the output can see exactly what each pass did. This is the first place in the book Chapter 7's printer is used for something other than testing itself -- and the fact that Chapter 7 proved it a genuine round-trip-verified serialization, not just a plausible-looking dump, is exactly what makes trusting these before/after printouts reasonable here.

```text
THE runPasses() FLOW, FOR A TWO-PASS PIPELINE:

  step 1: print the graph's STARTING state (via printGraphAsSource())
  step 2: run pass 1  -->  check the result is acyclic (topologicalSort())
                       -->  if NOT: throw, naming pass 1, stop immediately
                       -->  if OK: print the graph's new state
  step 3: run pass 2 on pass 1's own OUTPUT (never the original graph)
                       -->  check the result is acyclic
                       -->  if NOT: throw, naming pass 2, stop immediately
                       -->  if OK: print the graph's new state
  step 4: return the final graph, having survived every check along the way
```

File 014 below runs two different pipelines through `runPasses()`. The first, `[identityPass, canonicalizeNodeNames]`, is entirely well-behaved -- `identityPass()` is the simplest possible real transform pass (it rebuilds an exact, unchanged copy, existing purely to prove the manager correctly threads a graph through MULTIPLE passes in sequence, not just one), followed by Section 8.2's own canonicalization pass. Three distinct graph states get printed (the start, after `identityPass`, after `canonicalizeNodeNames`), and the final result's names are confirmed canonical. The second pipeline is the negative case this book tests in every chapter: `breakGraphByInsertingCycle()`, a deliberately broken "pass" that rebuilds an identity copy and then reaches past `Graph`'s own public API via `mutableNode()` -- the identical escape hatch Chapter 4's own File 006 used to prove its cycle detector worked -- to force a real cycle into its own output before returning it, simulating a real pass with a real bug. `runPasses()` is confirmed to throw immediately, naming `breakGraphByInsertingCycle` specifically, proving the post-pass validity check is not just decoration.

```text
WHAT breakGraphByInsertingCycle() ACTUALLY DOES, AND WHY IT'S A FAIR
TEST OF THE PASS MANAGER RATHER THAN A TRICK:

  it builds a perfectly normal identity copy first (same as
  identityPass()) -- then, ONLY as a deliberate test of the PassManager
  itself, calls mutableNode(0)->inputs.push_back(...) to give node 0
  (normally a leaf, an Input with NO inputs at all) an input pointing
  at the graph's own LAST node -- which already depends, through the
  chain of ordinary Add/Mul/ReLU nodes, back on node 0. A real cycle,
  through a real (if contrived) chain of dependencies.

  no ordinary TransformPass, built only through addInput/addConst/
  addUnary/addBinary, could ever produce this -- exactly like Chapter
  4's own cycle-freedom-by-construction argument. Reaching past the
  API on purpose is the ONLY way to construct a graph a normal pass
  literally cannot produce, which is exactly what makes it a fair,
  realistic test of what runPasses() would need to catch if a REAL
  pass ever had a REAL bug this severe.
```

```cpp
// Chapter 8: The Pass Manager: Structuring Compiler Transformations
// 014_pass_manager_runs_passes_and_validates_between_them.cpp
//
// Section 8.3 -- a real PassManager: something that runs a SEQUENCE of
// named TransformPasses over a Graph, one after another, printing a
// before/after dump of each one (finally putting Chapter 7's own
// round-trip-verified printer to real work) and checking the graph is
// still a valid DAG after every single pass -- not trusting each pass
// blindly, the same "check, don't just hope" discipline Chapter 4's
// topological sort and Chapter 6's shape validation both already
// established for their own, different reasons.
//
// This file reuses File 013's Value/Node/Graph/topologicalSort()/
// graphsStructurallyEqual()/TransformPass/canonicalizeNodeNames()
// completely unchanged, and adds two more passes purely to exercise the
// manager itself: identityPass() (the simplest possible real pass --
// rebuilds an identical copy, changing nothing at all) and, for the
// negative case every chapter in this book tests alongside its valid
// one, breakGraphByInsertingCycle() -- a deliberately broken "pass"
// that reaches past Graph's own public API (the same mutableNode()-style
// trick Chapter 4's own File 006 used) to hand back a graph containing
// a real cycle, proving the PassManager's own post-pass validity check
// actually catches a bad pass instead of silently accepting whatever
// it returns.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 014_pass_manager_runs_passes_and_validates_between_them.cpp -o 014_pass_manager_runs_passes_and_validates_between_them
// Run:     ./014_pass_manager_runs_passes_and_validates_between_them
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <functional>
#include <stdexcept>

// ==================== Value / Node / Graph (from Chapter 4 / File 013) ====================

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

    // Exposed ONLY so this file's own deliberately-broken test pass can
    // reach past the normal, cycle-safe API on purpose -- the same
    // escape hatch Chapter 4's File 006 used to prove its own cycle
    // detector actually works, reused here to prove the PASS MANAGER's
    // own post-pass validity check actually works.
    Node* mutableNode(int id) { return nodes_.at(static_cast<size_t>(id)).get(); }

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

// ==================== TransformPass / canonicalizeNodeNames() (from Section 8.1/8.2) ====================

using TransformPass = std::function<Graph(const Graph&)>;

static Graph canonicalizeNodeNames(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("canonicalizeNodeNames: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
        const Node* n = g.node(oldId);
        std::string canonicalName = "%" + std::to_string(result.size());
        Value newValue;
        if (n->op == OpKind::Input) {
            newValue = result.addInput(canonicalName);
        } else if (n->op == OpKind::Const) {
            newValue = result.addConst(n->constValue, canonicalName);
        } else if (n->inputs.size() == 1) {
            newValue = result.addUnary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId), canonicalName);
        } else {
            newValue = result.addBinary(n->op, oldIdToNewValue.at(n->inputs[0].nodeId),
                                         oldIdToNewValue.at(n->inputs[1].nodeId), canonicalName);
        }
        oldIdToNewValue[oldId] = newValue;
    }
    return result;
}

// ============================== Section 8.3: the PassManager itself (new) ==============================

// The simplest possible real TransformPass: rebuild an identical copy
// of the graph, in topological order, changing nothing at all. Exists
// purely so File 014's own pipeline test has more than one pass to
// chain together -- proving the manager threads a Graph through
// MULTIPLE passes correctly, not just one.
static Graph identityPass(const Graph& g) {
    TopoResult topo = topologicalSort(g);
    if (!topo.ok) throw std::runtime_error("identityPass: input graph is not acyclic");

    Graph result;
    std::map<int, Value> oldIdToNewValue;
    for (int oldId : topo.order) {
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

// A deliberately BROKEN "pass," for the negative test only: builds an
// identity copy exactly like identityPass() above, then reaches past
// the public API via mutableNode() (the same escape hatch Chapter 4's
// own File 006 used) to force a cycle into the result before returning
// it -- simulating a real pass with a real bug, to prove the
// PassManager's own post-pass check actually catches it.
static Graph breakGraphByInsertingCycle(const Graph& g) {
    Graph result = identityPass(g);
    if (result.size() < 2) throw std::runtime_error("breakGraphByInsertingCycle: needs at least 2 nodes");
    // Force node 0's inputs to include the LAST node -- node 0 normally
    // has none (it's a leaf), so this alone is enough to create a cycle
    // through the last node's own existing dependency chain back to 0.
    int lastId = static_cast<int>(result.size()) - 1;
    result.mutableNode(0)->inputs.push_back(Value{lastId, 0});
    return result;
}

struct NamedPass {
    std::string name;
    TransformPass fn;
};

// Runs every pass in `passes`, in order, threading the graph through
// each one. After EVERY pass, prints a before/after dump via Chapter
// 7's own printGraphAsSource() and checks the result is still a valid
// DAG via Chapter 4's own topologicalSort() -- if a pass ever returns
// an invalid graph, this throws immediately, naming exactly which pass
// was responsible, rather than letting a broken graph silently flow
// into whatever pass runs next.
static Graph runPasses(const Graph& initial, const std::vector<NamedPass>& passes) {
    if (passes.empty()) throw std::runtime_error("runPasses: at least one pass is required");
    printf("--- before any pass ---\n%s\n", printGraphAsSource(initial).c_str());

    // Graph deliberately has no copy constructor (it owns its nodes
    // through unique_ptr, exactly like Chapter 4's own Graph) -- so the
    // FIRST pass reads `initial` directly by const reference, and every
    // pass after that reads the previous pass's own returned Graph by
    // reference, moved into `current` only once it has already been
    // validated.
    Graph current = passes[0].fn(initial);
    {
        TopoResult topo = topologicalSort(current);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + passes[0].name +
                                      "' produced an invalid (non-acyclic) graph");
        }
    }
    printf("--- after pass '%s' ---\n%s\n", passes[0].name.c_str(), printGraphAsSource(current).c_str());

    for (size_t i = 1; i < passes.size(); ++i) {
        const NamedPass& p = passes[i];
        Graph next = p.fn(current);
        TopoResult topo = topologicalSort(next);
        if (!topo.ok) {
            throw std::runtime_error("runPasses: pass '" + p.name +
                                      "' produced an invalid (non-acyclic) graph");
        }
        printf("--- after pass '%s' ---\n%s\n", p.name.c_str(), printGraphAsSource(next).c_str());
        current = std::move(next);
    }
    return current;
}

int main() {
    printf("=== Section 8.3: PassManager runs a sequence of passes, validating between each ===\n\n");

    Graph diamond;
    Value a  = diamond.addInput("a");
    Value b  = diamond.addInput("b");
    Value t1 = diamond.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = diamond.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = diamond.addUnary(OpKind::ReLU, t1, "t3");
    diamond.addBinary(OpKind::Add, t2, t3, "out");

    printf(">>> Pipeline 1: [identityPass, canonicalizeNodeNames] over the diamond graph\n\n");
    Graph finalResult = runPasses(diamond, {
        {"identityPass", identityPass},
        {"canonicalizeNodeNames", canonicalizeNodeNames},
    });

    bool finalNamesCanonical = true;
    for (const auto& n : finalResult.nodes()) {
        if (n->debugName != "%" + std::to_string(n->id)) finalNamesCanonical = false;
    }
    printf("self-check: after both passes, every node's name is canonical (%s)\n",
           finalNamesCanonical ? "confirmed" : "MISMATCH");
    printf("self-check: the pipeline visited BOTH passes in order, not just the last one --\n");
    printf("the printed dumps above show 3 distinct states: before, after identityPass,\n");
    printf("and after canonicalizeNodeNames (visually confirmed by the 3 blocks above)\n\n");

    printf(">>> Pipeline 2: a deliberately BROKEN pass, to prove runPasses() catches it\n\n");
    bool caughtBrokenPass = false;
    std::string caughtMessage;
    try {
        runPasses(diamond, {
            {"identityPass", identityPass},
            {"breakGraphByInsertingCycle", breakGraphByInsertingCycle},
        });
    } catch (const std::exception& e) {
        caughtBrokenPass = true;
        caughtMessage = e.what();
    }
    printf("self-check: runPasses() threw when the broken pass ran, naming it specifically: \"%s\" (%s)\n",
           caughtMessage.c_str(), caughtBrokenPass ? "confirmed" : "MISMATCH");

    bool allOk = finalNamesCanonical && caughtBrokenPass;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 014_pass_manager_runs_passes_and_validates_between_them.cpp -o 014_pass_manager_runs_passes_and_validates_between_them
./014_pass_manager_runs_passes_and_validates_between_them
```

**Output:**

```text
=== Section 8.3: PassManager runs a sequence of passes, validating between each ===

>>> Pipeline 1: [identityPass, canonicalizeNodeNames] over the diamond graph

--- before any pass ---
a = input()
b = input()
t1 = add(a, b)
t2 = mul(t1, a)
t3 = relu(t1)
out = add(t2, t3)

--- after pass 'identityPass' ---
a = input()
b = input()
t1 = add(a, b)
t2 = mul(t1, a)
t3 = relu(t1)
out = add(t2, t3)

--- after pass 'canonicalizeNodeNames' ---
%0 = input()
%1 = input()
%2 = add(%0, %1)
%3 = mul(%2, %0)
%4 = relu(%2)
%5 = add(%3, %4)

self-check: after both passes, every node's name is canonical (confirmed)
self-check: the pipeline visited BOTH passes in order, not just the last one --
the printed dumps above show 3 distinct states: before, after identityPass,
and after canonicalizeNodeNames (visually confirmed by the 3 blocks above)

>>> Pipeline 2: a deliberately BROKEN pass, to prove runPasses() catches it

--- before any pass ---
a = input()
b = input()
t1 = add(a, b)
t2 = mul(t1, a)
t3 = relu(t1)
out = add(t2, t3)

--- after pass 'identityPass' ---
a = input()
b = input()
t1 = add(a, b)
t2 = mul(t1, a)
t3 = relu(t1)
out = add(t2, t3)

self-check: runPasses() threw when the broken pass ran, naming it specifically: "runPasses: pass 'breakGraphByInsertingCycle' produced an invalid (non-acyclic) graph" (confirmed)
```

!!! warning "[COMMON TRAP] Assuming `identityPass()` is pointless because it \"does nothing\""
    `identityPass()` really does produce a graph `graphsStructurallyEqual()` to its own input, every time -- in that sense, yes, it changes nothing about the computation. But its job in this chapter is not optimization, it's *testing the pipeline itself*: a `PassManager` whose only test ran a single pass could never prove it correctly threads a graph from ONE pass's output into the NEXT pass's input across multiple hops, which is precisely the thing `[identityPass, canonicalizeNodeNames]` proves by producing three genuinely distinct printed states rather than one. Part 2 will very shortly have several real, non-trivial passes to chain together for real; this chapter's own two-pass pipeline is a deliberately minimal proof that the chaining mechanism itself is correct before anything more elaborate is asked of it.

## Chapter Summary

This chapter formalized the second kind of pass this book's own graph needs, opening Part 2. Section 8.1 drew a precise line between an analysis pass (Chapters 6 and 7's `inferShapes()` and `printGraphAsSource()` -- reads a graph, produces something that is not a graph, never mutates its input) and a transform pass (reads a graph, produces a *different* graph), and made the resulting `TransformPass` signature, `Graph(const Graph&)`, a direct consequence of a real limitation already built into Chapter 4's own `Graph`: no in-place node removal, which Chapter 9's dead code elimination is specifically going to need. Section 8.2 built `canonicalizeNodeNames()`, this book's first genuine transform pass, solving a real problem (two structurally identical graphs printing as unrecognizably different text) by rewriting every node's name to reflect its own position, checked the strong way: `graphsStructurallyEqual()` (which never even looks at `debugName`) confirms the computation itself never changed, while the printed names demonstrably did. Section 8.3 built `runPasses()`, a real `PassManager` that threads a graph through a sequence of named passes, printing a trustworthy before/after dump of every single one via Chapter 7's own round-trip-verified printer, and checking the result is still a valid DAG after every pass via Chapter 4's own `topologicalSort()` -- confirmed both on a well-behaved two-pass pipeline and, the negative case this book tests in every chapter, on a deliberately broken pass that gets caught immediately and named specifically, rather than silently corrupting whatever ran after it. Part 2 now has real, working infrastructure -- a pass interface, a first working transform pass, and a manager that runs sequences of them safely -- for Chapter 9 to plug its first genuine optimizations into: constant folding and dead code elimination.

## Self-Check Questions

1. What concretely distinguishes an analysis pass from a transform pass in this book's own terms, using `inferShapes()` and `canonicalizeNodeNames()` as the two examples?
2. Why does a `TransformPass` have the signature `Graph(const Graph&)` -- reading an old graph and returning a new one -- rather than modifying its input `Graph` in place?
3. `canonicalizeNodeNames()` keeps an explicit `oldIdToNewValue` map rather than assuming a node's new id always equals its old id. Under what kind of future change would that assumption actually break?
4. Why is `graphsStructurallyEqual()` -- rather than, say, comparing the two graphs' printed text -- the right tool to confirm `canonicalizeNodeNames()` only changed names?
5. `runPasses()` checks the graph's validity after EVERY pass, not just once at the very end. What specific debugging problem does checking after every pass avoid, that checking only the final result would not?
6. Why does File 014's own `main()` run `[identityPass, canonicalizeNodeNames]` -- two passes -- rather than just running `canonicalizeNodeNames()` alone to test the pipeline?
7. `breakGraphByInsertingCycle()` has to reach past `Graph`'s own public API (via `mutableNode()`) to produce an invalid graph. Why can no pass built only from `addInput`/`addConst`/`addUnary`/`addBinary` ever produce one on its own?
8. Section 8.1's [COMMON TRAP] considers giving `Graph` a real `remove()` method instead of having passes rebuild a new graph. Name one concrete piece of bookkeeping a real `remove()` method would need to get right that the "rebuild, keep only what you want" approach never has to deal with at all.

## Where We Go Next

Part 2 now has everything it needs to start writing real optimizations: a precise vocabulary for what a pass is, a first working transform pass, and a `PassManager` that runs a sequence of passes safely, printing a trustworthy dump after each one and catching a broken pass immediately rather than letting it silently corrupt whatever runs next. Chapter 9, "Constant Folding and Dead Code Elimination," plugs the first two genuinely optimizing passes into exactly this infrastructure -- constant folding will replace a computation whose inputs are already known constants with the answer itself, and dead code elimination will finally put this chapter's own "rebuild rather than mutate" design to its intended use, dropping nodes nothing in the graph actually depends on.

## Worked Solutions

1. An analysis pass (`inferShapes()`) reads a `Graph` and produces something that is NOT a `Graph` at all -- a `map<int, Shape>` -- while never modifying the graph it read. A transform pass (`canonicalizeNodeNames()`) reads a `Graph` and produces a DIFFERENT `Graph` -- its result is a new instance of the exact same kind of thing it was given, built through the same `addInput`/`addConst`/`addUnary`/`addBinary` API any other code in this book uses, while (just like the analysis pass) never mutating its own input.
2. Because `Graph`, as Chapter 4 built it, has no way to remove a node once added -- only `addInput`/`addConst`/`addUnary`/`addBinary`, all strictly additive. Chapter 9's dead code elimination needs to drop nodes nothing downstream uses, which an in-place API without a real removal method cannot do. Reading an old graph and building a brand-new one lets a pass "drop" a node simply by never adding it to the new graph in the first place -- no removal method, and none of its bookkeeping problems, are ever needed.
3. For THIS book's own append-only `Graph` construction API, walking in topological order happens to assign new ids in the same order as old ids, so they currently coincide -- the same coincidence Chapter 6's `inferShapes()` and Chapter 7's printer both already rely on avoiding. That assumption would break for a pass that only rebuilds SOME of a graph's nodes rather than all of them (skipping some old nodes entirely would leave gaps, so old id 5 might no longer become new id 5), or one that visits nodes in some order other than strict topological/creation order. `canonicalizeNodeNames()`'s explicit map keeps working correctly in either case, because it never assumes the coincidence at all -- it looks up whatever the new Value actually was.
4. Because `graphsStructurallyEqual()` compares op kind and input ids only, and explicitly never looks at `debugName` at all -- it is checking exactly the property that is supposed to be UNCHANGED (the computation itself), completely independent of the property that IS supposed to change (the names). Comparing printed text instead would conflate the two: two graphs that compute the same thing but happen to have different canonical numbering (say, because they were canonicalized starting from a different node) could have different printed text despite being structurally identical, and a printed-text comparison would incorrectly call that a mismatch.
5. Checking only the final result cannot say WHICH pass in a multi-pass sequence actually introduced a problem -- by the time the final check runs, every later pass has already executed on top of whatever the earlier, buggy pass produced, and any of them could have further transformed (or obscured) the original mistake. Checking after every pass catches a problem at the exact pass that caused it, before any later pass ever gets a chance to run on already-broken input, which is exactly the difference File 014's own "what a PassManager that only checked the final result would risk" diagram makes concrete.
6. Running `canonicalizeNodeNames()` alone would only prove the manager can execute a SINGLE pass correctly -- it says nothing about whether the manager correctly takes one pass's OUTPUT and feeds it in as the NEXT pass's INPUT, which is the entire point of chaining passes at all. Running two passes in sequence, and confirming three genuinely distinct printed states (before any pass, after the first, after the second), is the minimum needed to demonstrate the hand-off between passes actually works, not just that a single pass in isolation does.
7. Every one of those four methods can only ever take a `Value` that some EARLIER call already returned -- there is no way to construct a `Value` referring to a node that doesn't exist yet in the graph being built. That means any node built only through this API always has inputs with strictly smaller ids than its own, which makes a cycle structurally impossible to construct through the normal API alone -- the identical cycle-freedom-by-construction argument Chapter 4 first proved for `Graph` itself.
8. A real `remove()` method would need to find and fix up every OTHER node in the graph whose own `inputs` list might reference the node being removed (or explicitly forbid removing a node any other node still depends on, which is its own separate check to get right) -- and it would also need to decide what happens to node ids afterward: do surviving nodes' ids shift down to close the gap, or does the removed id simply become a permanent hole in the graph? The "rebuild a new graph, keep only what belongs" approach never has to answer either question: a dropped node is just never re-added, and the new graph's ids are freely reassigned as the pass walks the old one, with no fix-up step and no decision about gaps ever required.

---

**Sources cited in this chapter:**

None. This chapter's own pass interface (`TransformPass`), `canonicalizeNodeNames()`, and `PassManager` (`runPasses()`) are original designs for this book, built entirely from vocabulary and machinery Chapters 2, 4, 5, 6, and 7 already established. The general distinction between analysis and transform passes is a well-known architectural pattern real compilers use (LLVM's own pass infrastructure is one widely known example), but this chapter is stating a general, widely-known architectural fact rather than a specific claim from any one project's documentation, so -- consistent with how Chapter 5 treated its own comparison to LLVM's flat IR text format -- no citation is attached to it.
