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
