// Chapter 7: Printing and Visualizing CUDA Hammer's IR
// 012_visualizing_the_ir_as_graphviz_dot.cpp
//
// Section 7.3 -- text is enough to feed a program back into itself, but
// a human trying to SEE a graph's shape (where does it branch, where
// does it merge back together, how deep is it) reads a picture far
// faster than a column of "name = op(args)" lines. This file exports a
// Graph to Graphviz's real DOT language -- not an invented ASCII-art
// format of this book's own -- so any of Graphviz's own tools can render
// it (`dot -Tsvg`, `dot -Tpng`, ...).
//
// The DOT language's actual grammar, straight from Graphviz's own
// documentation (https://graphviz.org/doc/info/lang.html): a directed
// graph is declared as `digraph ID { stmt_list }`; a directed edge uses
// the `->` operator ("An edgeop is -> in directed graphs and -- in
// undirected graphs"); a node or edge takes an attribute list in square
// brackets, `[ ID = ID ]`, for things like a display `label`. This file
// uses exactly that syntax -- nothing more.
//
// This chapter's real correctness claim is stronger than "the text we
// produce looks like DOT": the generated file is fed to the ACTUAL
// Graphviz `dot` binary (confirmed installed on this machine before
// running this file) and its exit code and output are checked -- if
// Graphviz's own parser rejects the file, this file's self-check fails.
// We are not eyeballing our own output; we are asking the real tool
// whose syntax we are claiming to speak.
//
// Reuses Chapter 4's Value/Node/Graph, and Chapter 6's Shape/
// broadcastShapes()/inferShapes() (which itself reuses Chapter 4's
// topologicalSort() unchanged) so each node's label in the picture can
// show its inferred shape too -- tying together everything Part 1 built:
// the graph (Ch4), how it's written down (Ch5), what it's allowed to
// compute (Ch6), and now, how it's actually SEEN (Ch7).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 012_visualizing_the_ir_as_graphviz_dot.cpp -o 012_visualizing_the_ir_as_graphviz_dot
// Run:     ./012_visualizing_the_ir_as_graphviz_dot
// (Requires the `dot` command from Graphviz to be installed for the
// self-check; the DOT text itself is produced either way.)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include <fstream>

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

// ==================== Shape / broadcasting / inference (from Chapter 6, unchanged) ====================

struct Shape {
    std::vector<int> dims;
};

static std::string shapeStr(const Shape& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s.dims[i]);
    }
    out += "]";
    return out;
}

static Shape broadcastShapes(const Shape& a, const Shape& b, const std::string& context) {
    size_t rank = std::max(a.dims.size(), b.dims.size());
    std::vector<int> result(rank);
    for (size_t i = 0; i < rank; ++i) {
        int da = (i < a.dims.size()) ? a.dims[a.dims.size() - 1 - i] : 1;
        int db = (i < b.dims.size()) ? b.dims[b.dims.size() - 1 - i] : 1;
        int outDim;
        if (da == db) {
            outDim = da;
        } else if (da == 1) {
            outDim = db;
        } else if (db == 1) {
            outDim = da;
        } else {
            throw std::runtime_error(context + ": shapes " + shapeStr(a) + " and " + shapeStr(b) +
                                      " are not broadcast-compatible");
        }
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
        std::string context = "node %" + std::to_string(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            auto it = declaredShapes.find(id);
            if (it == declaredShapes.end()) throw std::runtime_error(context + ": no declared shape");
            shapes[id] = it->second;
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        } else {
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

// ============================== Section 7.3: Graphviz DOT export (new) ==============================
//
// One node statement per Node ("n3 [label=\"...\"];"), one edge statement
// per (input -> consumer) pair ("n1 -> n3;"). Graphviz's own DOT id rules
// allow a bare, unquoted identifier for a node id as long as it is a
// letter/underscore followed by letters/digits/underscores -- "n3" always
// qualifies, which is exactly why node ids are printed as "n" + the
// integer id rather than risking a debugName that might contain a
// character DOT would need quoted.
static std::string exportToDot(const Graph& g, const std::map<int, Shape>* shapes = nullptr) {
    std::string out = "digraph IR {\n";
    out += "  node [shape=box, fontname=\"monospace\"];\n";

    for (const auto& n : g.nodes()) {
        std::string label = n->debugName + "\\n" + opKindStr(n->op);
        if (shapes) {
            auto it = shapes->find(n->id);
            if (it != shapes->end()) label += "\\n" + shapeStr(it->second);
        }
        out += "  n" + std::to_string(n->id) + " [label=\"" + label + "\"];\n";
    }
    out += "\n";
    for (const auto& n : g.nodes()) {
        for (const Value& in : n->inputs) {
            out += "  n" + std::to_string(in.nodeId) + " -> n" + std::to_string(n->id) + ";\n";
        }
    }
    out += "}\n";
    return out;
}

// Runs the real `dot` binary against a generated file and returns true
// only if it exits 0 -- i.e. Graphviz's OWN parser accepted the syntax.
static bool validateWithRealGraphviz(const std::string& dotText, const std::string& tmpPath) {
    std::ofstream f(tmpPath);
    f << dotText;
    f.close();
    std::string cmd = "dot -Tplain " + tmpPath + " -o " + tmpPath + ".out 2>" + tmpPath + ".err";
    int rc = std::system(cmd.c_str());
    return rc == 0;
}

int main() {
    printf("=== Section 7.3: exporting the graph as real Graphviz DOT ===\n\n");

    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");

    std::map<int, Shape> declared = {{a.nodeId, Shape{{3, 4}}}, {b.nodeId, Shape{{4}}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);

    std::string dotText = exportToDot(g, &shapes);
    printf("Generated DOT source:\n%s\n", dotText.c_str());

    // Structural self-checks on the text itself, independent of whether
    // `dot` is available: exactly one node statement per Node, and
    // exactly one edge statement per input edge in the whole graph.
    size_t nodeStatementCount = 0, pos = 0;
    while ((pos = dotText.find("[label=", pos)) != std::string::npos) { ++nodeStatementCount; pos += 7; }
    size_t edgeStatementCount = 0;
    pos = 0;
    while ((pos = dotText.find("->", pos)) != std::string::npos) { ++edgeStatementCount; pos += 2; }

    size_t expectedEdgeCount = 0;
    for (const auto& n : g.nodes()) expectedEdgeCount += n->inputs.size();

    bool nodeCountMatches = (nodeStatementCount == g.size());
    bool edgeCountMatches = (edgeStatementCount == expectedEdgeCount);
    printf("self-check: %zu node statements emitted, matching the graph's %zu nodes (%s)\n",
           nodeStatementCount, g.size(), nodeCountMatches ? "confirmed" : "MISMATCH");
    printf("self-check: %zu edge statements emitted, matching the graph's %zu real input edges (%s)\n",
           edgeStatementCount, expectedEdgeCount, edgeCountMatches ? "confirmed" : "MISMATCH");

    // Out is reached by every path through the diamond -- if the edge
    // count is right, "out" specifically must have in-degree 2 in the
    // DOT text (two "-> n5" occurrences, matching its two real inputs).
    size_t outInEdges = 0;
    pos = 0;
    std::string outTarget = "-> n" + std::to_string(out.nodeId) + ";";
    while ((pos = dotText.find(outTarget, pos)) != std::string::npos) { ++outInEdges; pos += outTarget.size(); }
    bool outInDegreeCorrect = (outInEdges == 2);
    printf("self-check: 'out' (n%d) has exactly 2 incoming edges in the DOT text (%s)\n",
           out.nodeId, outInDegreeCorrect ? "confirmed" : "MISMATCH");

    // The strong check: hand this exact text to the REAL Graphviz tool.
    bool graphvizAccepted = false;
    bool dotAvailable = (std::system("which dot > /dev/null 2>&1") == 0);
    if (dotAvailable) {
        graphvizAccepted = validateWithRealGraphviz(dotText, "/tmp/012_chapter7_graph.dot");
        printf("self-check: the real `dot` binary parsed this exact file without error (%s)\n",
               graphvizAccepted ? "confirmed" : "MISMATCH");
    } else {
        printf("(`dot` not found on this machine -- skipping the real-Graphviz validation step)\n");
    }

    bool allOk = nodeCountMatches && edgeCountMatches && outInDegreeCorrect && (!dotAvailable || graphvizAccepted);
    return allOk ? 0 : 1;
}
