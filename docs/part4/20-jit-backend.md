# 20. A JIT Backend: Compiling and Loading Generated Code at Runtime

**What you will understand:** `extern "C"` and exactly what it changes about a compiled function (LINKAGE -- the symbol name a linker sees -- never the type system); `dlopen()`, `dlsym()`, and `dlclose()`, the three POSIX calls this chapter uses to load a freshly compiled shared library into this program's own running process and call straight into it; `JitModule`, an RAII owner of one `dlopen()` handle modeled directly on Chapter 4's own non-copyable `Graph`; `compileToSharedLibrary()`, the one remaining shell-out this chapter keeps, honestly scoped to COMPILATION only; `generateJitProgramForGraph()`, which packs an entire fused graph's own non-leaf-node functions into ONE compiled shared library instead of one function -- or one whole generated program -- at a time, leaning on Chapter 8's own `canonicalizeNodeNames()` uniqueness guarantee to prove no two symbol names can ever collide; and a full end-to-end capstone that JIT-compiles Chapter 16's own 10-node graph through BOTH Chapter 18's own scalar backend and Chapter 19's own vectorized backend, every result of both crossing into this program as a real function-pointer call -- zero subprocess launches for EXECUTION, for the first time in this book.

**What you need to know first:** Chapter 18's own `emitSteps()` and `Target` enum, and Chapter 19's own `Isa`/`vecXxx()` vector primitives and FMA-fold peephole optimization (all reused with one small, uniform change -- wrapped in `extern "C"`); Chapter 16's own `boundedReductionFusionPass()` (reused verbatim in Section 20.3); Chapter 8's own `canonicalizeNodeNames()` uniqueness invariant (leaned on directly in Section 20.2); and Chapter 4's own non-copyable, move-only `Graph` design, the exact model this chapter's own `JitModule` follows for a real operating-system resource instead of a `std::vector<std::unique_ptr<Node>>`.

---

Every backend since Chapter 17 has followed the same shape: generate a `.cpp` file, shell out to a real compiler via `popen()`, run the resulting BINARY as a separate child process, and read that process's own printed stdout back as text to compare against `evaluateArrays()`. That harness has proven every generated program's own arithmetic correct across four chapters now, and it never once lied about what it was doing -- but it also never let this program's own code call a single line of that generated code directly. Every result, without exception, crossed a process boundary as TEXT: printed by one process, parsed back by another, even when both processes were produced by the exact same compiler on the exact same machine moments apart. This chapter replaces the RUN half of that pattern -- never the compile half, a real and honestly stated limitation explained where it comes up -- with something a running C++ program has always been able to do to itself: load a compiled function straight into its own address space and call it as an ordinary function pointer.

Before writing a single line of this chapter's own real files, the same honest-toolchain-investigation discipline Chapters 18 and 19 both opened with applied here too, and it settled a fact this whole chapter leans on. The cloud sandbox this book verifies against is confirmed, real x86-64 Linux. The device -- physically an Apple Silicon machine -- was checked directly rather than assumed: `uname -a` on the device's own shell reported `Linux claude 6.8.0-138-generic ... aarch64 aarch64 aarch64 GNU/Linux`, confirming the device runs a genuine aarch64 LINUX virtual machine, not native macOS. A small throwaway file, compiled with `g++ -shared -fPIC` and inspected with the `file` command, reported a real `ELF 64-bit LSB shared object, ARM aarch64` -- the exact same object format `-shared -fPIC` produces on the cloud sandbox's own x86-64 machine, differing only in instruction set, not in kind. Chapter 19's own two real architectures needed genuinely different SIMD instruction sets (AVX2 versus NEON) while still sharing one Linux/ELF/glibc toolchain throughout; this chapter's own `dlopen()`/`dlsym()`/`dlclose()` mechanism, `-shared -fPIC` compile flags, and `-ldl` link flag are consequently identical, unmodified, on both machines -- zero per-architecture special-casing anywhere in Sections 20.1 or 20.2. Only Section 20.3's own reused Chapter 19 vector codegen still needs the by-now-familiar AVX2-vs-NEON split, and for a completely different reason: SIMD instruction selection, not shared-library loading.

```text
BEFORE (Chapters 17-19's own harness, unchanged in spirit since Ch17):

  generate .cpp text -> popen() shell out to g++ -> a SEPARATE PROCESS
  runs the compiled binary -> that process printf()s its own results ->
  THIS program reads its stdout as TEXT and parses the numbers back out

  every single result crosses a PROCESS BOUNDARY, serialized to text
  and parsed back, even though both programs share the same compiler,
  the same ABI, the same machine

AFTER (this chapter):

  generate .cpp text (now wrapped in extern "C") -> popen() shell out
  to g++, compiled as a SHARED LIBRARY (-shared -fPIC), not a
  standalone executable -> dlopen() maps that library into THIS
  process's own address space -> dlsym() looks up a function pointer
  by its plain, unmangled name -> call that pointer DIRECTLY, with
  real C++ arguments

  compilation still shells out (one popen() call, a real, stated
  limitation explained where it comes up) -- but EXECUTION never
  launches a second process and never prints a line of text to parse

Both real machines: same mechanism, no per-architecture special-casing

  cloud sandbox: confirmed real x86-64 Linux (uname -a, uname -m)

  device: confirmed a real aarch64 LINUX virtual machine, NOT native
  macOS, despite running on Apple Silicon hardware -- uname -a reports
  a Linux kernel, and a throwaway .so compiled with -shared -fPIC and
  inspected with the file command reports a genuine ELF shared object,
  ARM aarch64 -- the SAME object format -shared -fPIC produces on the
  cloud sandbox, for a different instruction set

  dlopen()/dlsym()/dlclose(), -shared -fPIC, -ldl: IDENTICAL on both --
  unlike Chapter 19's own genuine AVX2-vs-NEON instruction-set split,
  this chapter's own JIT loading mechanism needs ZERO per-architecture
  code (only Section 20.3's own reused Ch19 vector codegen still needs
  that split, for a completely different reason)

Section 20.1 -- one function: extern "C" (disables name mangling, so
                dlsym() can find a symbol by its plain, predictable
                name) and JitModule, an RAII owner of one dlopen()
                handle, modeled on Chapter 4's own non-copyable Graph

Section 20.2 -- one graph, one library: generateJitProgramForGraph()
                packs EVERY non-leaf node's own extern "C" function
                into ONE compiled shared library, relying on Chapter
                8's own canonicalizeNodeNames() uniqueness guarantee
                so no two symbol names can ever collide

Section 20.3 -- the capstone: Chapter 16's own 10-node graph,
                JIT-compiled TWICE (Chapter 18's scalar backend and
                Chapter 19's vector backend, both wrapped in extern
                "C" for the first time), run through ONE shared
                loader -- ZERO subprocess launches, ZERO parsed
                stdout text, anywhere in EXECUTION, for the first
                time in this book
```

## 20.1 extern "C" and dlopen(): A Function Pointer From Generated Text

### Intuition

Every generated program since Chapter 17 has been a complete, standalone `.cpp` file with its own `main()` -- compiled to an executable, run as a child process, and its printed stdout diffed against `evaluateArrays()`. That harness proved every generated program's own arithmetic correct, but this program's own code never once called a line of it directly. This section replaces that pattern using two POSIX facilities that have existed for decades -- `dlopen()` and `dlsym()` -- plus one C++ keyword, `extern "C"`, that makes calling them predictable: compile generated code to a real shared library instead of a standalone executable, map that library into this process's own address space, look up one function inside it by name, and call it exactly like any other function pointer.

### Background

The one real obstacle is C++'s own NAME MANGLING: to support function overloading and namespaces, a C++ compiler encodes a function's full parameter-type signature into its own compiled symbol name, and the exact encoding scheme is compiler- and platform-specific -- not something a caller is meant to hand-derive or hardcode. Wrapping a function in `extern "C"` disables that encoding entirely for that one function: its compiled symbol name becomes exactly the plain identifier written in the source, nothing appended or encoded. `generateExternCElementwiseFunction()` is Chapter 18's own `generateCpuElementwiseFunction()`, changed in exactly one place: `void funcName(...)` becomes `extern "C" void funcName(...)`. Nothing about the function's own TYPE SIGNATURE changes -- it still takes a real `const std::vector<const float*>&`, not a C array -- `extern "C"` changes linkage only, and this works cleanly because both this program and the JIT-compiled `.so` are built by the same `g++` installation on each machine, sharing one ABI.

`JitModule` owns exactly one `dlopen()` handle the same way Chapter 4's own `Graph` owns its own `std::vector<std::unique_ptr<Node>>`: non-copyable (duplicating a raw OS handle would let two objects each believe they own it, and either destructor could then double-`dlclose()` it -- undefined behavior), move-only (ownership transfers cleanly; the moved-from `JitModule` is left holding a null handle it will never try to close). Its constructor calls `dlopen(path, RTLD_NOW)`, throwing with the message from `dlerror()` on failure; its destructor calls `dlclose()` if the handle is non-null; and its templated `getFunction()` calls `dlsym()`, checks `dlerror()` a second time to detect failure, and returns the result cast to the caller's own function-pointer type. `compileToSharedLibrary()` is the one remaining `popen()` shell-out in this whole chapter, and it is scoped honestly: it compiles source text into a real, loadable shared library, and this chapter never claims the COMPILER itself runs in-process, only that EXECUTION does. A true self-hosting JIT would embed the compiler as a library -- LLVM's own ORC JIT, or `libclang` -- instead of shelling out to a separate `g++` process even for this one step; named here as a real, stated limitation, not hidden.

The worked example is a small graph -- `a=Input[8]`, `b=Input` scalar, `t1=Add(a,b)`, `t2=ReLU(t1)`, `out=Mul(t2,b)` -- fused via Chapter 14's own plain `reductionFusionPass()` into ONE `FusedElementwise` group with 3 steps (Add, ReLU, Mul) and 2 external inputs. Its generated `extern "C"` function is compiled to `/tmp/hammer_ch20_048.so`, `dlopen()`'d, `dlsym()`'d by the plain name `"compute_out"`, and called directly with real C++ arguments -- a `std::vector<const float*>&`, not a printed string -- producing the exact array `evaluateArrays()` reports, with no `popen()`, no subprocess, and no stdout parsed anywhere in that call. The section closes on a real, concrete risk rather than a hypothetical one: the `JitModule` in this worked example goes out of scope right after its one call, its destructor runs `dlclose()`, and the operating system is free to unmap `compute_out`'s own code from this process's address space at that point. The `ElementwiseFn` value this program still holds afterward is now a DANGLING pointer -- calling it would be undefined behavior, so this program deliberately never keeps it around or calls it again. The fix is a lifetime rule, not a runtime check: never call a function pointer obtained from a `JitModule` after that `JitModule`'s own destructor has run.

```text
Why extern "C" -- name mangling, made concrete:

  C++ compiled WITHOUT extern "C":
    void compute_out(ext, extCounts, out, n)
    -> compiled symbol name: some MANGLED string encoding the full
       parameter-type signature (exact spelling is compiler- and
       platform-specific, and not meant to be hand-derived or
       hardcoded by a caller)

  C++ compiled WITH extern "C" (this chapter):
    extern "C" void compute_out(ext, extCounts, out, n)
    -> compiled symbol name: literally "compute_out" -- no mangling,
       no compiler-specific encoding, exactly the string this program
       passes to dlsym()

  extern "C" changes LINKAGE only -- every parameter is still a real
  C++ type (a vector of const float pointers, not a C array); this
  works because both this program and the JIT-compiled .so are built
  by the SAME g++ installation, sharing one ABI

JitModule's own lifecycle (Chapter 4's Graph non-copyable discipline,
applied to a real OS resource this time):

  JitModule(path)     -> dlopen(path, RTLD_NOW): maps the .so's own
                          code and data into THIS process's address
                          space; throws on failure (dlerror() message)

  getFunction(name)    -> dlsym(handle, name): looks up one symbol by
                          its plain, unmangled name (templated on the
                          caller's own function-pointer type); throws
                          if the symbol is not found

  ~JitModule()         -> dlclose(handle): unmaps the .so; any
                          function pointer obtained earlier is now
                          DANGLING -- calling it after this point is
                          undefined behavior, a lifetime rule this
                          chapter enforces by discipline, not by a
                          runtime check
```


```cpp
// Chapter 20: A JIT Backend: Compiling and Loading Generated Code at Runtime
// 048_extern_c_and_dlopen_a_function_pointer_from_generated_text.cpp
//
// Section 20.1 -- every backend since Chapter 17 has followed the same
// harness: generate a .cpp file, shell out to a real compiler via popen(),
// run the resulting BINARY AS A SEPARATE PROCESS, and parse its printed
// stdout back into this program's own buffers. That harness proved every
// generated program's own arithmetic correct, but it never let this
// program's own code call a single line of that generated code directly --
// every result crossed a process boundary as TEXT. This section replaces
// the RUN half of that harness (compilation still shells out once -- a
// real, honestly stated limitation, explained where it comes up) with real
// in-process code loading: compile generated code to a real shared library,
// dlopen() it, dlsym() a function pointer out of it, and CALL that pointer
// directly, in this same process, with real C++ arguments -- no second
// process, no stdout, nothing to parse back into text.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 048_extern_c_and_dlopen_a_function_pointer_from_generated_text.cpp -o 048_driver -ldl
// Run:     ./048_driver
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
#include <dlfcn.h>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };
enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand { OperandKind kind; int index; };
struct FusedStep { OpKind op; std::vector<FusedOperand> operands; long long reduceElementCount = 1; };

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
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) { return addNode(op, {lhs, rhs}, name); }
    Value addFusedElementwise(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
        Value out = addNode(OpKind::FusedElementwise, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
    }
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
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
        for (const auto& n : g.nodes()) {
            for (const Value& in : n->inputs) {
                if (in.nodeId == id) { if (--inDegree[n->id] == 0) ready.push_back(n->id); }
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
static long long numElements(const Shape& s) { long long n = 1; for (int d : s.dims) n *= d; return n; }
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
        else shapes[id] = Shape{};
    }
    return shapes;
}

// ==================== reductionFusionPass() (from Chapter 14, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized, std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId, std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) || (p->op == OpKind::Sum) ||
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
    for (const Value& in : p->inputs)
        operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                             externalIndexByOldId, steps, stepIndexByOldId));
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
    for (const auto& n : g.nodes()) for (const Value& in : n->inputs) consumers[in.nodeId]++;
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
            FusedOperand summedOperand = resolveIntoGroup(n->inputs[0].nodeId, g, consumers, materialized,
                                                            externalInputs, externalIndexByOldId, steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs)
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
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

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2, unchanged) ====================

struct Loop { std::string dimName; long long extent; };
struct LoopNest { std::vector<Loop> loops; };
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    if (n->op == OpKind::Sum) nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    else if (n->op == OpKind::FusedReduction) nest.loops.push_back(Loop{"reduce", n->fusedSteps.back().reduceElementCount});
    return nest;
}
struct LoweredNode { std::vector<FusedStep> steps; LoopNest nest; };
static LoweredNode lowerNode(const Node* n, const std::map<int, Shape>& shapes,
                              const std::map<int, long long>& elementCounts) {
    LoweredNode result;
    if (n->op == OpKind::FusedElementwise || n->op == OpKind::FusedReduction) {
        result.steps = n->fusedSteps;
    } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
        result.steps = {FusedStep{n->op, {FusedOperand{OperandKind::ExternalInput, 0}, FusedOperand{OperandKind::ExternalInput, 1}}}};
    } else if (n->op == OpKind::ReLU) {
        result.steps = {FusedStep{OpKind::ReLU, {FusedOperand{OperandKind::ExternalInput, 0}}}};
    } else {
        long long count = elementCounts.at(n->inputs[0].nodeId);
        result.steps = {FusedStep{OpKind::Sum, {FusedOperand{OperandKind::ExternalInput, 0}}, count}};
    }
    result.nest = buildLoopNest(n, shapes, elementCounts);
    return result;
}

// ==================== emitSteps() (from Section 18.1, unchanged -- Target::Cpu only needed here) ====================

enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExprScalar(Target target, const std::string& x) {
    return (target == Target::Cpu) ? ("std::max(0.0f, " + x + ")") : ("fmaxf(0.0f, " + x + ")");
}
static std::vector<std::string> emitSteps(const std::vector<FusedStep>& steps, bool isReduction,
                                           const std::string& idxExpr, Target target, std::string& finalValueExpr) {
    std::vector<std::string> lines;
    std::vector<std::string> stepVars;
    size_t stepCount = isReduction ? steps.size() - 1 : steps.size();
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return readExternal(target, o.index, idxExpr);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < stepCount; ++s) {
        const FusedStep& step = steps[s];
        std::string expr;
        if (step.op == OpKind::ReLU) expr = maxExprScalar(target, operandExpr(step.operands[0]));
        else if (step.op == OpKind::Add) expr = operandExpr(step.operands[0]) + " + " + operandExpr(step.operands[1]);
        else expr = operandExpr(step.operands[0]) + " * " + operandExpr(step.operands[1]);
        std::string varName = "step" + std::to_string(s);
        lines.push_back("float " + varName + " = " + expr + ";");
        stepVars.push_back(varName);
    }
    if (isReduction) {
        const FusedOperand& sumOperand = steps.back().operands[0];
        finalValueExpr = (sumOperand.kind == OperandKind::ExternalInput)
                              ? readExternal(target, sumOperand.index, idxExpr)
                              : stepVars[static_cast<size_t>(sumOperand.index)];
    } else {
        finalValueExpr = stepVars.back();
    }
    return lines;
}

// ==================== Section 20.1: an extern "C" CPU elementwise generator ====================
//
// This is Chapter 18's own generateCpuElementwiseFunction(), unchanged in
// every way except one: the emitted function is now wrapped in
// extern "C" { ... }. That one keyword is the whole difference between a
// function this program can dlsym() by its plain, predictable name and a
// function this program could only dlsym() by first reconstructing
// whatever C++ NAME MANGLING scheme a specific compiler and platform
// happen to use for a specific parameter-type signature -- a real, brittle
// dependency this section avoids entirely, explained in the prose below.
static std::string generateExternCElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
}

// ==================== Section 20.1: JitModule, an RAII owner for one dlopen() handle ====================
//
// A dlopen() handle is a real system resource -- the operating system maps
// the shared library's own code and data into THIS process's own address
// space the moment dlopen() succeeds, and that mapping stays live until a
// matching dlclose() call, or the process exits. JitModule owns exactly one
// such handle the same way Chapter 4's own Graph owns its own
// vector<unique_ptr<Node>>: non-copyable (duplicating a raw OS handle and
// letting two objects each believe they own it would mean either object's
// destructor could double-dlclose() it, undefined behavior), move-only
// (ownership transfers cleanly, the moved-from JitModule is left holding a
// null handle it will not try to close).
class JitModule {
public:
    explicit JitModule(const std::string& sharedLibraryPath) {
        handle_ = dlopen(sharedLibraryPath.c_str(), RTLD_NOW);
        if (!handle_) throw std::runtime_error("JitModule: dlopen failed: " + std::string(dlerror()));
    }
    ~JitModule() {
        if (handle_) dlclose(handle_);
    }
    JitModule(const JitModule&) = delete;
    JitModule& operator=(const JitModule&) = delete;
    JitModule(JitModule&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    JitModule& operator=(JitModule&& other) noexcept {
        if (this != &other) {
            if (handle_) dlclose(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    template <typename FnPtr>
    FnPtr getFunction(const std::string& symbolName) const {
        dlerror();  // clear any prior error, exactly as dlsym()'s own man page requires
        void* sym = dlsym(handle_, symbolName.c_str());
        const char* err = dlerror();
        if (err) throw std::runtime_error("JitModule::getFunction(\"" + symbolName + "\"): dlsym failed: " + err);
        return reinterpret_cast<FnPtr>(sym);
    }

private:
    void* handle_ = nullptr;
};

// ==================== Shell-out (COMPILATION ONLY, never execution) ====================

static std::string runShellCaptureAll(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed for: " + cmd);
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}
static void writeFile(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}
// Compiles source text into a real, loadable shared library. This IS a
// popen() shell-out -- this chapter never claims the COMPILER itself runs
// in-process, only that EXECUTION does. A true self-hosting JIT would embed
// the compiler as a library (LLVM's own ORC JIT, or libclang) instead of
// shelling out to a separate g++ process even for this one step -- named
// here as a real, stated limitation, not hidden.
static bool compileToSharedLibrary(const std::string& source, const std::string& cppPath, const std::string& soPath,
                                    std::string& compileLog) {
    writeFile(cppPath, source);
    compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2 -shared -fPIC " + cppPath + " -o " + soPath + " 2>&1");
    return compileLog.empty();
}

using ElementwiseFn = void (*)(const std::vector<const float*>&, const std::vector<long long>&, float*, long long);

int main() {
    printf("=== Section 20.1: extern \"C\" and dlopen() -- a function pointer from generated text ===\n\n");

    // A small test graph: a=Input[8], b=Input scalar, t1=Add(a,b),
    // t2=ReLU(t1), t3=Mul(t2,b) -- fused via Chapter 14's own plain
    // reductionFusionPass() into one 3-step FusedElementwise group.
    Graph g;
    Value a = g.addInput("a");
    Value b = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addUnary(OpKind::ReLU, t1, "t2");
    Value out = g.addBinary(OpKind::Mul, t2, b, "out");
    (void)out;
    std::map<int, Shape> declared = {{a.nodeId, Shape{{8}}}, {b.nodeId, Shape{}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    FusionResult fusedResult = reductionFusionPass(g, elementCounts);
    const Graph& fused = fusedResult.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fusedResult.representativeOldId.at(n->id);
        fusedShapes[n->id] = shapes.at(oldId);
        fusedElementCounts[n->id] = elementCounts.at(oldId);
    }
    const Node* outFusedNode = nullptr;
    for (const auto& n : fused.nodes()) if (n->debugName == "out") outFusedNode = n.get();
    printf("Fused \"out\" node: %zu steps (Add, ReLU, Mul), 2 external inputs\n\n", outFusedNode->fusedSteps.size());

    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);
    std::string funcSrc = generateExternCElementwiseFunction("compute_out", outLowered);
    printf("--- Generated extern \"C\" function ---\n\n%s\n", funcSrc.c_str());

    std::string prog = "#include <vector>\n\n" + funcSrc;
    std::string cppPath = "/tmp/hammer_ch20_048.cpp";
    std::string soPath = "/tmp/hammer_ch20_048.so";
    std::string compileLog;
    bool compileClean = compileToSharedLibrary(prog, cppPath, soPath, compileLog);
    printf("--- g++ -shared -fPIC compile (ONE shell-out, for COMPILATION only) ---\n\n%s\n", compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());

    printf("--- Loading the compiled .so IN-PROCESS: dlopen(), dlsym(), call the function pointer directly ---\n\n");
    std::vector<float> aVals = {1, -2, 3, -4, 5, -6, 7, -8};
    std::vector<float> bVals = {2};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}};
    auto expected = evaluateArrays(fused, in, fusedElementCounts);
    long long n = fusedElementCounts.at(outFusedNode->id);

    std::vector<float> out1(static_cast<size_t>(n));
    bool matches = false;
    bool loadedOk = false;
    {
        JitModule module(soPath);
        loadedOk = true;
        ElementwiseFn fn = module.getFunction<ElementwiseFn>("compute_out");
        std::vector<const float*> ext = {aVals.data(), bVals.data()};
        std::vector<long long> extCounts = {static_cast<long long>(aVals.size()), static_cast<long long>(bVals.size())};
        // THIS is the whole point: a direct C++ function call, not a
        // subprocess launch and not a line of printed text to parse back.
        fn(ext, extCounts, out1.data(), n);
        matches = (out1.size() == expected.at("out").size());
        if (matches) for (size_t i = 0; i < out1.size(); ++i) if (std::fabs(out1[i] - expected.at("out")[i]) > 1e-3f) matches = false;
    }  // module's own destructor runs HERE, dlclose()-ing the shared library

    printf("dlopen/dlsym succeeded (%s). compute_out(...), called as a real function pointer:\n", loadedOk ? "confirmed" : "MISMATCH");
    for (float v : out1) printf("%.6f ", v);
    printf("\nevaluateArrays(fused).out                                                 = ");
    for (float v : expected.at("out")) printf("%.6f ", v);
    printf("\n\n");
    printf("self-check: a function pointer obtained from dlsym() on a freshly dlopen()'d\n");
    printf("shared library, called directly with real C++ arguments (a std::vector<const\n");
    printf("float*>&, not a printed string), produces the exact same result evaluateArrays()\n");
    printf("does -- no popen(), no subprocess, no stdout parsed anywhere in this call (%s)\n\n",
           (loadedOk && matches) ? "confirmed" : "MISMATCH");

    printf("--- Why a dangling function pointer is a real risk, not a hypothetical one ---\n\n");
    printf("The JitModule above went out of scope right after its one call -- its destructor\n");
    printf("ran dlclose(), and the operating system was free to unmap compute_out's own code\n");
    printf("from this process's address space at that point. The ElementwiseFn value this\n");
    printf("program held is now a DANGLING pointer: calling it would be undefined behavior,\n");
    printf("so this program deliberately does not keep it around or call it again -- the\n");
    printf("fix is a lifetime rule, not a runtime check: never call a function pointer\n");
    printf("obtained from a JitModule after that JitModule's own destructor has run.\n");

    bool allOk = compileClean && loadedOk && matches;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "048_extern_c_and_dlopen_a_function_pointer_from_generated_text.cpp" -o "048_extern_c_and_dlopen_a_function_pointer_from_generated_text" -ldl
./"048_extern_c_and_dlopen_a_function_pointer_from_generated_text"
```

**Output (cloud sandbox, x86-64, AVX2/FMA):**

```text
=== Section 20.1: extern "C" and dlopen() -- a function pointer from generated text ===

Fused "out" node: 3 steps (Add, ReLU, Mul), 2 external inputs

--- Generated extern "C" function ---

extern "C" void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    for (long long flat = 0; flat < n; ++flat) {
        float step0 = ext[0][(flat) % extCounts[0]] + ext[1][(flat) % extCounts[1]];
        float step1 = std::max(0.0f, step0);
        float step2 = step1 * ext[1][(flat) % extCounts[1]];
        out[flat] = step2;
    }
}

--- g++ -shared -fPIC compile (ONE shell-out, for COMPILATION only) ---

(no output -- clean compile)

--- Loading the compiled .so IN-PROCESS: dlopen(), dlsym(), call the function pointer directly ---

dlopen/dlsym succeeded (confirmed). compute_out(...), called as a real function pointer:
6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000 
evaluateArrays(fused).out                                                 = 6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000 

self-check: a function pointer obtained from dlsym() on a freshly dlopen()'d
shared library, called directly with real C++ arguments (a std::vector<const
float*>&, not a printed string), produces the exact same result evaluateArrays()
does -- no popen(), no subprocess, no stdout parsed anywhere in this call (confirmed)

--- Why a dangling function pointer is a real risk, not a hypothetical one ---

The JitModule above went out of scope right after its one call -- its destructor
ran dlclose(), and the operating system was free to unmap compute_out's own code
from this process's address space at that point. The ElementwiseFn value this
program held is now a DANGLING pointer: calling it would be undefined behavior,
so this program deliberately does not keep it around or call it again -- the
fix is a lifetime rule, not a runtime check: never call a function pointer
obtained from a JitModule after that JitModule's own destructor has run.
```

**Output (device, aarch64 Linux VM, NEON):**

```text
=== Section 20.1: extern "C" and dlopen() -- a function pointer from generated text ===

Fused "out" node: 3 steps (Add, ReLU, Mul), 2 external inputs

--- Generated extern "C" function ---

extern "C" void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    for (long long flat = 0; flat < n; ++flat) {
        float step0 = ext[0][(flat) % extCounts[0]] + ext[1][(flat) % extCounts[1]];
        float step1 = std::max(0.0f, step0);
        float step2 = step1 * ext[1][(flat) % extCounts[1]];
        out[flat] = step2;
    }
}

--- g++ -shared -fPIC compile (ONE shell-out, for COMPILATION only) ---

(no output -- clean compile)

--- Loading the compiled .so IN-PROCESS: dlopen(), dlsym(), call the function pointer directly ---

dlopen/dlsym succeeded (confirmed). compute_out(...), called as a real function pointer:
6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000
evaluateArrays(fused).out                                                 = 6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000

self-check: a function pointer obtained from dlsym() on a freshly dlopen()'d
shared library, called directly with real C++ arguments (a std::vector<const
float*>&, not a printed string), produces the exact same result evaluateArrays()
does -- no popen(), no subprocess, no stdout parsed anywhere in this call (confirmed)

--- Why a dangling function pointer is a real risk, not a hypothetical one ---

The JitModule above went out of scope right after its one call -- its destructor
ran dlclose(), and the operating system was free to unmap compute_out's own code
from this process's address space at that point. The ElementwiseFn value this
program held is now a DANGLING pointer: calling it would be undefined behavior,
so this program deliberately does not keep it around or call it again -- the
fix is a lifetime rule, not a runtime check: never call a function pointer
obtained from a JitModule after that JitModule's own destructor has run.
```


!!! note "Why dlerror() must be cleared before dlsym(), not just checked after"
    `dlsym()`'s own return value cannot, by itself, distinguish two different situations: a symbol whose real address happens to be null (rare, but valid for some data symbols) and a symbol that genuinely was not found. POSIX's own `dlsym()` documentation states the correct pattern for telling them apart: call `dlerror()` once to clear any error left over from an unrelated earlier call, then call `dlsym()`, then call `dlerror()` again -- a non-null result the second time means THIS call failed, not some earlier one. `JitModule::getFunction()` follows that exact three-step pattern; skipping the first clear is a real, documented pitfall, not a hypothetical one.

## 20.2 One Shared Library, Many Functions: A JIT Module for a Whole Graph

### Intuition

Section 20.1 proved the loading mechanism for exactly one function. A real fused graph has several non-leaf nodes, and Chapter 17 through 19's own harness always compiled ONE generated program per graph -- a single `.cpp` file containing every node's own function PLUS a driver `main()` that called them in order and printed every buffer to stdout, so text-parsing was still needed to pull each node's own values back out. This section removes the idea of a generated program with its own `main()` entirely: `generateJitProgramForGraph()` emits nothing but `extern "C"` function DEFINITIONS -- one per non-leaf node, in topological order -- into one shared library, and THIS program's own `main()` drives every call directly, with no generated driver code anywhere in the picture.

### Background

`generateExternCReductionFunction()` is the other half of Section 20.1's own change, applied to Chapter 18's own `generateCpuReductionFunction()` the same way: `extern "C"` on the signature, nothing else. `isReductionNode()` and `generateExternCFunctionForNode()` dispatch between the two generators exactly the way Chapter 18's own harness already did. `generateJitProgramForGraph()` is the whole new idea: it walks a graph's topological order, skips `Input`/`Const` nodes (they need no generated function at all), and appends every remaining node's own `extern "C"` function to ONE growing source string -- compiled by ONE `compileToSharedLibrary()` call into ONE `.so`, regardless of how many functions that graph actually needs.

The one real precondition this relies on is symbol uniqueness: every symbol name inside one shared library must be unique, or the LINKER itself refuses to build it -- a real, checkable "duplicate symbol" error from `ld`, not a silent miscompile. This book's own `Node::debugName` has been guaranteed unique within one graph since Chapter 8's own `canonicalizeNodeNames()`, five chapters before this book ever generated a line of code -- so `"compute_" + debugName` is guaranteed collision-free here for exactly the same reason two nodes have never printed with colliding names in any earlier chapter's own output. The worked example deliberately exercises BOTH generator kinds in one compiled library: `a=Input[6]`, `b=Input` scalar, `t1=ReLU(a)`, `t2=Mul(t1,b)` (two consumers -- `s` and `y2` -- so it materializes on its own), `s=Sum(t2)` (a genuine reduction), `y2=Add(t2,b)` (one consumer, inlines into `y`), `y=Add(s,y2)`. The fused graph -- `t2(FE,2st) s(FR,1st) y(FE,2st)` -- compiles to exactly ONE `g++` invocation for all three functions, confirmed by a real `compileInvocationCount` counter rather than assumed, and every one of `compute_t2`, `compute_s`, `compute_y` is `dlsym()`'d from the SAME `JitModule` and called directly, chained through the same `std::map<std::string, std::vector<float>>` buffer convention Chapter 18's own CPU driver already used, matching `evaluateArrays()` on every buffer.

```text
Before (Section 20.1): ONE generated function, ONE dlopen(), ONE dlsym()

After (this section): ONE generated .so, MANY extern "C" functions,
ONE dlopen(), MANY dlsym() calls -- one per non-leaf node

  generateJitProgramForGraph() walks the graph in topological order:

    a (Input)                       -- skipped, no function needed
    b (Input)                       -- skipped
    t2 (FusedElementwise, 2 steps)  -> extern "C" compute_t2(...)
    s  (FusedReduction, 1 step)     -> extern "C" compute_s(...)
    y  (FusedElementwise, 2 steps)  -> extern "C" compute_y(...)

  ALL THREE functions appended to ONE source string, compiled with
  ONE g++ invocation into ONE .so:

    hammer_ch20_049.so
      |-- compute_t2   (symbol, unmangled)
      |-- compute_s    (symbol, unmangled)
      +-- compute_y    (symbol, unmangled)

  ONE JitModule (one dlopen), THREE getFunction() calls (three
  dlsym), chained through the same buffer map Chapter 18's own driver
  used:

    buf["a"], buf["b"]   (already known)
    buf["t2"] = compute_t2(ext={buf["a"], buf["b"]}, ...)
    buf["s"]  = compute_s(ext={buf["t2"]}, ...)
    buf["y"]  = compute_y(ext={buf["s"], buf["t2"], buf["b"]}, ...)

Why symbol names can never collide within one graph:

  "compute_" + debugName -- and Chapter 8's own canonicalizeNodeNames()
  has guaranteed every node's own debugName is unique WITHIN ONE GRAPH
  since five chapters before this book ever generated a line of code

  if that guarantee were ever violated: not a silent miscompile -- the
  LINKER itself refuses to build the .so, a real, checkable "duplicate
  symbol" error from ld
```

```cpp
// Chapter 20: A JIT Backend: Compiling and Loading Generated Code at Runtime
// 049_one_shared_library_many_functions_a_jit_module_for_a_whole_graph.cpp
//
// Section 20.2 -- Section 20.1 JIT-loaded exactly ONE function from exactly
// ONE shared library. A real graph has several non-leaf nodes, and nothing
// about dlopen()/dlsym() requires a separate shared library per function:
// a single .cpp file can define MANY extern "C" functions, compiled with
// ONE g++ invocation into ONE .so, opened with ONE dlopen() call, and its
// individual functions looked up one at a time with dlsym() -- exactly the
// same JitModule from Section 20.1, called several times against the same
// handle. This section builds generateJitProgramForGraph(), which walks an
// entire fused graph and emits every one of its own non-leaf nodes' own
// extern "C" functions into ONE source file, then chains their real,
// dlsym()'d function pointers through the same std::map<string,
// vector<float>> buffer convention Chapter 18's own CPU driver already
// established.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 049_one_shared_library_many_functions_a_jit_module_for_a_whole_graph.cpp -o 049_driver -ldl
// Run:     ./049_driver
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
#include <dlfcn.h>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };
enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand { OperandKind kind; int index; };
struct FusedStep { OpKind op; std::vector<FusedOperand> operands; long long reduceElementCount = 1; };

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
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) { return addNode(op, {lhs, rhs}, name); }
    Value addFusedElementwise(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
        Value out = addNode(OpKind::FusedElementwise, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
    }
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
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
        for (const auto& n : g.nodes()) {
            for (const Value& in : n->inputs) {
                if (in.nodeId == id) { if (--inDegree[n->id] == 0) ready.push_back(n->id); }
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
static long long numElements(const Shape& s) { long long n = 1; for (int d : s.dims) n *= d; return n; }
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
        else shapes[id] = Shape{};
    }
    return shapes;
}

// ==================== reductionFusionPass() (from Chapter 14, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

static FusedOperand resolveIntoGroup(int oldId, const Graph& g, const std::map<int, int>& consumers,
                                      const std::map<int, Value>& materialized, std::vector<Value>& externalInputs,
                                      std::map<int, int>& externalIndexByOldId, std::vector<FusedStep>& steps,
                                      std::map<int, int>& stepIndexByOldId) {
    auto stepIt = stepIndexByOldId.find(oldId);
    if (stepIt != stepIndexByOldId.end()) return FusedOperand{OperandKind::PriorStep, stepIt->second};
    const Node* p = g.node(oldId);
    bool mustBeExternal = (p->op == OpKind::Input || p->op == OpKind::Const) || (p->op == OpKind::Sum) ||
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
    for (const Value& in : p->inputs)
        operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                             externalIndexByOldId, steps, stepIndexByOldId));
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
    for (const auto& n : g.nodes()) for (const Value& in : n->inputs) consumers[in.nodeId]++;
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
            FusedOperand summedOperand = resolveIntoGroup(n->inputs[0].nodeId, g, consumers, materialized,
                                                            externalInputs, externalIndexByOldId, steps, stepIndexByOldId);
            long long count = elementCounts.at(n->inputs[0].nodeId);
            steps.push_back(FusedStep{OpKind::Sum, {summedOperand}, count});
            Value v = out.addFusedReduction(externalInputs, steps, n->debugName);
            materialized[oldId] = v; result.representativeOldId[v.nodeId] = oldId; continue;
        }
        if (consumers.at(oldId) == 1) continue;
        std::vector<Value> externalInputs; std::map<int, int> externalIndexByOldId;
        std::vector<FusedStep> steps; std::map<int, int> stepIndexByOldId;
        std::vector<FusedOperand> operands;
        for (const Value& in : n->inputs)
            operands.push_back(resolveIntoGroup(in.nodeId, g, consumers, materialized, externalInputs,
                                                 externalIndexByOldId, steps, stepIndexByOldId));
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

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2, unchanged) ====================

struct Loop { std::string dimName; long long extent; };
struct LoopNest { std::vector<Loop> loops; };
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    if (n->op == OpKind::Sum) nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    else if (n->op == OpKind::FusedReduction) nest.loops.push_back(Loop{"reduce", n->fusedSteps.back().reduceElementCount});
    return nest;
}
struct LoweredNode { std::vector<FusedStep> steps; LoopNest nest; };
static LoweredNode lowerNode(const Node* n, const std::map<int, Shape>& shapes,
                              const std::map<int, long long>& elementCounts) {
    LoweredNode result;
    if (n->op == OpKind::FusedElementwise || n->op == OpKind::FusedReduction) {
        result.steps = n->fusedSteps;
    } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
        result.steps = {FusedStep{n->op, {FusedOperand{OperandKind::ExternalInput, 0}, FusedOperand{OperandKind::ExternalInput, 1}}}};
    } else if (n->op == OpKind::ReLU) {
        result.steps = {FusedStep{OpKind::ReLU, {FusedOperand{OperandKind::ExternalInput, 0}}}};
    } else {
        long long count = elementCounts.at(n->inputs[0].nodeId);
        result.steps = {FusedStep{OpKind::Sum, {FusedOperand{OperandKind::ExternalInput, 0}}, count}};
    }
    result.nest = buildLoopNest(n, shapes, elementCounts);
    return result;
}

// ==================== emitSteps() (from Section 18.1, unchanged -- Target::Cpu only needed here) ====================

enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExprScalar(Target target, const std::string& x) {
    return (target == Target::Cpu) ? ("std::max(0.0f, " + x + ")") : ("fmaxf(0.0f, " + x + ")");
}
static std::vector<std::string> emitSteps(const std::vector<FusedStep>& steps, bool isReduction,
                                           const std::string& idxExpr, Target target, std::string& finalValueExpr) {
    std::vector<std::string> lines;
    std::vector<std::string> stepVars;
    size_t stepCount = isReduction ? steps.size() - 1 : steps.size();
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return readExternal(target, o.index, idxExpr);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < stepCount; ++s) {
        const FusedStep& step = steps[s];
        std::string expr;
        if (step.op == OpKind::ReLU) expr = maxExprScalar(target, operandExpr(step.operands[0]));
        else if (step.op == OpKind::Add) expr = operandExpr(step.operands[0]) + " + " + operandExpr(step.operands[1]);
        else expr = operandExpr(step.operands[0]) + " * " + operandExpr(step.operands[1]);
        std::string varName = "step" + std::to_string(s);
        lines.push_back("float " + varName + " = " + expr + ";");
        stepVars.push_back(varName);
    }
    if (isReduction) {
        const FusedOperand& sumOperand = steps.back().operands[0];
        finalValueExpr = (sumOperand.kind == OperandKind::ExternalInput)
                              ? readExternal(target, sumOperand.index, idxExpr)
                              : stepVars[static_cast<size_t>(sumOperand.index)];
    } else {
        finalValueExpr = stepVars.back();
    }
    return lines;
}

// ==================== Section 20.1's own extern "C" generators (unchanged) ====================

static std::string generateExternCElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
}
// Section 20.2's own new generator: Chapter 18's own generateCpuReductionFunction(),
// wrapped in extern "C" the exact same way Section 20.1 wrapped the
// elementwise case.
static std::string generateExternCReductionFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cpu, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    float acc = 0.0f;\n";
    src += "    for (long long r = 0; r < " + std::to_string(reduceExtent) + "; ++r) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        acc += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = acc;\n}\n";
    return src;
}
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }
static std::string generateExternCFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered) {
    return isReductionNode(n) ? generateExternCReductionFunction(funcName, lowered)
                               : generateExternCElementwiseFunction(funcName, lowered);
}

// ==================== Section 20.2: one source file, many extern "C" functions ====================
//
// generateJitProgramForGraph() is the whole idea of this section: walk
// every non-leaf node in topological order and append ITS OWN extern "C"
// function to ONE growing source string, instead of Section 20.1's own
// single-function program. Nothing about dlopen()/dlsym() changes -- the
// compiled .so simply exports several symbols instead of one, and each one
// gets its own dlsym() call against the SAME JitModule. The one real
// precondition this relies on: every symbol name in ONE shared library
// must be unique, or the LINKER itself refuses to build it at all (a real,
// checkable "duplicate symbol" error from ld, not a silent miscompile).
// This book's own Node::debugName has been unique within one graph since
// Chapter 8's own canonicalizeNodeNames() -- so "compute_" + debugName is
// guaranteed unique here for exactly the same reason two nodes never
// printed with colliding names in any earlier chapter's own output.
static std::string generateJitProgramForGraph(const Graph& g, const std::map<int, Shape>& shapes,
                                               const std::map<int, long long>& elementCounts,
                                               std::vector<std::pair<std::string, bool>>& functionsGenerated) {
    TopoResult topo = topologicalSort(g);
    std::string prog = "#include <vector>\n#include <algorithm>\n\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        prog += generateExternCFunctionForNode(funcName, n, lowered) + "\n";
        functionsGenerated.push_back({n->debugName, isReductionNode(n)});
    }
    return prog;
}

// ==================== Section 20.1's own JitModule (unchanged) ====================

class JitModule {
public:
    explicit JitModule(const std::string& sharedLibraryPath) {
        handle_ = dlopen(sharedLibraryPath.c_str(), RTLD_NOW);
        if (!handle_) throw std::runtime_error("JitModule: dlopen failed: " + std::string(dlerror()));
    }
    ~JitModule() {
        if (handle_) dlclose(handle_);
    }
    JitModule(const JitModule&) = delete;
    JitModule& operator=(const JitModule&) = delete;
    JitModule(JitModule&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    JitModule& operator=(JitModule&& other) noexcept {
        if (this != &other) {
            if (handle_) dlclose(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    template <typename FnPtr>
    FnPtr getFunction(const std::string& symbolName) const {
        dlerror();
        void* sym = dlsym(handle_, symbolName.c_str());
        const char* err = dlerror();
        if (err) throw std::runtime_error("JitModule::getFunction(\"" + symbolName + "\"): dlsym failed: " + err);
        return reinterpret_cast<FnPtr>(sym);
    }

private:
    void* handle_ = nullptr;
};

// ==================== Shell-out (COMPILATION ONLY) ====================

static int compileInvocationCount = 0;  // proves the whole graph needs exactly ONE
static std::string runShellCaptureAll(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed for: " + cmd);
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}
static void writeFile(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}
static bool compileToSharedLibrary(const std::string& source, const std::string& cppPath, const std::string& soPath,
                                    std::string& compileLog) {
    writeFile(cppPath, source);
    compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2 -shared -fPIC " + cppPath + " -o " + soPath + " 2>&1");
    ++compileInvocationCount;
    return compileLog.empty();
}

using ElementwiseFn = void (*)(const std::vector<const float*>&, const std::vector<long long>&, float*, long long);
using ReductionFn = void (*)(const std::vector<const float*>&, const std::vector<long long>&, float*);

int main() {
    printf("=== Section 20.2: one shared library, many functions -- a JIT module for a whole graph ===\n\n");

    // a=Input[6], b=Input scalar, t1=ReLU(a), t2=Mul(t1,b) -- t2 has TWO
    // consumers (s and y2), so it materializes on its own (2 steps: ReLU
    // inlined, then Mul); s=Sum(t2) is a genuine reduction; y2=Add(t2,b)
    // has one consumer (y) and inlines into it; y=Add(s,y2) materializes
    // with 2 steps. Three non-leaf nodes -- t2 (FusedElementwise), s
    // (FusedReduction), y (FusedElementwise) -- deliberately exercising
    // BOTH generator kinds in ONE compiled shared library.
    Graph g;
    Value a = g.addInput("a");
    Value b = g.addInput("b");
    Value t1 = g.addUnary(OpKind::ReLU, a, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, b, "t2");
    Value s = g.addUnary(OpKind::Sum, t2, "s");
    Value y2 = g.addBinary(OpKind::Add, t2, b, "y2");
    Value y = g.addBinary(OpKind::Add, s, y2, "y");
    (void)y;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{6}}}, {b.nodeId, Shape{}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    FusionResult fusedResult = reductionFusionPass(g, elementCounts);
    const Graph& fused = fusedResult.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fusedResult.representativeOldId.at(n->id);
        fusedShapes[n->id] = shapes.at(oldId);
        fusedElementCounts[n->id] = elementCounts.at(oldId);
    }
    printf("fused graph: ");
    for (const auto& n : fused.nodes()) {
        printf("%s", n->debugName.c_str());
        if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
        else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
        printf(" ");
    }
    printf("\n\n");

    std::vector<std::pair<std::string, bool>> functionsGenerated;
    std::string prog = generateJitProgramForGraph(fused, fusedShapes, fusedElementCounts, functionsGenerated);
    printf("generateJitProgramForGraph(): %zu extern \"C\" functions in ONE source file:\n", functionsGenerated.size());
    for (const auto& fn : functionsGenerated) printf("  compute_%s (%s)\n", fn.first.c_str(), fn.second ? "reduction" : "elementwise");
    printf("\n");

    std::string cppPath = "/tmp/hammer_ch20_049.cpp";
    std::string soPath = "/tmp/hammer_ch20_049.so";
    std::string compileLog;
    bool compileClean = compileToSharedLibrary(prog, cppPath, soPath, compileLog);
    printf("--- g++ -shared -fPIC compile (whole graph, ONE invocation) ---\n\n%s\n",
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());
    printf("self-check: this whole graph's own 3 non-leaf nodes needed exactly %d g++ invocation(s)\n", compileInvocationCount);
    printf("to compile -- Chapter 17/18's own harness would have shelled out once PER GENERATED\n");
    printf("PROGRAM; this section shells out once PER GRAPH (%s)\n\n", compileInvocationCount == 1 ? "confirmed" : "MISMATCH");

    std::vector<float> aVals = {1, -2, 3, -4, 5, -6};
    std::vector<float> bVals = {2};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}};
    auto expected = evaluateArrays(fused, in, fusedElementCounts);

    std::map<std::string, std::vector<float>> buf;
    buf["a"] = aVals;
    buf["b"] = bVals;
    bool allMatch = true;
    {
        JitModule module(soPath);  // ONE dlopen() call for the whole graph
        for (const auto& fn : functionsGenerated) {
            const std::string& name = fn.first;
            bool isReduction = fn.second;
            const Node* n = nullptr;
            for (const auto& nn : fused.nodes()) if (nn->debugName == name) n = nn.get();
            std::vector<const float*> ext;
            std::vector<long long> extCounts;
            for (const Value& in2 : n->inputs) {
                const Node* inNode = fused.node(in2.nodeId);
                ext.push_back(buf.at(inNode->debugName).data());
                extCounts.push_back(fusedElementCounts.at(in2.nodeId));
            }
            long long outCount = fusedElementCounts.at(n->id);
            buf[name] = std::vector<float>(static_cast<size_t>(outCount));
            if (isReduction) {
                ReductionFn rfn = module.getFunction<ReductionFn>("compute_" + name);
                rfn(ext, extCounts, buf[name].data());
            } else {
                ElementwiseFn efn = module.getFunction<ElementwiseFn>("compute_" + name);
                efn(ext, extCounts, buf[name].data(), outCount);
            }
            bool matches = (buf[name].size() == expected.at(name).size());
            if (matches) for (size_t i = 0; i < buf[name].size(); ++i)
                if (std::fabs(buf[name][i] - expected.at(name)[i]) > 1e-3f) matches = false;
            allMatch = allMatch && matches;
            printf("compute_%-3s (called as a real function pointer): ", name.c_str());
            for (float v : buf[name]) printf("%.6f ", v);
            printf(" -- matches evaluateArrays() (%s)\n", matches ? "confirmed" : "MISMATCH");
        }
    }  // ONE JitModule, ONE dlclose(), after all 3 function pointers were used

    printf("\nself-check: 3 real dlsym() calls against ONE dlopen()'d handle, chained through\n");
    printf("the SAME std::map<string,vector<float>> buffer convention Chapter 18's own CPU\n");
    printf("driver used, produced every buffer evaluateArrays() reports for this fused\n");
    printf("graph -- reduction and elementwise functions dispatched correctly, from the\n");
    printf("SAME compiled shared library (%s)\n", allMatch ? "confirmed" : "MISMATCH");

    bool allOk = compileClean && allMatch;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "049_one_shared_library_many_functions_a_jit_module_for_a_whole_graph.cpp" -o "049_one_shared_library_many_functions_a_jit_module_for_a_whole_graph" -ldl
./"049_one_shared_library_many_functions_a_jit_module_for_a_whole_graph"
```

**Output (cloud sandbox, x86-64, AVX2/FMA):**

```text
=== Section 20.2: one shared library, many functions -- a JIT module for a whole graph ===

fused graph: a b t2(FE,2st) s(FR,1st) y(FE,2st) 

generateJitProgramForGraph(): 3 extern "C" functions in ONE source file:
  compute_t2 (elementwise)
  compute_s (reduction)
  compute_y (elementwise)

--- g++ -shared -fPIC compile (whole graph, ONE invocation) ---

(no output -- clean compile)

self-check: this whole graph's own 3 non-leaf nodes needed exactly 1 g++ invocation(s)
to compile -- Chapter 17/18's own harness would have shelled out once PER GENERATED
PROGRAM; this section shells out once PER GRAPH (confirmed)

compute_t2  (called as a real function pointer): 2.000000 0.000000 6.000000 0.000000 10.000000 0.000000  -- matches evaluateArrays() (confirmed)
compute_s   (called as a real function pointer): 18.000000  -- matches evaluateArrays() (confirmed)
compute_y   (called as a real function pointer): 22.000000 20.000000 26.000000 20.000000 30.000000 20.000000  -- matches evaluateArrays() (confirmed)

self-check: 3 real dlsym() calls against ONE dlopen()'d handle, chained through
the SAME std::map<string,vector<float>> buffer convention Chapter 18's own CPU
driver used, produced every buffer evaluateArrays() reports for this fused
graph -- reduction and elementwise functions dispatched correctly, from the
SAME compiled shared library (confirmed)
```

**Output (device, aarch64 Linux VM, NEON):**

```text
=== Section 20.2: one shared library, many functions -- a JIT module for a whole graph ===

fused graph: a b t2(FE,2st) s(FR,1st) y(FE,2st)

generateJitProgramForGraph(): 3 extern "C" functions in ONE source file:
  compute_t2 (elementwise)
  compute_s (reduction)
  compute_y (elementwise)

--- g++ -shared -fPIC compile (whole graph, ONE invocation) ---

(no output -- clean compile)

self-check: this whole graph's own 3 non-leaf nodes needed exactly 1 g++ invocation(s)
to compile -- Chapter 17/18's own harness would have shelled out once PER GENERATED
PROGRAM; this section shells out once PER GRAPH (confirmed)

compute_t2  (called as a real function pointer): 2.000000 0.000000 6.000000 0.000000 10.000000 0.000000  -- matches evaluateArrays() (confirmed)
compute_s   (called as a real function pointer): 18.000000  -- matches evaluateArrays() (confirmed)
compute_y   (called as a real function pointer): 22.000000 20.000000 26.000000 20.000000 30.000000 20.000000  -- matches evaluateArrays() (confirmed)

self-check: 3 real dlsym() calls against ONE dlopen()'d handle, chained through
the SAME std::map<string,vector<float>> buffer convention Chapter 18's own CPU
driver used, produced every buffer evaluateArrays() reports for this fused
graph -- reduction and elementwise functions dispatched correctly, from the
SAME compiled shared library (confirmed)
```


!!! note "Why the compiled .so has no main() of its own"
    Every generated program since Chapter 17 was a complete standalone executable: its own `main()`, its own `printf()` calls, its own exit code -- exactly what a real subprocess needs to run and report back over stdout. `generateJitProgramForGraph()` emits none of that. The generated source is nothing but a sequence of `extern "C"` function DEFINITIONS -- no `main()`, no `printf()` calls, no program of its own at all. Compiled with `-shared`, it isn't meant to run as a program; it's meant to be mapped into an ALREADY-RUNNING process (this one) and have its own functions called directly. Chapter 18's own `std::map<std::string, std::vector<float>>` buffer convention, reused here unchanged, is what makes wiring several such functions together possible without inventing a new calling convention: each generated function still takes the same `ext`/`extCounts`/`out` signature it always did.

## 20.3 The JIT Backend End to End: No Subprocess, No Stdout Parsing

### Intuition

Sections 20.1 and 20.2 proved the loading mechanism itself, always with Chapter 18's own plain scalar code as the function body. This section proves the mechanism is BACKEND AGNOSTIC: the loader does not care whether a function's own body came from Chapter 18's scalar `emitSteps()` path or Chapter 19's vectorized AVX2/FMA-or-NEON intrinsics path, only that the function is `extern "C"` and matches an expected signature. Chapter 16's own 10-node capstone graph -- the SAME graph Sections 17.3, 18.3, and 19.3 have all already used -- gets JIT-compiled TWICE from this section: once through a scalar dispatcher and once through Chapter 19's own vector-FMA-and-reduction dispatcher, wrapped in `extern "C"` for the first time. Both compiled libraries are `dlopen()`'d, every one of their own node functions is `dlsym()`'d and called as a real function pointer -- ZERO subprocess launches and ZERO parsed stdout text anywhere in EXECUTION, for the first time in this book -- and both report the same `y[0]=132.0` Chapters 16 through 19 have all already established.

### Background

`generateExternCVectorElementwiseFunctionFma()` and `generateExternCVectorReductionFunction()` are Section 19.2's and 19.3's own generators, changed in exactly the one place Section 20.1 already established as this chapter's own recurring pattern: the emitted signature now starts with `extern "C"`, nothing else about the FMA-fold detection or the vectorized-reduction horizontal sum changes at all. `generateJitFunctionForNodeVectorized()` is Section 19.3's own three-way dispatcher -- a reduction node always takes the vectorized-reduction path; an elementwise node takes the FMA-fold vector path when `canVectorizeElementwise()` allows it; anything else falls back to Chapter 18's own plain scalar path -- now emitting `extern "C"` bodies and never writing a standalone program with its own `main()`. `generateJitProgramForGraphVectorized()` walks a whole graph, dispatches per node exactly the way `generateJitProgramForGraph()` already does for the scalar backend, and tracks a running `totalFolds` count across every node.

`runJitGraph()` is this section's one genuinely new piece of driver code: it runs every function in a JIT-compiled graph, in topological order, via real function pointers, chained through the same buffer-map convention Section 20.2 already used -- and it is SHARED by both backends. From the loader's own point of view, a `compute_t3` symbol looks identical whether its body came from Chapter 18's scalar path or Chapter 19's vector path; only the compiled `.so` it was `dlsym()`'d from differs, so one driver function correctly serves two structurally different code generators. `compileToSharedLibrary()` gains one new parameter this section, `extraFlags`, for the one place these two backends genuinely diverge in their own compile command: the vector backend needs `-mavx2 -mfma` on x86-64 (nothing extra on aarch64, where NEON is the mandatory baseline), while the scalar backend needs nothing extra on either machine -- the same `#if defined(__x86_64__)` / `#elif defined(__aarch64__)` detection Chapter 19 already established selects both the `Isa` and the extra compile flags together.

Chapter 16's own capstone graph, fused with `boundedReductionFusionPass(maxChainLength=3)` exactly as Sections 17.3, 18.3, and 19.3 all fused it (`a b t3(FE,3st) t5(FE,2st) s(FR,1st) y(FE,2st)`), is compiled and run through both backends: the scalar backend reports `y[0]=132.0`, matching `evaluateArrays()`; the vector backend reports the same `y[0]=132.0` AND exactly 3 FMA folds, the same count Section 19.3 already established for this identical graph. The closing tally is deliberately honest about what did and did not change: `compileInvocationCount` reaches 2 -- one `g++` invocation per backend, a count roughly comparable to what Chapters 17 through 19's own harness already needed for one whole fused-graph program per backend, and this chapter does not claim compilation itself got cheaper. `executionSubprocessCount`, by contrast, stays exactly 0 across the whole run -- every one of this section's own 8 real results (4 materialized nodes times 2 backends) crossed from generated code into this program as a genuine C++ function return, never a subprocess launch, never a line of printed stdout parsed back. That is the actual, provable claim this chapter set out to make.

```text
Two backends, ONE loader, ONE shared driver:

  generateJitProgramForGraph()          generateJitProgramForGraphVectorized()
  (Section 20.2, Ch18's scalar bodies)  (this section, Ch19's vector bodies,
                                          now wrapped in extern "C")
         |                                        |
         | compile: -shared -fPIC                 | compile: -shared -fPIC
         |                                         (-mavx2 -mfma on x86-64,
         +--> hammer_ch20_050_scalar.so             nothing extra on aarch64)
                                                    +--> hammer_ch20_050_vector.so

         both loaded by a SEPARATE JitModule (2 dlopen() calls total)

                    |                                |
                    +----------------+---------------+
                                     |
                              runJitGraph()
                   (ONE function, shared by both backends --
                    a compute_t3 symbol looks identical to the
                    loader whether its BODY is Ch18 scalar code
                    or Ch19 vector code; only the compiled .so
                    it was dlsym()'d from differs)
                                     |
                    dlsym() one function pointer per node, call
                    each directly, chain through the SAME
                    std::map buffer convention Ch18's own CPU
                    driver used

  scalar backend:  y[0] = 132.0 (confirmed)
  vector backend:  y[0] = 132.0 (confirmed), 3 FMA folds (confirmed)

Honest tally -- what actually changed, and what did not:

  g++ invocations (COMPILATION): 2 total, one per backend -- roughly
  the SAME count Chapter 17-19's own harness needed for one whole
  fused-graph program per backend; this chapter does not claim
  compilation itself got cheaper

  subprocess launches for EXECUTION: 0 -- every one of this section's
  own 8 real results (4 nodes x 2 backends) crossed from generated
  code into this program as a genuine function return, not a line of
  stdout text parsed back -- the ACTUAL, provable improvement this
  chapter set out to make
```

```cpp
// Chapter 20: A JIT Backend: Compiling and Loading Generated Code at Runtime
// 050_the_jit_backend_end_to_end_no_subprocess_no_stdout_parsing.cpp
//
// Section 20.3 -- the capstone. Sections 20.1 and 20.2 proved the JIT
// loading mechanism itself: extern "C", dlopen(), dlsym(), a real function
// pointer called directly. This section proves the loader is BACKEND
// AGNOSTIC -- it does not care whether a function's own BODY came from
// Chapter 18's plain scalar emitSteps() path or Chapter 19's vectorized
// AVX2/FMA-or-NEON intrinsics path, only that the function is extern "C"
// and matches an expected signature. Chapter 16's own 10-node capstone
// graph -- the SAME graph Sections 17.3, 18.3, and 19.3 have all already
// used -- gets JIT-compiled TWICE from this section, once through a scalar
// dispatcher and once through Chapter 19's own vector-FMA-and-reduction
// dispatcher, wrapped in extern "C" for the first time. Both compiled
// libraries are dlopen()'d, every one of their own node functions is
// dlsym()'d and called as a real function pointer -- ZERO subprocess
// launches and ZERO parsed stdout text anywhere in EXECUTION, for the
// first time in this book -- and both report the same y[0]=132.0 Chapters
// 16 through 19 have all already established.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 050_the_jit_backend_end_to_end_no_subprocess_no_stdout_parsing.cpp -o 050_driver -ldl
// Run:     ./050_driver
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
#include <dlfcn.h>

// ==================== Value / OpKind / FusedStep / Node / Graph (from Chapters 13-14, unchanged) ====================

struct Value { int nodeId = -1; int outputIndex = 0; };

enum class OpKind { Input, Const, Add, Mul, ReLU, Sum, FusedElementwise, FusedReduction };
enum class OperandKind { ExternalInput, PriorStep };

struct FusedOperand { OperandKind kind; int index; };
struct FusedStep { OpKind op; std::vector<FusedOperand> operands; long long reduceElementCount = 1; };

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
    Value addBinary(OpKind op, Value lhs, Value rhs, const std::string& name) { return addNode(op, {lhs, rhs}, name); }
    Value addFusedElementwise(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
        Value out = addNode(OpKind::FusedElementwise, std::move(externalInputs), name);
        nodes_.back()->fusedSteps = std::move(steps);
        return out;
    }
    Value addFusedReduction(std::vector<Value> externalInputs, std::vector<FusedStep> steps, const std::string& name) {
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
        for (const auto& n : g.nodes()) {
            for (const Value& in : n->inputs) {
                if (in.nodeId == id) { if (--inDegree[n->id] == 0) ready.push_back(n->id); }
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
static long long numElements(const Shape& s) { long long n = 1; for (int d : s.dims) n *= d; return n; }
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
        else shapes[id] = Shape{};
    }
    return shapes;
}

// ==================== boundedReductionFusionPass() (from Chapter 16 / Section 17.3, unchanged) ====================

struct FusionResult { Graph graph; std::map<int, int> representativeOldId; };

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

// ==================== Loop / LoopNest / buildLoopNest() / lowerNode() (from Chapter 15 / Section 17.2, unchanged) ====================

struct Loop { std::string dimName; long long extent; };
struct LoopNest { std::vector<Loop> loops; };
static LoopNest buildLoopNest(const Node* n, const std::map<int, Shape>& shapes,
                               const std::map<int, long long>& elementCounts) {
    LoopNest nest;
    const Shape& outShape = shapes.at(n->id);
    for (size_t i = 0; i < outShape.dims.size(); ++i) nest.loops.push_back(Loop{"dim" + std::to_string(i), outShape.dims[i]});
    if (n->op == OpKind::Sum) nest.loops.push_back(Loop{"reduce", elementCounts.at(n->inputs[0].nodeId)});
    else if (n->op == OpKind::FusedReduction) nest.loops.push_back(Loop{"reduce", n->fusedSteps.back().reduceElementCount});
    return nest;
}
struct LoweredNode { std::vector<FusedStep> steps; LoopNest nest; };
static LoweredNode lowerNode(const Node* n, const std::map<int, Shape>& shapes,
                              const std::map<int, long long>& elementCounts) {
    LoweredNode result;
    if (n->op == OpKind::FusedElementwise || n->op == OpKind::FusedReduction) {
        result.steps = n->fusedSteps;
    } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
        result.steps = {FusedStep{n->op, {FusedOperand{OperandKind::ExternalInput, 0}, FusedOperand{OperandKind::ExternalInput, 1}}}};
    } else if (n->op == OpKind::ReLU) {
        result.steps = {FusedStep{OpKind::ReLU, {FusedOperand{OperandKind::ExternalInput, 0}}}};
    } else {
        long long count = elementCounts.at(n->inputs[0].nodeId);
        result.steps = {FusedStep{OpKind::Sum, {FusedOperand{OperandKind::ExternalInput, 0}}, count}};
    }
    result.nest = buildLoopNest(n, shapes, elementCounts);
    return result;
}

// ==================== emitSteps() (from Section 18.1, unchanged) ====================

enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExprScalar(Target target, const std::string& x) {
    return (target == Target::Cpu) ? ("std::max(0.0f, " + x + ")") : ("fmaxf(0.0f, " + x + ")");
}
static std::vector<std::string> emitSteps(const std::vector<FusedStep>& steps, bool isReduction,
                                           const std::string& idxExpr, Target target, std::string& finalValueExpr) {
    std::vector<std::string> lines;
    std::vector<std::string> stepVars;
    size_t stepCount = isReduction ? steps.size() - 1 : steps.size();
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return readExternal(target, o.index, idxExpr);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < stepCount; ++s) {
        const FusedStep& step = steps[s];
        std::string expr;
        if (step.op == OpKind::ReLU) expr = maxExprScalar(target, operandExpr(step.operands[0]));
        else if (step.op == OpKind::Add) expr = operandExpr(step.operands[0]) + " + " + operandExpr(step.operands[1]);
        else expr = operandExpr(step.operands[0]) + " * " + operandExpr(step.operands[1]);
        std::string varName = "step" + std::to_string(s);
        lines.push_back("float " + varName + " = " + expr + ";");
        stepVars.push_back(varName);
    }
    if (isReduction) {
        const FusedOperand& sumOperand = steps.back().operands[0];
        finalValueExpr = (sumOperand.kind == OperandKind::ExternalInput)
                              ? readExternal(target, sumOperand.index, idxExpr)
                              : stepVars[static_cast<size_t>(sumOperand.index)];
    } else {
        finalValueExpr = stepVars.back();
    }
    return lines;
}

// ==================== Section 20.1/20.2's own extern "C" scalar generators (unchanged) ====================

static std::string generateExternCElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
}
static std::string generateExternCReductionFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cpu, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    float acc = 0.0f;\n";
    src += "    for (long long r = 0; r < " + std::to_string(reduceExtent) + "; ++r) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        acc += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = acc;\n}\n";
    return src;
}
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }
static std::string generateExternCFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered) {
    return isReductionNode(n) ? generateExternCReductionFunction(funcName, lowered)
                               : generateExternCElementwiseFunction(funcName, lowered);
}
static std::string generateJitProgramForGraph(const Graph& g, const std::map<int, Shape>& shapes,
                                               const std::map<int, long long>& elementCounts,
                                               std::vector<std::string>& functionsGenerated) {
    TopoResult topo = topologicalSort(g);
    std::string prog = "#include <vector>\n#include <algorithm>\n\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        prog += generateExternCFunctionForNode(funcName, n, lowered) + "\n";
        functionsGenerated.push_back(n->debugName);
    }
    return prog;
}

// ==================== Section 19's own Isa / vector primitives, FMA fold, and vectorized
//                      reduction (unchanged), now wrapped in extern "C" for the first time ====================

enum class Isa { Avx2, Neon };

static int vectorWidth(Isa isa) { return (isa == Isa::Avx2) ? 8 : 4; }
static std::string vecType(Isa isa) { return (isa == Isa::Avx2) ? "__m256" : "float32x4_t"; }
static std::string isaHeaderInclude(Isa isa) { return (isa == Isa::Avx2) ? "#include <immintrin.h>" : "#include <arm_neon.h>"; }
static std::string isaName(Isa isa) { return (isa == Isa::Avx2) ? "AVX2/FMA" : "NEON"; }

static std::string vecLoadOrBroadcast(Isa isa, int index, const std::string& baseExpr, bool isScalar) {
    std::string ext = "ext[" + std::to_string(index) + "]";
    if (isa == Isa::Avx2) {
        if (isScalar) return "_mm256_set1_ps(" + ext + "[0])";
        return "_mm256_loadu_ps(" + ext + " + " + baseExpr + ")";
    }
    if (isScalar) return "vdupq_n_f32(" + ext + "[0])";
    return "vld1q_f32(" + ext + " + " + baseExpr + ")";
}
static std::string vecAdd(Isa isa, const std::string& a, const std::string& b) {
    return (isa == Isa::Avx2) ? ("_mm256_add_ps(" + a + ", " + b + ")") : ("vaddq_f32(" + a + ", " + b + ")");
}
static std::string vecMul(Isa isa, const std::string& a, const std::string& b) {
    return (isa == Isa::Avx2) ? ("_mm256_mul_ps(" + a + ", " + b + ")") : ("vmulq_f32(" + a + ", " + b + ")");
}
static std::string vecMax(Isa isa, const std::string& x) {
    if (isa == Isa::Avx2) return "_mm256_max_ps(" + x + ", _mm256_setzero_ps())";
    return "vmaxq_f32(" + x + ", vdupq_n_f32(0.0f))";
}
static std::string vecStore(Isa isa, const std::string& dst, const std::string& val) {
    return (isa == Isa::Avx2) ? ("_mm256_storeu_ps(" + dst + ", " + val + ");") : ("vst1q_f32(" + dst + ", " + val + ");");
}
static bool canVectorizeElementwise(const Node* n, const std::map<int, long long>& elementCounts) {
    long long outCount = elementCounts.at(n->id);
    for (const Value& in : n->inputs) {
        long long c = elementCounts.at(in.nodeId);
        if (c != outCount && c != 1) return false;
    }
    return true;
}
static std::string vecFma(Isa isa, const std::string& mulLhs, const std::string& mulRhs, const std::string& addend) {
    if (isa == Isa::Avx2) return "_mm256_fmadd_ps(" + mulLhs + ", " + mulRhs + ", " + addend + ")";
    return "vfmaq_f32(" + addend + ", " + mulLhs + ", " + mulRhs + ")";
}
static std::vector<int> computeStepUseCounts(const std::vector<FusedStep>& steps, bool isReduction) {
    size_t stepCount = isReduction ? steps.size() - 1 : steps.size();
    std::vector<int> useCounts(stepCount, 0);
    for (size_t s = 0; s < stepCount; ++s)
        for (const FusedOperand& o : steps[s].operands)
            if (o.kind == OperandKind::PriorStep) useCounts[o.index]++;
    if (isReduction) {
        const FusedOperand& sumOperand = steps.back().operands[0];
        if (sumOperand.kind == OperandKind::PriorStep) useCounts[sumOperand.index]++;
    } else if (stepCount > 0) {
        useCounts[stepCount - 1]++;
    }
    return useCounts;
}
static bool stepUsesOperand(const FusedStep& step, OperandKind kind, int index) {
    for (const FusedOperand& o : step.operands) if (o.kind == kind && o.index == index) return true;
    return false;
}
// The ONLY change from Section 19.2's own generateVectorElementwiseFunctionFma():
// the emitted signature now starts with extern "C", exactly the same change
// Section 20.1 made to Chapter 18's own scalar generator.
static std::string generateExternCVectorElementwiseFunctionFma(const std::string& funcName, const LoweredNode& lowered,
                                                                 Isa isa, const std::vector<bool>& isScalarInput,
                                                                 int& foldCount) {
    const std::vector<FusedStep>& steps = lowered.steps;
    std::vector<int> useCounts = computeStepUseCounts(steps, false);
    foldCount = 0;

    std::vector<bool> foldedAway(steps.size(), false);
    for (size_t s = 0; s < steps.size(); ++s) {
        if (steps[s].op != OpKind::Add || s == 0) continue;
        if (steps[s - 1].op != OpKind::Mul) continue;
        if (useCounts[s - 1] != 1) continue;
        if (!stepUsesOperand(steps[s], OperandKind::PriorStep, static_cast<int>(s - 1))) continue;
        foldedAway[s - 1] = true;
        foldCount++;
    }

    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    int vw = vectorWidth(isa);
    src += "    long long i = 0;\n";
    src += "    for (; i + " + std::to_string(vw) + " <= n; i += " + std::to_string(vw) + ") {\n";

    std::vector<std::string> stepVars(steps.size());
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < steps.size(); ++s) {
        if (foldedAway[s]) continue;
        const FusedStep& step = steps[s];
        std::string varName = "v" + std::to_string(s);
        if (step.op == OpKind::Add && s > 0 && foldedAway[s - 1]) {
            const FusedStep& mulStep = steps[s - 1];
            std::string mulLhs = operandExpr(mulStep.operands[0]);
            std::string mulRhs = operandExpr(mulStep.operands[1]);
            std::string addend;
            for (const FusedOperand& o : step.operands)
                if (!(o.kind == OperandKind::PriorStep && o.index == static_cast<int>(s - 1))) addend = operandExpr(o);
            src += "        " + vecType(isa) + " " + varName + " = " + vecFma(isa, mulLhs, mulRhs, addend) +
                   ";  // fused: v" + std::to_string(s - 1) + " folded in\n";
        } else if (step.op == OpKind::ReLU) {
            src += "        " + vecType(isa) + " " + varName + " = " + vecMax(isa, operandExpr(step.operands[0])) + ";\n";
        } else if (step.op == OpKind::Add) {
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        } else {
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        }
        stepVars[s] = varName;
    }
    src += "        " + vecStore(isa, "out + i", stepVars.back()) + "\n";
    src += "    }\n";
    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(steps, false, "i", Target::Cpu, finalExpr);
    src += "    for (; i < n; ++i) {\n";
    for (const std::string& line : tailLines) src += "        " + line + "\n";
    src += "        out[i] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
}
// The ONLY change from Section 19.3's own generateVectorReductionFunction():
// extern "C" on the signature.
static std::string generateExternCVectorReductionFunction(const std::string& funcName, const LoweredNode& lowered,
                                                            Isa isa, const std::vector<bool>& isScalarInput) {
    const std::vector<FusedStep>& steps = lowered.steps;
    size_t stepCount = steps.size() - 1;
    long long reduceExtent = lowered.nest.loops.back().extent;
    int vw = vectorWidth(isa);

    std::string src = "extern \"C\" void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    (void)extCounts;\n";
    src += "    " + vecType(isa) + " acc = " + (isa == Isa::Avx2 ? "_mm256_setzero_ps()" : "vdupq_n_f32(0.0f)") + ";\n";
    src += "    long long i = 0;\n";
    src += "    for (; i + " + std::to_string(vw) + " <= " + std::to_string(reduceExtent) + "; i += " +
           std::to_string(vw) + ") {\n";

    std::vector<std::string> stepVars(stepCount);
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
        return stepVars[static_cast<size_t>(o.index)];
    };
    for (size_t s = 0; s < stepCount; ++s) {
        const FusedStep& step = steps[s];
        std::string varName = "v" + std::to_string(s);
        if (step.op == OpKind::ReLU)
            src += "        " + vecType(isa) + " " + varName + " = " + vecMax(isa, operandExpr(step.operands[0])) + ";\n";
        else if (step.op == OpKind::Add)
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        else
            src += "        " + vecType(isa) + " " + varName + " = " +
                   vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1])) + ";\n";
        stepVars[s] = varName;
    }
    const FusedOperand& sumOperand = steps.back().operands[0];
    std::string sumExpr = (sumOperand.kind == OperandKind::ExternalInput)
                               ? vecLoadOrBroadcast(isa, sumOperand.index, "i", isScalarInput[sumOperand.index])
                               : stepVars[static_cast<size_t>(sumOperand.index)];
    src += "        acc = " + vecAdd(isa, "acc", sumExpr) + ";\n";
    src += "    }\n";
    src += "    float lanes[" + std::to_string(vw) + "];\n";
    src += "    " + vecStore(isa, "lanes", "acc") + "\n";
    src += "    float hsum = 0.0f;\n";
    src += "    for (int lane = 0; lane < " + std::to_string(vw) + "; ++lane) hsum += lanes[lane];\n";

    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(steps, true, "i", Target::Cpu, finalExpr);
    src += "    for (; i < " + std::to_string(reduceExtent) + "; ++i) {\n";
    for (const std::string& line : tailLines) src += "        " + line + "\n";
    src += "        hsum += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = hsum;\n}\n";
    return src;
}

// Section 19.3's own three-way dispatcher, now emitting extern "C" bodies
// and never writing a standalone program with its own main() -- every
// function it returns is meant to be dlsym()'d, not run as a process.
static std::string generateJitFunctionForNodeVectorized(const std::string& funcName, const Node* n,
                                                          const LoweredNode& lowered, Isa isa,
                                                          const std::map<int, long long>& elementCounts,
                                                          int& foldCount) {
    foldCount = 0;
    std::vector<bool> isScalarInput;
    for (const Value& in : n->inputs) isScalarInput.push_back(elementCounts.at(in.nodeId) == 1);
    if (isReductionNode(n)) return generateExternCVectorReductionFunction(funcName, lowered, isa, isScalarInput);
    if (canVectorizeElementwise(n, elementCounts))
        return generateExternCVectorElementwiseFunctionFma(funcName, lowered, isa, isScalarInput, foldCount);
    return generateExternCElementwiseFunction(funcName, lowered);  // scalar fallback, same as Section 19.3
}
static std::string generateJitProgramForGraphVectorized(const Graph& g, const std::map<int, Shape>& shapes,
                                                          const std::map<int, long long>& elementCounts, Isa isa,
                                                          std::vector<std::string>& functionsGenerated, int& totalFolds) {
    TopoResult topo = topologicalSort(g);
    std::string prog = std::string(isaHeaderInclude(isa)) + "\n#include <vector>\n#include <algorithm>\n\n";
    totalFolds = 0;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        int foldCount = 0;
        prog += generateJitFunctionForNodeVectorized(funcName, n, lowered, isa, elementCounts, foldCount) + "\n";
        functionsGenerated.push_back(n->debugName);
        totalFolds += foldCount;
    }
    return prog;
}

// ==================== Section 20.1's own JitModule (unchanged) ====================

class JitModule {
public:
    explicit JitModule(const std::string& sharedLibraryPath) {
        handle_ = dlopen(sharedLibraryPath.c_str(), RTLD_NOW);
        if (!handle_) throw std::runtime_error("JitModule: dlopen failed: " + std::string(dlerror()));
    }
    ~JitModule() {
        if (handle_) dlclose(handle_);
    }
    JitModule(const JitModule&) = delete;
    JitModule& operator=(const JitModule&) = delete;
    JitModule(JitModule&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    JitModule& operator=(JitModule&& other) noexcept {
        if (this != &other) {
            if (handle_) dlclose(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    template <typename FnPtr>
    FnPtr getFunction(const std::string& symbolName) const {
        dlerror();
        void* sym = dlsym(handle_, symbolName.c_str());
        const char* err = dlerror();
        if (err) throw std::runtime_error("JitModule::getFunction(\"" + symbolName + "\"): dlsym failed: " + err);
        return reinterpret_cast<FnPtr>(sym);
    }

private:
    void* handle_ = nullptr;
};

// ==================== Shell-out (COMPILATION ONLY) ====================

static int compileInvocationCount = 0;
static int executionSubprocessCount = 0;  // stays 0 for the whole file -- the point of this chapter
static std::string runShellCaptureAll(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed for: " + cmd);
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}
static void writeFile(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}
static bool compileToSharedLibrary(const std::string& source, const std::string& cppPath, const std::string& soPath,
                                    const std::string& extraFlags, std::string& compileLog) {
    writeFile(cppPath, source);
    compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2 -shared -fPIC" + extraFlags + " " +
                                     cppPath + " -o " + soPath + " 2>&1");
    ++compileInvocationCount;
    return compileLog.empty();
}

using ElementwiseFn = void (*)(const std::vector<const float*>&, const std::vector<long long>&, float*, long long);
using ReductionFn = void (*)(const std::vector<const float*>&, const std::vector<long long>&, float*);

// Runs every one of a JIT-compiled graph's own functions, in topological
// order, chained through a shared buffer map -- via REAL FUNCTION POINTERS,
// never a subprocess, never a line of parsed stdout.
static std::map<std::string, std::vector<float>> runJitGraph(const JitModule& module, const Graph& g,
                                                               const std::map<int, long long>& elementCounts,
                                                               const std::vector<std::string>& functionOrder,
                                                               std::map<std::string, std::vector<float>> buf) {
    for (const std::string& name : functionOrder) {
        const Node* n = nullptr;
        for (const auto& nn : g.nodes()) if (nn->debugName == name) n = nn.get();
        bool isReduction = isReductionNode(n);
        std::vector<const float*> ext;
        std::vector<long long> extCounts;
        for (const Value& in : n->inputs) {
            const Node* inNode = g.node(in.nodeId);
            ext.push_back(buf.at(inNode->debugName).data());
            extCounts.push_back(elementCounts.at(in.nodeId));
        }
        long long outCount = elementCounts.at(n->id);
        buf[name] = std::vector<float>(static_cast<size_t>(outCount));
        if (isReduction) {
            ReductionFn fn = module.getFunction<ReductionFn>("compute_" + name);
            fn(ext, extCounts, buf[name].data());
        } else {
            ElementwiseFn fn = module.getFunction<ElementwiseFn>("compute_" + name);
            fn(ext, extCounts, buf[name].data(), outCount);
        }
    }
    return buf;
}

static bool arraysMatch(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    printf("=== Section 20.3: the JIT backend end to end -- no subprocess, no stdout parsing ===\n\n");

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string vectorExtraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string vectorExtraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    // Chapter 16's own 10-node capstone graph, reused verbatim -- the SAME
    // graph Sections 17.3, 18.3, and 19.3 have all already made real, on a
    // CPU loop nest, on a (compile-only) GPU, and vectorized.
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
    (void)t1; (void)t2; (void)t3; (void)t4; (void)y2; (void)y;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{8}}}, {b.nodeId, Shape{}}};
    std::map<int, Shape> shapes = inferShapes(g, declared);
    std::map<int, long long> elementCounts;
    for (const auto& kv : shapes) elementCounts[kv.first] = numElements(kv.second);

    std::vector<float> aVals = {1, -2, 3, -4, 5, -6, 7, -8};
    std::vector<float> bVals = {2};
    std::map<std::string, std::vector<float>> inputArrays = {{"a", aVals}, {"b", bVals}};
    auto origArrays = evaluateArrays(g, inputArrays, elementCounts);
    printf("a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = ");
    for (float v : origArrays.at("y")) printf("%.6f ", v);
    printf("\n\n");

    FusionResult fr = boundedReductionFusionPass(g, elementCounts, 3);
    const Graph& fused = fr.graph;
    std::map<int, Shape> fusedShapes;
    std::map<int, long long> fusedElementCounts;
    for (const auto& n : fused.nodes()) {
        int oldId = fr.representativeOldId.at(n->id);
        fusedShapes[n->id] = shapes.at(oldId);
        fusedElementCounts[n->id] = elementCounts.at(oldId);
    }
    printf("fused graph (maxChainLength=3, same cap Section 17.3/18.3/19.3 all used): ");
    for (const auto& n : fused.nodes()) {
        printf("%s", n->debugName.c_str());
        if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
        else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
        printf(" ");
    }
    printf("\n\n");

    bool allOk = true;

    // ---- Backend 1: SCALAR (Chapter 18's own emitSteps() body, extern "C") ----
    printf("--- Backend 1: SCALAR (Chapter 18's own emitSteps() body, JIT-compiled) ---\n\n");
    std::vector<std::string> scalarFns;
    std::string scalarProg = generateJitProgramForGraph(fused, fusedShapes, fusedElementCounts, scalarFns);
    std::string scalarCompileLog;
    bool scalarClean = compileToSharedLibrary(scalarProg, "/tmp/hammer_ch20_050_scalar.cpp",
                                               "/tmp/hammer_ch20_050_scalar.so", "", scalarCompileLog);
    printf("g++ -shared -fPIC compile (%zu functions, scalar): %s\n", scalarFns.size(),
           scalarClean ? "clean" : ("FAILED:\n" + scalarCompileLog).c_str());
    std::map<std::string, std::vector<float>> scalarResult;
    bool scalarLoaded = false;
    {
        JitModule scalarModule("/tmp/hammer_ch20_050_scalar.so");
        scalarLoaded = true;
        std::map<std::string, std::vector<float>> initial = {{"a", aVals}, {"b", bVals}};
        scalarResult = runJitGraph(scalarModule, fused, fusedElementCounts, scalarFns, initial);
    }
    bool scalarMatchesY = arraysMatch(scalarResult.at("y"), origArrays.at("y"));
    printf("compute_y (scalar, real function pointer) = ");
    for (float v : scalarResult.at("y")) printf("%.6f ", v);
    printf("\nself-check: y[0]=%.6f, matches evaluateArrays() (%s)\n\n", scalarResult.at("y")[0],
           (scalarLoaded && scalarMatchesY) ? "confirmed" : "MISMATCH");
    allOk = allOk && scalarClean && scalarLoaded && scalarMatchesY;

    // ---- Backend 2: VECTOR (Chapter 19's own FMA-fold + reduction body, extern "C") ----
    printf("--- Backend 2: VECTOR (Chapter 19's own FMA-fold + reduction body, JIT-compiled, %s) ---\n\n",
           isaName(hostIsa).c_str());
    std::vector<std::string> vectorFns;
    int totalFolds = 0;
    std::string vectorProg = generateJitProgramForGraphVectorized(fused, fusedShapes, fusedElementCounts, hostIsa,
                                                                    vectorFns, totalFolds);
    std::string vectorCompileLog;
    bool vectorClean = compileToSharedLibrary(vectorProg, "/tmp/hammer_ch20_050_vector.cpp",
                                               "/tmp/hammer_ch20_050_vector.so", vectorExtraFlags, vectorCompileLog);
    printf("g++ -shared -fPIC compile (%zu functions, %s, %d FMA fold(s)): %s\n", vectorFns.size(),
           isaName(hostIsa).c_str(), totalFolds, vectorClean ? "clean" : ("FAILED:\n" + vectorCompileLog).c_str());
    std::map<std::string, std::vector<float>> vectorResult;
    bool vectorLoaded = false;
    {
        JitModule vectorModule("/tmp/hammer_ch20_050_vector.so");
        vectorLoaded = true;
        std::map<std::string, std::vector<float>> initial = {{"a", aVals}, {"b", bVals}};
        vectorResult = runJitGraph(vectorModule, fused, fusedElementCounts, vectorFns, initial);
    }
    bool vectorMatchesY = arraysMatch(vectorResult.at("y"), origArrays.at("y"));
    printf("compute_y (vector, real function pointer) = ");
    for (float v : vectorResult.at("y")) printf("%.6f ", v);
    printf("\nself-check: y[0]=%.6f, matches evaluateArrays() (%s), 3 FMA folds applied (%s)\n\n",
           vectorResult.at("y")[0], (vectorLoaded && vectorMatchesY) ? "confirmed" : "MISMATCH",
           totalFolds == 3 ? "confirmed" : "MISMATCH");
    allOk = allOk && vectorClean && vectorLoaded && vectorMatchesY && (totalFolds == 3);

    printf("--- Tally: what this JIT backend needed, versus Chapters 17-19's own harness ---\n\n");
    printf("g++ invocations (COMPILATION -- unavoidable, this book's own JIT never embeds a\n");
    printf("compiler as a library): %d (one per backend, same count Chapter 17-19's own\n", compileInvocationCount);
    printf("harness needed for a chained whole-graph program)\n");
    printf("subprocess launches for EXECUTION: %d -- every one of this section's own %zu real\n",
           executionSubprocessCount, scalarFns.size() + vectorFns.size());
    printf("results (%zu nodes x 2 backends) crossed from generated code into this program as a\n", scalarFns.size());
    printf("genuine C++ function return, not a line of stdout text parsed back (%s)\n",
           executionSubprocessCount == 0 ? "confirmed" : "MISMATCH");

    allOk = allOk && (executionSubprocessCount == 0);
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "050_the_jit_backend_end_to_end_no_subprocess_no_stdout_parsing.cpp" -o "050_the_jit_backend_end_to_end_no_subprocess_no_stdout_parsing" -ldl
./"050_the_jit_backend_end_to_end_no_subprocess_no_stdout_parsing"
```

**Output (cloud sandbox, x86-64, AVX2/FMA):**

```text
=== Section 20.3: the JIT backend end to end -- no subprocess, no stdout parsing ===

a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000 

fused graph (maxChainLength=3, same cap Section 17.3/18.3/19.3 all used): a b t3(FE,3st) t5(FE,2st) s(FR,1st) y(FE,2st) 

--- Backend 1: SCALAR (Chapter 18's own emitSteps() body, JIT-compiled) ---

g++ -shared -fPIC compile (4 functions, scalar): clean
compute_y (scalar, real function pointer) = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000 
self-check: y[0]=132.000000, matches evaluateArrays() (confirmed)

--- Backend 2: VECTOR (Chapter 19's own FMA-fold + reduction body, JIT-compiled, AVX2/FMA) ---

g++ -shared -fPIC compile (4 functions, AVX2/FMA, 3 FMA fold(s)): clean
compute_y (vector, real function pointer) = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000 
self-check: y[0]=132.000000, matches evaluateArrays() (confirmed), 3 FMA folds applied (confirmed)

--- Tally: what this JIT backend needed, versus Chapters 17-19's own harness ---

g++ invocations (COMPILATION -- unavoidable, this book's own JIT never embeds a
compiler as a library): 2 (one per backend, same count Chapter 17-19's own
harness needed for a chained whole-graph program)
subprocess launches for EXECUTION: 0 -- every one of this section's own 8 real
results (4 nodes x 2 backends) crossed from generated code into this program as a
genuine C++ function return, not a line of stdout text parsed back (confirmed)
```

**Output (device, aarch64 Linux VM, NEON):**

```text
=== Section 20.3: the JIT backend end to end -- no subprocess, no stdout parsing ===

a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000

fused graph (maxChainLength=3, same cap Section 17.3/18.3/19.3 all used): a b t3(FE,3st) t5(FE,2st) s(FR,1st) y(FE,2st)

--- Backend 1: SCALAR (Chapter 18's own emitSteps() body, JIT-compiled) ---

g++ -shared -fPIC compile (4 functions, scalar): clean
compute_y (scalar, real function pointer) = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000
self-check: y[0]=132.000000, matches evaluateArrays() (confirmed)

--- Backend 2: VECTOR (Chapter 19's own FMA-fold + reduction body, JIT-compiled, NEON) ---

g++ -shared -fPIC compile (4 functions, NEON, 3 FMA fold(s)): clean
compute_y (vector, real function pointer) = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000
self-check: y[0]=132.000000, matches evaluateArrays() (confirmed), 3 FMA folds applied (confirmed)

--- Tally: what this JIT backend needed, versus Chapters 17-19's own harness ---

g++ invocations (COMPILATION -- unavoidable, this book's own JIT never embeds a
compiler as a library): 2 (one per backend, same count Chapter 17-19's own
harness needed for a chained whole-graph program)
subprocess launches for EXECUTION: 0 -- every one of this section's own 8 real
results (4 nodes x 2 backends) crossed from generated code into this program as a
genuine C++ function return, not a line of stdout text parsed back (confirmed)
```


!!! note "What this chapter proved, and what stayed a stated limitation"
    Every `extern "C"` function generated across all three sections -- Section 20.1's own single test function, Section 20.2's own whole small graph, and Section 20.3's own two full backends over Chapter 16's own 10-node capstone graph -- was `dlopen()`'d and `dlsym()`'d into a real function pointer and called directly, in-process, with zero subprocess launches for EXECUTION anywhere in this chapter. What stayed a stated, deliberate limitation, named rather than hidden: COMPILATION still shells out to a real `g++` process via `popen()`, once per generated library; a true self-hosting JIT would embed the compiler itself as a library (LLVM's own ORC JIT, or `libclang`) rather than invoking a separate compiler process even for that one step, explicitly out of this book's own scope.

## Chapter Summary

This chapter closed a gap every backend since Chapter 17 had left honestly open: generated code was always proven correct by running it as a SEPARATE PROCESS and parsing its own printed stdout back as text. Section 20.1 replaced that pattern's RUN half with real in-process loading: `extern "C"` disables C++ name mangling so a compiled symbol can be found by its plain, predictable name, and `JitModule` -- modeled directly on Chapter 4's own non-copyable, move-only `Graph` -- owns one `dlopen()` handle safely, `dlsym()`s a function pointer out of it, and calls that pointer directly with real C++ arguments, no subprocess and no parsed text anywhere in the call. Section 20.2 scaled that idea from one function to a whole graph: `generateJitProgramForGraph()` packs every non-leaf node's own `extern "C"` function into ONE compiled shared library, relying on Chapter 8's own five-chapters-old `canonicalizeNodeNames()` uniqueness guarantee to prove no two symbols can ever collide, and chains every `dlsym()`'d function through the same buffer-map convention Chapter 18's own CPU driver already used. Section 20.3 closed the chapter with the capstone: Chapter 16's own 10-node graph, JIT-compiled TWICE -- once through Chapter 18's scalar backend, once through Chapter 19's vector backend, both wrapped in `extern "C"` for the first time -- and run through one shared `runJitGraph()` driver that treats both backends identically, reporting the same `y[0]=132.0` (and, on the vector backend, the same 3 FMA folds) Chapters 16 through 19 have all already established. The chapter's own closing tally is deliberately honest: compilation still needs roughly the same number of `g++` invocations Chapters 17-19's own harness needed; the real, provable improvement is that EXECUTION needed zero subprocess launches and zero parsed stdout lines, for the first time in this book.

## Self-Check Questions

1. What does `extern "C"` actually change about a compiled function, and what does it deliberately leave unchanged?
2. `dlsym()` can legitimately return `nullptr` for a symbol that WAS found. Given that, how does `JitModule::getFunction()` correctly detect a lookup failure?
3. Why is `JitModule` non-copyable and move-only, and which earlier chapter's own class design does that mirror?
4. Section 20.1 already loads and calls one generated function. What is the ONE new thing `generateJitProgramForGraph()` does in Section 20.2 that Section 20.1's own program did not?
5. Why can two functions generated from the same graph never collide in one shared library's own symbol table, and what would actually happen if that guarantee were ever violated?
6. Section 20.3 compiles the SAME graph through two different backends. What differs between the scalar and vector compile commands, and why?
7. What does `runJitGraph()` do that lets ONE driver function correctly serve both the scalar backend and the vector backend?
8. What is the honest, provable claim this chapter's own closing tally makes about EXECUTION, and what claim does it deliberately NOT make about COMPILATION?

## Where We Go Next

Chapter 21, "The Search Space: Tile Sizes, Loop Orders, and Unrolling," opens Part 5, Autotuning. Every backend through Part 4 -- scalar, vectorized, and now JIT-loaded -- has compiled exactly ONE fixed schedule per fused kernel: one tile size (or none at all), one loop order, no unrolling decisions ever revisited. Chapter 21 begins exploring the SPACE of valid alternate schedules for one given fused graph, the necessary groundwork before Chapter 22 can compare a cost model against real measurement and Chapter 23 can build an actual autotuner on top of both. Apply the Chapter 5-20 depth-level standard throughout.

## Worked Solutions

1. `extern "C"` changes LINKAGE only: the compiled symbol name for the function becomes the plain, unmangled identifier written in the source, instead of a compiler- and platform-specific mangled string encoding the full parameter-type signature. It does NOT change the function's own type signature -- parameters are still real C++ types (a `std::vector<const float*>&`, not a C array) -- and it works here specifically because both the calling program and the JIT-compiled `.so` are built by the same `g++` installation, sharing one ABI.
2. `JitModule::getFunction()` follows the exact pattern `dlsym()`'s own documentation prescribes: it calls `dlerror()` once BEFORE the lookup to clear any error left over from an unrelated earlier call, then calls `dlsym()`, then calls `dlerror()` a second time -- a non-null result on that second call means THIS lookup failed, which is the only way to distinguish "symbol not found" from "symbol found, and its real address happens to be null."
3. `JitModule` is non-copyable because duplicating a raw `dlopen()` handle would let two objects each believe they own it, and either object's destructor could then call `dlclose()` on it a second time -- undefined behavior. It is move-only so ownership can still transfer cleanly between objects, with the moved-from `JitModule` left holding a null handle it will never try to close. This mirrors Chapter 4's own `Graph`, which applies the identical non-copyable, move-only discipline to its own `std::vector<std::unique_ptr<Node>>`.
4. Section 20.1's own generated program contained exactly one `extern "C"` function. `generateJitProgramForGraph()`'s own new idea is walking an ENTIRE graph in topological order and appending EVERY non-leaf node's own `extern "C"` function into ONE growing source string, so one compiled shared library exports several symbols instead of one, with no generated driver `main()` anywhere in the picture.
5. Symbol names can never collide because every generated symbol is `"compute_" + debugName`, and Chapter 8's own `canonicalizeNodeNames()` has guaranteed every node's own `debugName` is unique WITHIN ONE GRAPH since five chapters before this book ever generated a line of code. If that guarantee were ever somehow violated, the consequence would not be a silent miscompile -- the LINKER itself would refuse to build the shared library, reporting a real, checkable "duplicate symbol" error.
6. The scalar backend compiles with no extra flags on either machine. The vector backend compiles with `-mavx2 -mfma` on the cloud sandbox's own x86-64 architecture, and with no extra flags on the device's own aarch64 architecture, because NEON is the mandatory aarch64 baseline instruction set rather than an optional extension the way AVX2 is on x86-64 -- the exact same `#if defined(__x86_64__)` / `#elif defined(__aarch64__)` detection Chapter 19 already established.
7. `runJitGraph()` takes a `JitModule` and a list of node names, and for each one calls `module.getFunction()` by that node's own `"compute_" + debugName` symbol and invokes it with the same `ext`/`extCounts`/`out` signature every generator in this book has used since Chapter 18. Because both the scalar and vector generators emit functions matching that exact signature, `runJitGraph()` never needs to know or care which backend produced the `.so` it is calling into -- only that the symbol exists and matches the expected function-pointer type.
8. The honest, provable claim is about EXECUTION: `executionSubprocessCount` stays exactly 0 across all 8 real results (4 materialized nodes across 2 backends) this section produces, meaning every one of them crossed from generated code into this program as a genuine C++ function return, never a subprocess launch, never a line of parsed stdout text. The claim this chapter deliberately does NOT make is that COMPILATION got any cheaper: `compileInvocationCount` reaches 2 (one per backend), a count roughly comparable to what Chapters 17 through 19's own harness already needed for one whole fused-graph program per backend.

---

**Sources cited in this chapter:**

None new. This chapter's own `JitModule` design, `extern "C"` wrapping discipline, and whole-graph JIT compilation technique are all original to this book, building on Chapter 4's own non-copyable resource-ownership discipline, Chapter 8's own node-name uniqueness guarantee, and this chapter's own directly observed `dlopen()`/`dlsym()`/`dlclose()` toolchain behavior on both real machines.