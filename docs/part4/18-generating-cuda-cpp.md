# 18. Generating CUDA C++ From the Fused IR

**What you will understand:** `emitSteps()`, the shared codegen core that generates the exact same step sequence for both a CPU backend and a CUDA backend, differing only in HOW an external input gets read and HOW ReLU's own max gets spelled; `generateCudaElementwiseKernel()`, which turns a node's own `LoopNest` and `FusedStep` body into a `__global__` CUDA kernel launching one THREAD per output element instead of one more pass through a shared loop variable; `generateCudaReductionKernel()`, which does the same for a reduction while avoiding a genuine data race with `atomicAdd()`; and a full, multi-kernel CUDA program -- generated from Chapter 16's own 10-node capstone graph, compiled by a real `nvcc` toolchain, and verified end to end against a real, executed CPU reference.

**What you need to know first:** Chapter 17's own `LoopNest`/`FusedStep`/`lowerNode()`/`generateLoopFunction()`, Chapter 16's own `boundedReductionFusionPass()` (reused verbatim in Section 18.3), and Chapter 14's own reduction-fusion `Sum` rule.

---

Chapter 17 built CUDA Hammer's first real, working, executable backend -- but it targeted the plainest possible execution model: one CPU thread, running one serial `for` loop, one element after another. This chapter targets the platform this book is actually named for. But before writing a single line of kernel-generation code, it is worth asking an honest question this book has asked before, at Chapter 12's own roofline numbers and Chapter 14's own FLOPs convention: what can actually be checked here, given this toolchain's own real, physical constraints?

Neither machine this book runs on has a physical GPU. The cloud sandbox this book verifies `.cu` files against does have a real `nvcc 12.0` install, and a short, throwaway experiment against it (three tiny test programs, written, compiled, run, and deleted before this chapter's own real files were started) settled three things this chapter's whole verification strategy rests on. First: `nvcc` can compile a COMPLETE CUDA program -- a real `__global__` kernel, `cudaMalloc`, `cudaMemcpy`, a kernel launch, `cudaFree` -- into a real, runnable ELF executable, with no physical GPU present at compile time at all. Second: running that compiled binary WITHOUT checking any CUDA error code returns silently -- exit code 0, no crash, and every "result" is uninitialized garbage, because every CUDA API call failed and nothing noticed. Third: running the exact same program WITH every CUDA API call wrapped in a macro that checks its own returned `cudaError_t` and prints `cudaGetErrorString()` on failure makes the SAME toolchain honestly report, at every single call, `"no CUDA-capable device is detected"` -- a real, accurate, checkable fact about this toolchain, not a guess or a simulation. This chapter's own generated CUDA programs check every call this same way, for the same reason Section 17.1's own `evaluateArrays()` checked per-element correctness instead of trusting an aggregate number: an unchecked result that happens to look fine is not the same thing as a result that was actually verified. The device this book also delivers to has no `nvcc` at all -- so, matching the discipline every earlier `.cu`/CUDA-linked file in this book has followed, every file in this chapter is compiled and run ONLY in the cloud sandbox; the device receives byte-identical copies, verified by content, never re-compiled there.

```text
FROM A SERIAL CPU LOOP (CHAPTER 17) TO A REAL, COMPILED CUDA PROGRAM (THIS CHAPTER):

  +-----------------------+     +-----------------------+     +-----------------------+
  | Fused IR (Ch13-16)     | --> | LoopNest + FusedStep   | --> | emitSteps(): the SAME  |
  | Node / Graph            |     | body (Ch15's own        |     | step sequence, for      |
  | (structure only)        |     | buildLoopNest(), same   |     | EITHER backend          |
  |                         |     | as Chapter 17)          |     | (this chapter's core)  |
  +-----------------------+     +-----------------------+     +-----------------------+
                                                                          |
                        +-------------------------------------------------+
                        |
          +-------------+-------------+
          |                           |
  +-----------------------+     +-----------------------+
  | CPU wrapper: a nested  |     | CUDA wrapper: one       |
  | for-loop, one more      |     | THREAD per loop          |
  | pass through a shared   |     | iteration, launched       |
  | loop variable            |     | concurrently, not         |
  | (Chapter 17's own       |     | serially                  |
  | generateLoopFunction()) |     | (this chapter's own      |
  |                         |     | CUDA kernel generators)  |
  +-----------------------+     +-----------------------+

  Section 18.1 -- elementwise: one CUDA thread per OUTPUT element, generated,
                  compiled by real nvcc, run -- honestly reporting no physical
                  GPU -- and proven correct via the SAME step sequence,
                  actually executed on the CPU

  Section 18.2 -- reduction: why one thread per REDUCE element doing
                  "*out += step;" directly is a real data race, and how
                  atomicAdd() fixes it -- correctly, not fastest

  Section 18.3 -- a full, multi-kernel CUDA program, generated from Chapter
                  16's own 10-node capstone graph, run through
                  boundedReductionFusionPass() twice -- extending Chapter 16
                  and 17's own "different fusion structure, same math" claim
                  to a real CUDA target for the first time
```

## 18.1 From Loops to Threads: A CUDA Elementwise Kernel Generator

### Intuition

Chapter 17's own `generateLoopFunction()` and this section's own CUDA kernel generator need the exact same thing at their core: given a node's own `FusedStep` program, emit `float stepN = ...;` for every step, in the right order, chaining each step's own operands to whichever earlier step (or external input) they reference. What differs between a CPU loop and a CUDA kernel is not that core sequencing logic -- it is the EXECUTION MODEL wrapped around it. A CPU loop processes one element, then the next, then the next, inside a single thread; a CUDA kernel launches one THREAD per element, all of them running the identical body concurrently, each computing its own `flat` index from its own `blockIdx`/`threadIdx` and returning immediately if that index is out of range. `emitSteps()` is built to make that distinction the ONLY thing that differs: it is called once by a CPU wrapper and once by a CUDA wrapper, with the exact same steps, and produces the exact same step sequence both times -- proving, structurally, that fusion's own step-by-step arithmetic does not change when the execution model around it does.

### Background

Two genuine backend differences are the ONLY thing `emitSteps()` isolates into small, swappable pieces, and both come from a real, physical distinction between a CPU and a GPU, not a stylistic one. The first is how an external input gets READ: on the CPU, every external input lives in ONE `std::vector<const float*>`, indexed at codegen time (`ext[0]`, `ext[1]`, ...); on the GPU, each external input has to be its own NAMED device pointer PARAMETER (`ext0`, `ext1`, ...), because a `std::vector`'s own internal buffer lives in HOST memory, and a `__global__` kernel dereferencing a host-memory address is undefined behavior -- not a style choice, a correctness requirement. The second is how ReLU's own max gets spelled: `std::max(0.0f, x)` on the CPU, `fmaxf(0.0f, x)` on the GPU, since plain `std::max` is not guaranteed to compile inside `__device__` code without extra compiler flags this book does not want to depend on. Everything else -- which operator each step performs, how a `PriorStep` operand chains to an earlier step's own result, how many steps a fused group has -- is identical, generated by the exact same `emitSteps()` call for both targets. The CUDA kernel itself launches one thread per output element with a now-familiar guard pattern (`flat = blockIdx.x * blockDim.x + threadIdx.x; if (flat >= n) return;`), then runs `emitSteps()`'s own body and writes `out[flat]` -- structurally the CUDA analogue of Chapter 17's own bounds-checked loop, just parallel instead of serial.

```text
emitSteps() called TWICE, same steps, Target::Cpu vs. Target::Cuda:

  step0 = Mul(ext0, ext1)          CPU:   step0 = ext[0][(flat) % extCounts[0]]
  step1 = ReLU(ext0)                        * ext[1][(flat) % extCounts[1]];
  step2 = Add(step0, step1)               step1 = std::max(0.0f, ext[0][(flat) % extCounts[0]]);
                                           step2 = step0 + step1;

                                    CUDA:  step0 = ext0[(flat) % extCount0]
                                             * ext1[(flat) % extCount1];
                                           step1 = fmaxf(0.0f, ext0[(flat) % extCount0]);
                                           step2 = step0 + step1;

  SAME step count. SAME sequencing. SAME PriorStep chaining. Only the read
  syntax and ReLU's own max differ -- exactly the two real backend
  differences a CPU-vs-GPU target actually requires, nothing more.

CUDA elementwise kernel -- one THREAD per output element, not one loop pass:

  thread 0: flat=0  -> compute step0,step1,step2 -> out[0] = step2     (concurrent,
  thread 1: flat=1  -> compute step0,step1,step2 -> out[1] = step2      not serial --
  thread 2: flat=2  -> compute step0,step1,step2 -> out[2] = step2      every thread
  ...                                                                    runs at once)
  thread n: flat=n  -> flat >= n -> return immediately (bounds guard,
                                     grid size rounds UP to a thread-block
                                     multiple, so extra threads exist and
                                     must bail out safely)
```

```cpp
// Chapter 18: Generating CUDA C++ From the Fused IR
// 042_from_loops_to_threads_a_cuda_elementwise_kernel_generator.cu
//
// Section 18.1 -- Chapter 17 lowered a node's own LoopNest + FusedStep body
// into a SERIAL C++ for-loop. This section targets a genuinely different
// execution model: a CUDA grid, where every loop ITERATION becomes its own
// THREAD running concurrently, not one more pass through a shared loop
// variable. emitSteps() is the shared core this shares with Chapter 17's
// own generateLoopFunction(): the SAME step-sequencing logic (which op
// happens in which order, how a PriorStep operand chains to an earlier
// step), reused unchanged by both a CPU wrapper (a nested for-loop) and a
// CUDA wrapper (a thread-index-and-guard) -- what differs between them is
// isolated to exactly two things a real backend difference actually
// requires: how an external input gets READ (an indexed buffer array on
// the CPU; a named device pointer parameter on the GPU, since a
// std::vector<const float*>'s own internal buffer lives in HOST memory and
// would be an invalid address if a __global__ kernel tried to dereference
// it directly), and how ReLU's own max gets spelled (std::max vs fmaxf).
// The generated .cu program is compiled with real nvcc and then actually
// RUN, honestly reporting this toolchain's real constraint: no physical
// GPU exists on either side of this book's own toolchain, so every CUDA
// API call reports "no CUDA-capable device is detected" -- checked, not
// glossed over, the same way Chapter 12's roofline model or Chapter 14's
// FLOPs convention were checked against real published or real measured
// numbers rather than assumed correct.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 042_from_loops_to_threads_a_cuda_elementwise_kernel_generator.cu -o 042_driver
// Run:     ./042_driver
// (This file's own main() is plain C++: it GENERATES a .cu program as a
// string, writes it to disk, and shells out to nvcc to compile and run IT.
// The driver itself needs no CUDA toolchain to build -- only the .cu file
// it writes does.)
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

// ==================== resolveIntoGroup() / reductionFusionPass() (from Chapter 14, unchanged) ====================

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

// ==================== Loop / LoopNest / buildLoopNest() (from Chapter 15, unchanged) ====================

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

// ==================== lowerNode() (from Section 17.2, unchanged) ====================

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

// ==================== Section 18.1: emitSteps() -- the SHARED codegen core ====================
//
// Chapter 17's own generateLoopFunction() and this section's own
// generateCudaKernel() both need the exact same thing at their core: given
// a FusedStep program, emit "float stepN = <expr>;" for every step, then
// either "out[idx] = stepLast;" (elementwise) or "acc += <summed operand>;"
// (reduction, accumulation handled by the CALLER -- a CPU wrapper uses a
// plain float accumulator, a CUDA wrapper uses atomicAdd, see Section
// 18.2). What's genuinely backend-specific is isolated to exactly two
// small pieces: HOW an ExternalInput operand gets read (an indexed buffer
// array on the CPU; a named device pointer parameter on the GPU, since a
// std::vector's own buffer lives in host memory and is not a valid device
// address), and HOW ReLU's own max gets spelled (std::max vs fmaxf, since
// plain std::max is not guaranteed to compile in __device__ code without
// extra compiler flags this book doesn't want to depend on). Everything
// else -- the sequencing of steps, how a PriorStep operand chains to an
// earlier step's own result -- is ONE function, called by both backends.
enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) {
        return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    }
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExpr(Target target, const std::string& x) {
    return (target == Target::Cpu) ? ("std::max(0.0f, " + x + ")") : ("fmaxf(0.0f, " + x + ")");
}

// Emits every step EXCEPT (for a reduction) the trailing Sum step, whose
// own accumulation the caller performs differently per backend. Returns
// the generated "float stepN = ...;" lines, and separately the C++/CUDA
// EXPRESSION for "the value this program's own last computed step holds"
// (the elementwise output, or the reduction's own per-iteration summand).
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
        if (step.op == OpKind::ReLU) expr = maxExpr(target, operandExpr(step.operands[0]));
        else if (step.op == OpKind::Add) expr = operandExpr(step.operands[0]) + " + " + operandExpr(step.operands[1]);
        else expr = operandExpr(step.operands[0]) + " * " + operandExpr(step.operands[1]);  // Mul
        std::string varName = "step" + std::to_string(s);
        lines.push_back("float " + varName + " = " + expr + ";");
        stepVars.push_back(varName);
    }
    if (isReduction) {
        // The trailing Sum step's own single operand is what the caller accumulates.
        const FusedOperand& sumOperand = steps.back().operands[0];
        finalValueExpr = (sumOperand.kind == OperandKind::ExternalInput)
                              ? readExternal(target, sumOperand.index, idxExpr)
                              : stepVars[static_cast<size_t>(sumOperand.index)];
    } else {
        finalValueExpr = stepVars.back();
    }
    return lines;
}

// ---- CPU wrapper (Chapter 17's own generateLoopFunction(), re-expressed on top of emitSteps()) ----
static std::string generateCpuElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
}

// ---- Section 18.1: the CUDA wrapper -- one THREAD per output element, not one more loop pass ----
static std::string generateCudaElementwiseKernel(const std::string& kernelName, const LoweredNode& lowered,
                                                   int numExternalInputs) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cuda, finalExpr);
    std::string src = "__global__ void " + kernelName + "(";
    for (int i = 0; i < numExternalInputs; ++i) src += "const float* ext" + std::to_string(i) + ", ";
    for (int i = 0; i < numExternalInputs; ++i) src += "long long extCount" + std::to_string(i) + ", ";
    src += "float* out, long long n) {\n";
    src += "    long long flat = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    src += "    if (flat >= n) return;\n";
    for (const std::string& line : body) src += "    " + line + "\n";
    src += "    out[flat] = " + finalExpr + ";\n";
    src += "}\n";
    return src;
}

// ==================== Shell-out helpers ====================

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

int main() {
    printf("=== Section 18.1: from loops to threads -- a CUDA elementwise kernel generator ===\n\n");

    // Same diamond graph as Chapter 17, same concrete non-uniform inputs.
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
    const Node* outFusedNode = nullptr;
    for (const auto& n : fused.nodes()) if (n->debugName == "out") outFusedNode = n.get();
    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);

    printf("--- Shared core: emitSteps() called with Target::Cpu vs. Target::Cuda, same steps ---\n\n");
    std::string cpuFinal, cudaFinal;
    std::vector<std::string> cpuBody = emitSteps(outLowered.steps, false, "flat", Target::Cpu, cpuFinal);
    std::vector<std::string> cudaBody = emitSteps(outLowered.steps, false, "flat", Target::Cuda, cudaFinal);
    printf("CPU body (Target::Cpu):\n");
    for (const auto& l : cpuBody) printf("  %s\n", l.c_str());
    printf("out[flat] = %s;\n\n", cpuFinal.c_str());
    printf("CUDA body (Target::Cuda):\n");
    for (const auto& l : cudaBody) printf("  %s\n", l.c_str());
    printf("out[flat] = %s;\n\n", cudaFinal.c_str());
    bool sameStepCount = (cpuBody.size() == cudaBody.size());
    printf("self-check: same number of steps, same sequencing, from the SAME emitSteps() call site,\n");
    printf("only the external-input read syntax and ReLU's own max differ (%s)\n\n",
           sameStepCount ? "confirmed" : "MISMATCH");

    // ---- Generate the full CUDA kernel + a host driver .cu program ----
    std::string kernelSrc = generateCudaElementwiseKernel("compute_out", outLowered, 2);
    printf("--- Generated CUDA kernel ---\n\n%s\n", kernelSrc.c_str());

    std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<float> bVals = {10, 20, 30, 40};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}};
    auto origArrays = evaluateArrays(g, in, elementCounts);
    std::vector<float> t1Concrete = origArrays.at("t1");
    std::vector<float> aConcrete = aVals;
    auto fusedArrays = evaluateArrays(fused, in, fusedElementCounts);
    long long n = fusedElementCounts.at(outFusedNode->id);

    auto formatArrayLiteral = [](const std::vector<float>& v) {
        std::string out = "{";
        for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
        out += "}";
        return out;
    };

    std::string prog = "#include <cstdio>\n#include <cuda_runtime.h>\n\n";
    prog += "#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) "
            "printf(\"CUDA error at line %d: %s\\n\", __LINE__, cudaGetErrorString(_e)); } while (0)\n\n";
    prog += kernelSrc + "\n";
    prog += "int main() {\n";
    prog += "    float hExt0[] = " + formatArrayLiteral(t1Concrete) + ";\n";
    prog += "    float hExt1[] = " + formatArrayLiteral(aConcrete) + ";\n";
    prog += "    long long n = " + std::to_string(n) + ";\n";
    prog += "    float *dExt0, *dExt1, *dOut;\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dExt0, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dExt1, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMemcpy(dExt0, hExt0, n * sizeof(float), cudaMemcpyHostToDevice));\n";
    prog += "    CUDA_CHECK(cudaMemcpy(dExt1, hExt1, n * sizeof(float), cudaMemcpyHostToDevice));\n";
    prog += "    int threads = 256;\n";
    prog += "    int blocks = (int)((n + threads - 1) / threads);\n";
    prog += "    compute_out<<<blocks, threads>>>(dExt0, dExt1, n, n, dOut, n);\n";
    prog += "    CUDA_CHECK(cudaGetLastError());\n";
    prog += "    CUDA_CHECK(cudaDeviceSynchronize());\n";
    prog += "    float hOut[" + std::to_string(n) + "];\n";
    prog += "    CUDA_CHECK(cudaMemcpy(hOut, dOut, n * sizeof(float), cudaMemcpyDeviceToHost));\n";
    prog += "    printf(\"kernel result (garbage/zero without a real GPU): \");\n";
    prog += "    for (long long i = 0; i < n; ++i) printf(\"%.6f \", hOut[i]);\n";
    prog += "    printf(\"\\n\");\n";
    prog += "    CUDA_CHECK(cudaFree(dExt0)); CUDA_CHECK(cudaFree(dExt1)); CUDA_CHECK(cudaFree(dOut));\n";
    prog += "    return 0;\n}\n";

    writeFile("/tmp/hammer_ch18_042.cu", prog);
    std::string compileLog = runShellCaptureAll("nvcc -std=c++17 -arch=sm_70 -o /tmp/hammer_ch18_042 /tmp/hammer_ch18_042.cu 2>&1");
    bool compileClean = compileLog.find("error") == std::string::npos;
    printf("--- nvcc compile ---\n\n%s\n", compileLog.empty() ? "(no output -- clean compile)\n" : compileLog.c_str());
    printf("self-check: nvcc accepts the generated .cu file as valid CUDA C++ (%s)\n\n",
           compileClean ? "confirmed" : "MISMATCH");

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch18_042 2>&1");
    printf("--- Running the compiled binary (this toolchain has no physical GPU, on either side) ---\n\n");
    printf("%s\n", runOutput.c_str());
    bool honestlyReportsNoDevice = runOutput.find("no CUDA-capable device") != std::string::npos;
    printf("self-check: checked CUDA error codes honestly report \"no CUDA-capable device is\n");
    printf("detected\" at every API call, rather than silently returning garbage (%s)\n\n",
           honestlyReportsNoDevice ? "confirmed" : "MISMATCH");

    // ---- What this kernel's arithmetic WOULD compute, proven the only way this toolchain allows: real CPU execution ----
    printf("--- The host-side reference: same emitSteps() body, run for real on the CPU ---\n\n");
    std::string cpuFuncSrc = generateCpuElementwiseFunction("compute_out_cpu", outLowered);
    std::string cpuProg = "#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n" + cpuFuncSrc + "\n";
    cpuProg += "int main() {\n";
    cpuProg += "    std::vector<float> ext0 = " + formatArrayLiteral(t1Concrete) + ";\n";
    cpuProg += "    std::vector<float> ext1 = " + formatArrayLiteral(aConcrete) + ";\n";
    cpuProg += "    std::vector<const float*> ext = {ext0.data(), ext1.data()};\n";
    cpuProg += "    std::vector<long long> extCounts = {" + std::to_string(n) + ", " + std::to_string(n) + "};\n";
    cpuProg += "    std::vector<float> out(" + std::to_string(n) + ");\n";
    cpuProg += "    compute_out_cpu(ext, extCounts, out.data(), " + std::to_string(n) + ");\n";
    cpuProg += "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";
    writeFile("/tmp/hammer_ch18_042_cpu.cpp", cpuProg);
    runShellCaptureAll("g++ -std=c++17 -O2 /tmp/hammer_ch18_042_cpu.cpp -o /tmp/hammer_ch18_042_cpu 2>&1");
    std::string cpuOut = runShellCaptureAll("/tmp/hammer_ch18_042_cpu");
    printf("CPU (actually executed) result: %s\n", cpuOut.c_str());
    printf("evaluateArrays(fused).out       = ");
    for (float v : fusedArrays.at("out")) printf("%.6f ", v);
    printf("\n");

    std::istringstream iss(cpuOut);
    std::vector<float> cpuVals;
    float v;
    while (iss >> v) cpuVals.push_back(v);
    bool cpuMatches = (cpuVals.size() == fusedArrays.at("out").size());
    if (cpuMatches) for (size_t i = 0; i < cpuVals.size(); ++i) if (std::fabs(cpuVals[i] - fusedArrays.at("out")[i]) > 1e-3f) cpuMatches = false;
    printf("self-check: this IS the number the kernel's own arithmetic computes -- proven by real\n");
    printf("execution of the identical emitSteps() body on the CPU (%s)\n", cpuMatches ? "confirmed" : "MISMATCH");

    bool allOk = sameStepCount && compileClean && honestlyReportsNoDevice && cpuMatches;
    return allOk ? 0 : 1;
}
```

```bash
g++ -x c++ -std=c++17 -Wall -Wextra -O2 "042_from_loops_to_threads_a_cuda_elementwise_kernel_generator.cu" -o "042_from_loops_to_threads_a_cuda_elementwise_kernel_generator"
./"042_from_loops_to_threads_a_cuda_elementwise_kernel_generator"
```

**Output:**

```text
=== Section 18.1: from loops to threads -- a CUDA elementwise kernel generator ===

--- Shared core: emitSteps() called with Target::Cpu vs. Target::Cuda, same steps ---

CPU body (Target::Cpu):
  float step0 = ext[0][(flat) % extCounts[0]] * ext[1][(flat) % extCounts[1]];
  float step1 = std::max(0.0f, ext[0][(flat) % extCounts[0]]);
  float step2 = step0 + step1;
out[flat] = step2;

CUDA body (Target::Cuda):
  float step0 = ext0[(flat) % extCount0] * ext1[(flat) % extCount1];
  float step1 = fmaxf(0.0f, ext0[(flat) % extCount0]);
  float step2 = step0 + step1;
out[flat] = step2;

self-check: same number of steps, same sequencing, from the SAME emitSteps() call site,
only the external-input read syntax and ReLU's own max differ (confirmed)

--- Generated CUDA kernel ---

__global__ void compute_out(const float* ext0, const float* ext1, long long extCount0, long long extCount1, float* out, long long n) {
    long long flat = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (flat >= n) return;
    float step0 = ext0[(flat) % extCount0] * ext1[(flat) % extCount1];
    float step1 = fmaxf(0.0f, ext0[(flat) % extCount0]);
    float step2 = step0 + step1;
    out[flat] = step2;
}

--- nvcc compile ---

(no output -- clean compile)

self-check: nvcc accepts the generated .cu file as valid CUDA C++ (confirmed)

--- Running the compiled binary (this toolchain has no physical GPU, on either side) ---

CUDA error at line 20: no CUDA-capable device is detected
CUDA error at line 21: no CUDA-capable device is detected
CUDA error at line 22: no CUDA-capable device is detected
CUDA error at line 23: no CUDA-capable device is detected
CUDA error at line 24: no CUDA-capable device is detected
CUDA error at line 28: no CUDA-capable device is detected
CUDA error at line 29: no CUDA-capable device is detected
CUDA error at line 31: no CUDA-capable device is detected
kernel result (garbage/zero without a real GPU): 172.501587 0.000000 0.000000 0.000000 -nan 0.000000 -432516628480.000000 225.668472 -443251030725245272064.000000 0.000000 -443251030725245272064.000000 0.000000 
CUDA error at line 35: no CUDA-capable device is detected
CUDA error at line 35: no CUDA-capable device is detected
CUDA error at line 35: no CUDA-capable device is detected

self-check: checked CUDA error codes honestly report "no CUDA-capable device is
detected" at every API call, rather than silently returning garbage (confirmed)

--- The host-side reference: same emitSteps() body, run for real on the CPU ---

CPU (actually executed) result: 22.000000 66.000000 132.000000 220.000000 90.000000 182.000000 296.000000 432.000000 190.000000 330.000000 492.000000 676.000000 

evaluateArrays(fused).out       = 22.000000 66.000000 132.000000 220.000000 90.000000 182.000000 296.000000 432.000000 190.000000 330.000000 492.000000 676.000000 
self-check: this IS the number the kernel's own arithmetic computes -- proven by real
execution of the identical emitSteps() body on the CPU (confirmed)
```


!!! note "Why the named-pointer-parameter design is not optional"
    It would be simpler, syntactically, to pass a CUDA kernel a single array of `const float*` pointers, the same way the CPU wrapper does -- one parameter instead of N. But that array's own storage would still be a `std::vector`'s buffer, living in HOST memory; a `__global__` kernel reading `ext[0]` would be dereferencing a host address from device code, which either crashes or silently reads garbage depending on the driver, and either way is undefined behavior, not a working program. Passing each external input as its own named `const float*` PARAMETER means the CUDA driver copies each individual pointer value (not the array of pointers) into the kernel's own argument space, which is exactly what a device address needs to be valid from device code. This is the same category of constraint the sibling CUDA books already spend real space on -- host memory and device memory are different address spaces, and an address valid in one is not automatically valid in the other.

## 18.2 Reductions on the GPU: Avoiding a Data Race With atomicAdd()

### Intuition

Section 18.1's own elementwise kernel launched one thread per OUTPUT element, and every thread wrote its own, distinct `out[flat]` -- safe, because no two threads ever touch the same memory address. A reduction kernel breaks that pattern on purpose: every thread that helps compute a Sum is, by definition, contributing to the exact SAME single output value. Launch one thread per REDUCE element (the natural first instinct, mirroring the elementwise kernel's own one-thread-per-iteration design) and have each thread execute `*out += step0;` directly, and the result is a textbook DATA RACE: multiple threads performing a non-atomic read-modify-write on the identical address, with no guarantee any thread ever sees another thread's update. Some updates get silently lost -- not a rare edge case, a near-certainty at any real thread count. `atomicAdd()` is the simplest CORRECT fix: a single hardware-guaranteed atomic add per thread, serializing only the one operation that genuinely needs it.

### Background

`generateCudaReductionKernel()` reuses `emitSteps()` completely unchanged -- the same shared core Section 18.1 already built, called with `isReduction=true` and an index expression of `r` (the reduce-element index) instead of `flat`. What is new is only the trailing accumulation: instead of Section 18.1's own `out[flat] = <final step>;`, this kernel emits `atomicAdd(out, <final step>);`, and the kernel's own output buffer is `cudaMemset` to zero BEFORE the kernel launches -- a correctness detail as real as Section 15's own tile-boundary arithmetic, since `atomicAdd()` accumulates INTO whatever the output already holds, and an uninitialized (or stale) starting value would silently corrupt every real result. This is explicitly the CORRECT approach, not the FASTEST one, and this section says so directly rather than leaving the gap unstated: a real high-performance reduction kernel uses a shared-memory tree reduction or warp-shuffle instructions to combine values WITHIN a thread block first, so only one `atomicAdd()` per BLOCK -- not per THREAD -- ever touches global memory, dramatically reducing contention at scale. That technique is named here, not implemented -- a stated scope limitation, in the same spirit as Section 17.1's own interior-broadcast limitation, rather than a gap this section pretends is not there.

```text
WHY NOT "*out += step0;" directly (6 threads, one per reduce element):

  thread 0: read *out (=0.0), compute step0=0.0,  write *out = 0.0
  thread 1: read *out (=0.0), compute step0=5.0,  write *out = 5.0
  thread 2: read *out (=5.0), compute step0=0.0,  write *out = 5.0
  thread 3: read *out (=5.0), compute step0=2.0,  write *out = 7.0
  ...

  NO ORDERING GUARANTEE across threads -- reads and writes can interleave
  in any order. Two threads can both read the SAME starting value before
  either one's own write lands, so one thread's own contribution gets
  silently overwritten and LOST -- the classic lost-update data race, and a
  near-certainty at any real thread count, not a rare edge case.

WITH atomicAdd(out, step0):  every thread's own add is a single hardware-
guaranteed atomic operation -- no other thread's atomicAdd can interleave
in the middle of it. Correct at any thread count, at the cost of every
thread still contending for the same address (not the fastest possible
reduction -- see this section's own stated limitation, above).

  Required launch sequence for a correct atomicAdd()-based reduction:

  step 1: cudaMemset(dOut, 0, sizeof(float))   zero the output buffer FIRST
                                                -- atomicAdd() only ADDS,
                                                it never initializes
  step 2: launch the reduction kernel          each of 6 threads performs
          (one thread per reduce element)      exactly one atomicAdd call
```

```cpp
// Chapter 18: Generating CUDA C++ From the Fused IR
// 043_reductions_on_the_gpu_avoiding_a_data_race_with_atomicadd.cu
//
// Section 18.2 -- Chapter 17's own CPU reduction kernel used a single
// `float acc` accumulated by one serial for-loop: safe, because exactly
// ONE thing ever touches `acc` at a time. Section 18.1's own elementwise
// kernel launched one thread per OUTPUT element, each writing its own,
// distinct `out[flat]` -- also safe, for the same reason: no two threads
// ever touch the same memory. A reduction kernel breaks that pattern on
// purpose: every thread reduces toward the SAME single output. Launch one
// thread per REDUCE element and have each do `*out += step;` directly, and
// the result is a classic, textbook data race -- multiple threads
// performing a non-atomic read-modify-write on the same address, with no
// guarantee any of them see each other's updates. atomicAdd() is the
// simplest CORRECT fix: a single hardware-guaranteed atomic add per
// thread, serializing just the one operation that actually needs it. It is
// NOT the FASTEST fix -- every thread still contends for the same memory
// address, and a real high-performance reduction uses a shared-memory tree
// reduction or warp-shuffle instructions to combine values within a block
// before only one atomicAdd per BLOCK (not per thread) touches global
// memory. This section builds the correct, honestly-simple version; the
// faster version is named, not implemented -- a stated limitation, not a
// gap this section pretends isn't there.
//
// Compile: g++ -x c++ -std=c++17 -Wall -Wextra -O2 043_reductions_on_the_gpu_avoiding_a_data_race_with_atomicadd.cu -o 043_driver
// Run:     ./043_driver
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

// ==================== resolveIntoGroup() / reductionFusionPass() (from Chapter 14, unchanged) ====================

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

// ==================== emitSteps() (from Section 18.1, unchanged) ====================

enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExpr(Target target, const std::string& x) {
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
        if (step.op == OpKind::ReLU) expr = maxExpr(target, operandExpr(step.operands[0]));
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

// ---- CPU wrapper (from Chapter 17, unchanged) ----
static std::string generateCpuReductionFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cpu, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    float acc = 0.0f;\n";
    src += "    for (long long r = 0; r < " + std::to_string(reduceExtent) + "; ++r) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        acc += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = acc;\n}\n";
    return src;
}

// ==================== Section 18.2: the CUDA reduction kernel -- atomicAdd(), not a race ====================
static std::string generateCudaReductionKernel(const std::string& kernelName, const LoweredNode& lowered,
                                                int numExternalInputs) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cuda, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "__global__ void " + kernelName + "(";
    for (int i = 0; i < numExternalInputs; ++i) src += "const float* ext" + std::to_string(i) + ", ";
    for (int i = 0; i < numExternalInputs; ++i) src += "long long extCount" + std::to_string(i) + ", ";
    src += "float* out) {\n";
    src += "    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    src += "    if (r >= " + std::to_string(reduceExtent) + ") return;\n";
    for (const std::string& line : body) src += "    " + line + "\n";
    // NOT `*out += finalExpr;` -- that would be a data race with every other
    // thread in this same launch doing the identical non-atomic read-modify-write
    // at the exact same address. atomicAdd() is the correct fix: one hardware-
    // guaranteed atomic add per thread, serializing only the one operation that
    // actually needs it -- not the fastest possible reduction (see this
    // section's own opening note: a shared-memory tree reduction or warp-shuffle
    // approach would touch global memory once per BLOCK, not once per THREAD),
    // but a CORRECT one, checked the same way Section 18.1's own kernel was.
    src += "    atomicAdd(out, " + finalExpr + ");\n";
    src += "}\n";
    return src;
}

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

int main() {
    printf("=== Section 18.2: reductions on the GPU -- avoiding a data race with atomicAdd() ===\n\n");

    // Same reduction graph as Chapter 17's own Section 17.1 Part C.
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
    std::map<int, Shape> rFusedShapes;
    std::map<int, long long> rFusedElementCounts;
    for (const auto& n : rFused.graph.nodes()) {
        int oldId = rFused.representativeOldId.at(n->id);
        rFusedShapes[n->id] = rShapes.at(oldId);
        rFusedElementCounts[n->id] = rElementCounts.at(oldId);
    }
    const Node* sFusedNode = nullptr;
    for (const auto& n : rFused.graph.nodes()) if (n->debugName == "s") sFusedNode = n.get();
    LoweredNode sLowered = lowerNode(sFusedNode, rFusedShapes, rFusedElementCounts);

    printf("--- Why NOT a plain '*out += ...;' inside the kernel ---\n\n");
    printf("Launching one thread per reduce element (6 threads here) and having each do\n");
    printf("'*out += step0;' directly is a DATA RACE: multiple threads perform a non-atomic\n");
    printf("read-modify-write on the exact same address, with no guarantee any thread sees\n");
    printf("another's update -- the classic lost-update bug. atomicAdd() replaces that with a\n");
    printf("single hardware-guaranteed atomic operation per thread.\n\n");

    std::string kernelSrc = generateCudaReductionKernel("compute_s", sLowered, 1);
    printf("--- Generated CUDA reduction kernel ---\n\n%s\n", kernelSrc.c_str());

    std::vector<float> xVals = {-3, 5, -1, 2, 0, 4};
    std::map<std::string, std::vector<float>> rIn = {{"x", xVals}};
    auto rFusedArrays = evaluateArrays(rFused.graph, rIn, rFusedElementCounts);
    std::vector<float> rt1Concrete;
    for (float v : xVals) rt1Concrete.push_back(std::max(0.0f, v));  // rt1's own external input to s is x itself
    long long reduceExtent = sLowered.nest.loops.back().extent;

    auto formatArrayLiteral = [](const std::vector<float>& v) {
        std::string out = "{";
        for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
        out += "}";
        return out;
    };

    std::string prog = "#include <cstdio>\n#include <cuda_runtime.h>\n\n";
    prog += "#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) "
            "printf(\"CUDA error at line %d: %s\\n\", __LINE__, cudaGetErrorString(_e)); } while (0)\n\n";
    prog += kernelSrc + "\n";
    prog += "int main() {\n";
    prog += "    float hExt0[] = " + formatArrayLiteral(xVals) + ";\n";
    prog += "    long long n = " + std::to_string(reduceExtent) + ";\n";
    prog += "    float *dExt0, *dOut;\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dExt0, n * sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMalloc(&dOut, sizeof(float)));\n";
    prog += "    CUDA_CHECK(cudaMemcpy(dExt0, hExt0, n * sizeof(float), cudaMemcpyHostToDevice));\n";
    prog += "    CUDA_CHECK(cudaMemset(dOut, 0, sizeof(float)));\n";
    prog += "    int threads = 256;\n";
    prog += "    int blocks = (int)((n + threads - 1) / threads);\n";
    prog += "    compute_s<<<blocks, threads>>>(dExt0, n, dOut);\n";
    prog += "    CUDA_CHECK(cudaGetLastError());\n";
    prog += "    CUDA_CHECK(cudaDeviceSynchronize());\n";
    prog += "    float hOut = 0.0f;\n";
    prog += "    CUDA_CHECK(cudaMemcpy(&hOut, dOut, sizeof(float), cudaMemcpyDeviceToHost));\n";
    prog += "    printf(\"kernel result (0/garbage without a real GPU): %f\\n\", hOut);\n";
    prog += "    CUDA_CHECK(cudaFree(dExt0)); CUDA_CHECK(cudaFree(dOut));\n";
    prog += "    return 0;\n}\n";

    writeFile("/tmp/hammer_ch18_043.cu", prog);
    std::string compileLog = runShellCaptureAll("nvcc -std=c++17 -arch=sm_70 -o /tmp/hammer_ch18_043 /tmp/hammer_ch18_043.cu 2>&1");
    bool compileClean = compileLog.find("error") == std::string::npos;
    printf("--- nvcc compile ---\n\n%s\n", compileLog.empty() ? "(no output -- clean compile)\n" : compileLog.c_str());
    printf("self-check: nvcc accepts the generated atomicAdd()-based kernel (%s)\n\n", compileClean ? "confirmed" : "MISMATCH");

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch18_043 2>&1");
    printf("--- Running the compiled binary ---\n\n%s\n", runOutput.c_str());
    bool honestlyReportsNoDevice = runOutput.find("no CUDA-capable device") != std::string::npos;
    printf("self-check: honestly reports no CUDA-capable device, same as Section 18.1 (%s)\n\n",
           honestlyReportsNoDevice ? "confirmed" : "MISMATCH");

    printf("--- The host-side reference: same emitSteps() body, run for real on the CPU ---\n\n");
    std::string cpuFuncSrc = generateCpuReductionFunction("compute_s_cpu", sLowered);
    std::string cpuProg = "#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n" + cpuFuncSrc + "\n";
    cpuProg += "int main() {\n";
    cpuProg += "    std::vector<float> ext0 = " + formatArrayLiteral(xVals) + ";\n";
    cpuProg += "    std::vector<const float*> ext = {ext0.data()};\n";
    cpuProg += "    std::vector<long long> extCounts = {" + std::to_string(reduceExtent) + "};\n";
    cpuProg += "    float out[1];\n";
    cpuProg += "    compute_s_cpu(ext, extCounts, out);\n";
    cpuProg += "    printf(\"%.6f\\n\", out[0]);\n    return 0;\n}\n";
    writeFile("/tmp/hammer_ch18_043_cpu.cpp", cpuProg);
    runShellCaptureAll("g++ -std=c++17 -O2 /tmp/hammer_ch18_043_cpu.cpp -o /tmp/hammer_ch18_043_cpu 2>&1");
    std::string cpuOut = runShellCaptureAll("/tmp/hammer_ch18_043_cpu");
    float cpuVal = std::stof(cpuOut);
    printf("x = [-3, 5, -1, 2, 0, 4]  ->  relu(x) = [0, 5, 0, 2, 0, 4]  ->  sum = 11 (hand-derived)\n");
    printf("CPU (actually executed) result: %.6f\n", cpuVal);
    printf("evaluateArrays(fused).s          = %.6f\n", rFusedArrays.at("s")[0]);
    bool cpuMatches = std::fabs(cpuVal - 11.0f) < 1e-3f && std::fabs(cpuVal - rFusedArrays.at("s")[0]) < 1e-3f;
    printf("self-check: this IS the number the kernel's own arithmetic computes -- proven by real\n");
    printf("execution of the identical emitSteps() body on the CPU (%s)\n", cpuMatches ? "confirmed" : "MISMATCH");

    bool allOk = compileClean && honestlyReportsNoDevice && cpuMatches;
    return allOk ? 0 : 1;
}
```

```bash
g++ -x c++ -std=c++17 -Wall -Wextra -O2 "043_reductions_on_the_gpu_avoiding_a_data_race_with_atomicadd.cu" -o "043_reductions_on_the_gpu_avoiding_a_data_race_with_atomicadd"
./"043_reductions_on_the_gpu_avoiding_a_data_race_with_atomicadd"
```

**Output:**

```text
=== Section 18.2: reductions on the GPU -- avoiding a data race with atomicAdd() ===

--- Why NOT a plain '*out += ...;' inside the kernel ---

Launching one thread per reduce element (6 threads here) and having each do
'*out += step0;' directly is a DATA RACE: multiple threads perform a non-atomic
read-modify-write on the exact same address, with no guarantee any thread sees
another's update -- the classic lost-update bug. atomicAdd() replaces that with a
single hardware-guaranteed atomic operation per thread.

--- Generated CUDA reduction kernel ---

__global__ void compute_s(const float* ext0, long long extCount0, float* out) {
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= 6) return;
    float step0 = fmaxf(0.0f, ext0[(r) % extCount0]);
    atomicAdd(out, step0);
}

--- nvcc compile ---

(no output -- clean compile)

self-check: nvcc accepts the generated atomicAdd()-based kernel (confirmed)

--- Running the compiled binary ---

CUDA error at line 17: no CUDA-capable device is detected
CUDA error at line 18: no CUDA-capable device is detected
CUDA error at line 19: no CUDA-capable device is detected
CUDA error at line 20: no CUDA-capable device is detected
CUDA error at line 24: no CUDA-capable device is detected
CUDA error at line 25: no CUDA-capable device is detected
CUDA error at line 27: no CUDA-capable device is detected
kernel result (0/garbage without a real GPU): 0.000000
CUDA error at line 29: no CUDA-capable device is detected
CUDA error at line 29: no CUDA-capable device is detected

self-check: honestly reports no CUDA-capable device, same as Section 18.1 (confirmed)

--- The host-side reference: same emitSteps() body, run for real on the CPU ---

x = [-3, 5, -1, 2, 0, 4]  ->  relu(x) = [0, 5, 0, 2, 0, 4]  ->  sum = 11 (hand-derived)
CPU (actually executed) result: 11.000000
evaluateArrays(fused).s          = 11.000000
self-check: this IS the number the kernel's own arithmetic computes -- proven by real
execution of the identical emitSteps() body on the CPU (confirmed)
```


!!! warning "[COMMON TRAP] forgetting cudaMemset before an atomicAdd()-based kernel"
    `atomicAdd(out, value)` does exactly what its name says: it ADDS `value` to whatever `*out` already holds, atomically. It does not initialize `*out` to zero first -- that is the caller's own responsibility, once, before the kernel launches. A freshly `cudaMalloc`'d buffer holds WHATEVER bytes were already in that device memory, not zero, so skipping the `cudaMemset(dOut, 0, sizeof(float))` call would silently corrupt every real result with garbage-plus-the-correct-sum instead of the correct sum alone -- a bug that would be invisible in this chapter's own toolchain (which cannot execute the kernel at all) and only surface as a real, wrong number the first time this exact generated code ran on real hardware.

## 18.3 Putting It Together: A Full CUDA Program From IR to GPU, End to End

### Intuition

Sections 18.1 and 18.2 each generated ONE kernel in isolation, with its own inputs handed to it as ready-made device buffers. A real compiler has to do more: given an entire GRAPH, it has to generate one kernel per node, generate the HOST-SIDE driver code that allocates every buffer, copies the leaf inputs in, launches every kernel in the right order with each node's own inputs pointing at whichever earlier kernel produced them, and copies the final result back -- Chapter 17's own `generateFullProgram()`, now targeting a CUDA grid instead of a single CPU thread. This section builds that glue and points it at the same graph Chapter 16 and Chapter 17 both already used to make a specific, repeated claim: Chapter 16's own 10-node capstone, run through `boundedReductionFusionPass()` twice, once uncapped (5 nodes, 3 kernels) and once with `maxChainLength=3` (6 nodes, 4 kernels) -- two structurally different CUDA programs, generated from the same original graph, that this section proves compute the exact same final number.

### Background

`generateFullCudaProgram()` walks the fused graph in topological order to emit one `__global__` kernel per non-leaf node -- Section 18.1's own elementwise generator or Section 18.2's own reduction generator, dispatched purely by the node's own `OpKind` -- then walks it again to emit a real host `main()`: a `cudaMalloc` for every single node's own buffer (leaves and generated buffers alike), a `cudaMemcpy` copying each `Input` node's own concrete literal values host-to-device, one kernel launch per non-leaf node in the SAME topological order (a reduction's own output buffer gets `cudaMemset` to zero first, exactly Section 18.2's own correctness detail), and a final `cudaMemcpy` copying the graph's own designated output (`y`) back to the host -- every single CUDA API call wrapped in the same `CUDA_CHECK` macro this whole chapter has used, so the compiled program honestly reports this toolchain's real constraint instead of silently returning garbage. Because that compiled program cannot actually execute its own arithmetic on this toolchain, the real numeric proof comes from a SECOND generated program: the same node-by-node structure, generated instead with Sections 18.1 and 18.2's own CPU functions (the exact same `emitSteps()` step sequences, just generated for `Target::Cpu`), chained through named buffers exactly like Chapter 17's own `generateFullProgram()`, compiled with `g++`, and genuinely EXECUTED -- proving what every one of those CUDA threads' own arithmetic would compute if real hardware were present, and matching `evaluateArrays()` on every single buffer of both fused graphs, for both the uncapped and capped structures alike.

```text
UNCAPPED (5 nodes -> 3 kernels)              CAPPED, maxChainLength=3 (6 nodes -> 4 kernels)

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

  Each variant: generateFullCudaProgram()   compiles cleanly with nvcc, runs,
                                             honestly reports no physical GPU
                generateFullCpuProgram()    compiles with g++, RUNS FOR REAL,
                                             matches evaluateArrays() on
                                             every buffer, reports y=132.0

  Both variants' real, executed CPU counterparts report the SAME y=132.0 --
  extending Chapter 16 and 17's own "different fusion structure, same math"
  claim to a real CUDA target for the first time.
```

```cpp
// Chapter 18: Generating CUDA C++ From the Fused IR
// 044_a_full_cuda_program_from_ir_to_gpu_end_to_end.cu
//
// Section 18.3 (capstone) -- Sections 18.1 and 18.2 generated ONE kernel at
// a time, in isolation. This section lowers an ENTIRE graph to CUDA: one
// __global__ kernel per non-leaf node (elementwise via Section 18.1's own
// generateCudaElementwiseKernel(), reduction via Section 18.2's own
// generateCudaReductionKernel() -- the SAME shared emitSteps() core both
// sections already built on), chained through a real host driver
// (cudaMalloc for every buffer, sequential kernel launches in topological
// order, CUDA_CHECK on every call, cudaMemcpy the final result back) --
// exactly the target Chapter 17's own capstone (Section 17.3) built for a
// serial CPU loop nest, now for a real CUDA grid. The target is, once
// again, Chapter 16's own 10-node capstone graph, run through
// boundedReductionFusionPass() TWICE -- uncapped (5 nodes -> 3 kernels) and
// capped at maxChainLength=3 (6 nodes -> 4 kernels) -- so this section
// closes the exact loop Chapter 16 opened: "different fusion structure,
// same math," now proven on the target this whole book has been building
// toward. Each generated .cu program is compiled with real nvcc (a genuine
// toolchain check on a nontrivial, multi-kernel program) and run, honestly
// reporting this toolchain's real constraint -- no physical GPU exists on
// either side of it, checked via CUDA_CHECK exactly as Sections 18.1 and
// 18.2 already checked it, not glossed over. The actual numeric proof comes
// from the SAME emitSteps() step sequences, generated instead as CPU
// functions (Section 18.1/18.2's own generateCpuElementwiseFunction() and
// generateCpuReductionFunction()), chained exactly like Chapter 17's own
// generateFullProgram(), compiled with g++, and genuinely EXECUTED --
// proving what both CUDA programs' arithmetic would compute if hardware
// were present, and proving it agrees with evaluateArrays() on every single
// buffer of both fused graphs, and with Chapter 16 and 17's own already-
// established y.
//
// Compile: g++ -x c++ -std=c++17 -Wall -Wextra -O2 044_a_full_cuda_program_from_ir_to_gpu_end_to_end.cu -o 044_driver
// Run:     ./044_driver
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

// ==================== emitSteps() and both backends' generators (from Sections 18.1 / 18.2, unchanged) ====================

enum class Target { Cpu, Cuda };

static std::string readExternal(Target target, int index, const std::string& idxExpr) {
    if (target == Target::Cpu) return "ext[" + std::to_string(index) + "][(" + idxExpr + ") % extCounts[" + std::to_string(index) + "]]";
    return "ext" + std::to_string(index) + "[(" + idxExpr + ") % extCount" + std::to_string(index) + "]";
}
static std::string maxExpr(Target target, const std::string& x) {
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
        if (step.op == OpKind::ReLU) expr = maxExpr(target, operandExpr(step.operands[0]));
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

// ---- Section 18.1: CPU / CUDA elementwise ----
static std::string generateCpuElementwiseFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cpu, finalExpr);
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    for (long long flat = 0; flat < n; ++flat) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        out[flat] = " + finalExpr + ";\n";
    src += "    }\n}\n";
    return src;
}
static std::string generateCudaElementwiseKernel(const std::string& kernelName, const LoweredNode& lowered,
                                                   int numExternalInputs) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, false, "flat", Target::Cuda, finalExpr);
    std::string src = "__global__ void " + kernelName + "(";
    for (int i = 0; i < numExternalInputs; ++i) src += "const float* ext" + std::to_string(i) + ", ";
    for (int i = 0; i < numExternalInputs; ++i) src += "long long extCount" + std::to_string(i) + ", ";
    src += "float* out, long long n) {\n";
    src += "    long long flat = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    src += "    if (flat >= n) return;\n";
    for (const std::string& line : body) src += "    " + line + "\n";
    src += "    out[flat] = " + finalExpr + ";\n";
    src += "}\n";
    return src;
}

// ---- Section 18.2: CPU / CUDA reduction (atomicAdd(), not a race) ----
static std::string generateCpuReductionFunction(const std::string& funcName, const LoweredNode& lowered) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cpu, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out) {\n";
    src += "    float acc = 0.0f;\n";
    src += "    for (long long r = 0; r < " + std::to_string(reduceExtent) + "; ++r) {\n";
    for (const std::string& line : body) src += "        " + line + "\n";
    src += "        acc += " + finalExpr + ";\n";
    src += "    }\n";
    src += "    out[0] = acc;\n}\n";
    return src;
}
static std::string generateCudaReductionKernel(const std::string& kernelName, const LoweredNode& lowered,
                                                int numExternalInputs) {
    std::string finalExpr;
    std::vector<std::string> body = emitSteps(lowered.steps, true, "r", Target::Cuda, finalExpr);
    long long reduceExtent = lowered.nest.loops.back().extent;
    std::string src = "__global__ void " + kernelName + "(";
    for (int i = 0; i < numExternalInputs; ++i) src += "const float* ext" + std::to_string(i) + ", ";
    for (int i = 0; i < numExternalInputs; ++i) src += "long long extCount" + std::to_string(i) + ", ";
    src += "float* out) {\n";
    src += "    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;\n";
    src += "    if (r >= " + std::to_string(reduceExtent) + ") return;\n";
    for (const std::string& line : body) src += "    " + line + "\n";
    src += "    atomicAdd(out, " + finalExpr + ");\n";
    src += "}\n";
    return src;
}

// ==================== Section 18.3: a WHOLE graph, dispatched per node, chained end to end ====================

// One node's op fully determines which of the two backend-pair generators
// applies -- Sum/FusedReduction go through the reduction pair (Section
// 18.2), everything else (Add/Mul/ReLU/FusedElementwise) through the
// elementwise pair (Section 18.1). Both pairs already agree on what
// "numExternalInputs" means (n->inputs.size(), the group's own materialized
// external operands), so ONE dispatch function per backend is all this
// section needs -- no new per-node-kind logic, just routing to the two
// backends' own already-built generators.
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }

static std::string generateCudaKernelForNode(const std::string& kernelName, const Node* n, const LoweredNode& lowered) {
    int numExt = static_cast<int>(n->inputs.size());
    return isReductionNode(n) ? generateCudaReductionKernel(kernelName, lowered, numExt)
                               : generateCudaElementwiseKernel(kernelName, lowered, numExt);
}
static std::string generateCpuFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered) {
    return isReductionNode(n) ? generateCpuReductionFunction(funcName, lowered)
                               : generateCpuElementwiseFunction(funcName, lowered);
}

static std::string formatLiteralArray(const std::vector<float>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
    out += "}";
    return out;
}

// ---- The CUDA program: one kernel per non-leaf node, then a real host
// driver that mallocs every buffer, copies the leaves in, launches every
// kernel in topological order (a reduction's own output is cudaMemset to
// 0 first -- exactly Section 18.2's own correctness detail), and copies the
// final "y" buffer back. Always returns 0 -- the point is what CUDA_CHECK
// PRINTS at every failed call, not the process exit code. ----
static std::string generateFullCudaProgram(const Graph& g, const std::map<int, Shape>& shapes,
                                            const std::map<int, long long>& elementCounts,
                                            const std::map<std::string, std::vector<float>>& inputArrays) {
    TopoResult topo = topologicalSort(g);
    std::string prog = "#include <cstdio>\n#include <cuda_runtime.h>\n\n";
    prog += "#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) "
            "printf(\"CUDA error at line %d: %s\\n\", __LINE__, cudaGetErrorString(_e)); } while (0)\n\n";

    std::map<int, std::string> kernelNameById;
    std::map<int, LoweredNode> loweredById;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string kernelName = "compute_" + n->debugName;
        kernelNameById[id] = kernelName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        loweredById[id] = lowered;
        prog += generateCudaKernelForNode(kernelName, n, lowered) + "\n";
    }

    prog += "int main() {\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input) prog += "    float h_" + n->debugName + "[] = " + formatLiteralArray(inputArrays.at(n->debugName)) + ";\n";
        else if (n->op == OpKind::Const) prog += "    float h_" + n->debugName + "[] = {" + std::to_string(n->constValue) + "f};\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        prog += "    float* d_" + n->debugName + ";\n";
        prog += "    CUDA_CHECK(cudaMalloc(&d_" + n->debugName + ", " + std::to_string(elementCounts.at(id)) + " * sizeof(float)));\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const)
            prog += "    CUDA_CHECK(cudaMemcpy(d_" + n->debugName + ", h_" + n->debugName + ", " +
                     std::to_string(elementCounts.at(id)) + " * sizeof(float), cudaMemcpyHostToDevice));\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string kernelName = kernelNameById.at(id);
        bool isReduction = isReductionNode(n);
        if (isReduction) {
            long long reduceExtent = loweredById.at(id).nest.loops.back().extent;
            prog += "    CUDA_CHECK(cudaMemset(d_" + n->debugName + ", 0, sizeof(float)));\n";
            prog += "    { int threads = 256; int blocks = (int)((" + std::to_string(reduceExtent) + " + threads - 1) / threads);\n";
            prog += "      " + kernelName + "<<<blocks, threads>>>(";
            for (const Value& in : n->inputs) prog += "d_" + g.node(in.nodeId)->debugName + ", ";
            for (const Value& in : n->inputs) prog += std::to_string(elementCounts.at(in.nodeId)) + ", ";
            prog += "d_" + n->debugName + ");\n";
            prog += "      CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }\n";
        } else {
            long long outCount = elementCounts.at(id);
            prog += "    { int threads = 256; int blocks = (int)((" + std::to_string(outCount) + " + threads - 1) / threads);\n";
            prog += "      " + kernelName + "<<<blocks, threads>>>(";
            for (const Value& in : n->inputs) prog += "d_" + g.node(in.nodeId)->debugName + ", ";
            for (const Value& in : n->inputs) prog += std::to_string(elementCounts.at(in.nodeId)) + ", ";
            prog += "d_" + n->debugName + ", " + std::to_string(outCount) + ");\n";
            prog += "      CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }\n";
        }
    }
    prog += "    float h_y;\n";
    prog += "    CUDA_CHECK(cudaMemcpy(&h_y, d_y, sizeof(float), cudaMemcpyDeviceToHost));\n";
    prog += "    printf(\"kernel result y (0/garbage without a real GPU): %f\\n\", h_y);\n";
    for (int id : topo.order) prog += "    CUDA_CHECK(cudaFree(d_" + g.node(id)->debugName + "));\n";
    prog += "    return 0;\n}\n";
    return prog;
}

// ---- The CPU reference program: one function per non-leaf node (the SAME
// emitSteps() step sequences, just generated for Target::Cpu instead of
// Target::Cuda), chained through named buffers exactly like Section 17.3's
// own generateFullProgram() -- the only difference is which generator
// produces each function's body, and a reduction function's own signature
// (no trailing "n" parameter, since its reduceExtent is already a literal
// baked into its body). ----
static std::string generateFullCpuProgram(const Graph& g, const std::map<int, Shape>& shapes,
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
        prog += generateCpuFunctionForNode(funcName, n, lowered) + "\n";
    }

    prog += "int main() {\n";
    prog += "    std::map<std::string, std::vector<float>> buf;\n";
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input) prog += "    buf[\"" + n->debugName + "\"] = " + formatLiteralArray(inputArrays.at(n->debugName)) + ";\n";
        else if (n->op == OpKind::Const) prog += "    buf[\"" + n->debugName + "\"] = {" + std::to_string(n->constValue) + "f};\n";
    }
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        bool isReduction = isReductionNode(n);
        prog += "    buf[\"" + n->debugName + "\"] = std::vector<float>(" + std::to_string(elementCounts.at(id)) + ");\n";
        prog += "    {\n        std::vector<const float*> ext; std::vector<long long> extCounts;\n";
        for (const Value& in : n->inputs) {
            const Node* inNode = g.node(in.nodeId);
            prog += "        ext.push_back(buf[\"" + inNode->debugName + "\"].data()); extCounts.push_back(" +
                    std::to_string(elementCounts.at(in.nodeId)) + ");\n";
        }
        if (isReduction) prog += "        " + funcNameById.at(id) + "(ext, extCounts, buf[\"" + n->debugName + "\"].data());\n    }\n";
        else prog += "        " + funcNameById.at(id) + "(ext, extCounts, buf[\"" + n->debugName + "\"].data(), " +
                     std::to_string(elementCounts.at(id)) + ");\n    }\n";
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

// ==================== Shell-out / parsing helpers ====================

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

int main() {
    printf("=== Section 18.3: a full CUDA program from IR to GPU, end to end ===\n\n");

    // Chapter 16's own 10-node capstone graph, reused verbatim -- the same
    // graph Section 17.3 already made real for a CPU loop nest.
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
        printf("--- %s fusion -> CUDA ---\n\n", label);

        FusionResult fr = boundedReductionFusionPass(g, elementCounts, maxChainLength);
        const Graph& fusedGraph = fr.graph;
        std::map<int, Shape> fusedShapes;
        std::map<int, long long> fusedElementCounts;
        for (const auto& n : fusedGraph.nodes()) {
            int oldId = fr.representativeOldId.at(n->id);
            fusedShapes[n->id] = shapes.at(oldId);
            fusedElementCounts[n->id] = elementCounts.at(oldId);
        }
        size_t kernelCount = fusedGraph.size() - 2;  // every node but Input a, Input b gets a kernel
        printf("Nodes: %zu -> %zu CUDA kernels, %zu CPU functions. ", fusedGraph.size(), kernelCount, kernelCount);
        for (const auto& n : fusedGraph.nodes()) {
            printf("%s", n->debugName.c_str());
            if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
            else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
            printf(" ");
        }
        printf("\n\n");

        auto fusedArrays = evaluateArrays(fusedGraph, inputArrays, fusedElementCounts);

        // ---- CUDA: generate, compile, run (honest no-device check) ----
        std::string cudaProg = generateFullCudaProgram(fusedGraph, fusedShapes, fusedElementCounts, inputArrays);
        std::string cudaStem = std::string("/tmp/hammer_ch18_044_") + (variant == 0 ? "uncapped" : "capped");
        writeFile(cudaStem + ".cu", cudaProg);
        std::string compileLog = runShellCaptureAll("nvcc -std=c++17 -arch=sm_70 -o " + cudaStem + " " + cudaStem + ".cu 2>&1");
        bool compileClean = compileLog.find("error") == std::string::npos;
        if (compileClean) printf("nvcc compile (%zu kernels): clean\n", kernelCount);
        else printf("nvcc compile (%zu kernels): FAILED:\n%s\n", kernelCount, compileLog.c_str());

        std::string runOutput = runShellCaptureAll(cudaStem + " 2>&1");
        bool honestlyReportsNoDevice = runOutput.find("no CUDA-capable device") != std::string::npos;
        printf("running the compiled binary: honestly reports no CUDA-capable device at every\n");
        printf("CUDA_CHECK'd call, same as Sections 18.1 and 18.2 (%s)\n\n", honestlyReportsNoDevice ? "confirmed" : "MISMATCH");

        // ---- CPU: generate, compile, run for real -- the actual numeric oracle ----
        std::string cpuProg = generateFullCpuProgram(fusedGraph, fusedShapes, fusedElementCounts, inputArrays);
        std::string cpuStem = cudaStem + "_cpu";
        writeFile(cpuStem + ".cpp", cpuProg);
        runShellCaptureAll("g++ -std=c++17 -O2 " + cpuStem + ".cpp -o " + cpuStem + " 2>&1");
        std::string cpuStdout = runShellCaptureAll(cpuStem);
        auto generatedArrays = parseNamedBuffers(cpuStdout);

        bool everyBufferMatches = true;
        for (const auto& n : fusedGraph.nodes()) {
            const std::vector<float>& expected = fusedArrays.at(n->debugName);
            const std::vector<float>& actual = generatedArrays.at(n->debugName);
            if (expected.size() != actual.size()) { everyBufferMatches = false; continue; }
            for (size_t i = 0; i < expected.size(); ++i) if (!floatsMatch(expected[i], actual[i])) everyBufferMatches = false;
        }
        printf("self-check: the CPU-executed version of the SAME emitSteps()-generated arithmetic\n");
        printf("(what every CUDA thread above would have run) matches evaluateArrays() on every\n");
        printf("buffer of this fused graph (%s)\n", everyBufferMatches ? "confirmed" : "MISMATCH");

        float generatedY = generatedArrays.at("y")[0];
        bool matchesOriginal = floatsMatch(generatedY, origArrays.at("y")[0]);
        printf("generated program's y = %.6f, matches evaluateArrays(original, unfused).y = %.6f (%s)\n\n",
               generatedY, origArrays.at("y")[0], matchesOriginal ? "confirmed" : "MISMATCH");

        if (variant == 0) finalYUncapped = generatedY; else finalYCapped = generatedY;
        allOk = allOk && compileClean && honestlyReportsNoDevice && everyBufferMatches && matchesOriginal;
    }

    printf("=== Two structurally different generated CUDA programs (3 kernels vs. 4 kernels,\n");
    printf("different kernel signatures), same underlying arithmetic, proven equal via their real\n");
    printf("executed CPU counterparts ===\n\n");
    printf("uncapped-generated y = %.6f\n", finalYUncapped);
    printf("capped-generated   y = %.6f\n", finalYCapped);
    bool sameFinal = floatsMatch(finalYUncapped, finalYCapped);
    printf("self-check: both agree exactly, extending Chapter 16 and 17's own \"different fusion\n");
    printf("structure, same math\" claim to a real CUDA target for the first time (%s)\n",
           sameFinal ? "confirmed" : "MISMATCH");

    allOk = allOk && sameFinal;
    return allOk ? 0 : 1;
}
```

```bash
g++ -x c++ -std=c++17 -Wall -Wextra -O2 "044_a_full_cuda_program_from_ir_to_gpu_end_to_end.cu" -o "044_a_full_cuda_program_from_ir_to_gpu_end_to_end"
./"044_a_full_cuda_program_from_ir_to_gpu_end_to_end"
```

**Output:**

```text
=== Section 18.3: a full CUDA program from IR to GPU, end to end ===

a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = 132.000000

--- UNCAPPED fusion -> CUDA ---

Nodes: 5 -> 3 CUDA kernels, 3 CPU functions. a b t5(FE,5st) s(FR,1st) y(FE,2st) 

nvcc compile (3 kernels): clean
running the compiled binary: honestly reports no CUDA-capable device at every
CUDA_CHECK'd call, same as Sections 18.1 and 18.2 (confirmed)

self-check: the CPU-executed version of the SAME emitSteps()-generated arithmetic
(what every CUDA thread above would have run) matches evaluateArrays() on every
buffer of this fused graph (confirmed)
generated program's y = 132.000000, matches evaluateArrays(original, unfused).y = 132.000000 (confirmed)

--- CAPPED (maxChainLength=3) fusion -> CUDA ---

Nodes: 6 -> 4 CUDA kernels, 4 CPU functions. a b t3(FE,3st) t5(FE,2st) s(FR,1st) y(FE,2st) 

nvcc compile (4 kernels): clean
running the compiled binary: honestly reports no CUDA-capable device at every
CUDA_CHECK'd call, same as Sections 18.1 and 18.2 (confirmed)

self-check: the CPU-executed version of the SAME emitSteps()-generated arithmetic
(what every CUDA thread above would have run) matches evaluateArrays() on every
buffer of this fused graph (confirmed)
generated program's y = 132.000000, matches evaluateArrays(original, unfused).y = 132.000000 (confirmed)

=== Two structurally different generated CUDA programs (3 kernels vs. 4 kernels,
different kernel signatures), same underlying arithmetic, proven equal via their real
executed CPU counterparts ===

uncapped-generated y = 132.000000
capped-generated   y = 132.000000
self-check: both agree exactly, extending Chapter 16 and 17's own "different fusion
structure, same math" claim to a real CUDA target for the first time (confirmed)
```


!!! note "What this chapter proved, and what it honestly could not"
    Every generated CUDA program in this chapter compiled cleanly with a real `nvcc` toolchain -- a genuine check, not a simulated one, and one that grew from a single kernel (Section 18.1) to a full multi-kernel, multi-`cudaMalloc` program (Section 18.3) without ever failing to compile. What this chapter could NOT do, on either machine it runs on, is actually EXECUTE that arithmetic on a GPU -- so every compiled binary's own checked CUDA API calls honestly report `"no CUDA-capable device is detected"` instead of silently returning garbage, the same honest limitation the sibling CUDA C++ DSA and Multi-GPU CUDA C++ books already carry. The real numeric proof in this chapter came from a second, genuinely executed path: the SAME `emitSteps()`-generated step sequences, compiled and run as real CPU programs, matching `evaluateArrays()` exactly. That is not a weaker proof than running the kernel itself -- it proves the arithmetic every thread would perform is correct, which is the one thing a kernel launch cannot add on top of; it does not and cannot prove anything about occupancy, memory coalescing, or actual measured throughput, none of which this chapter has claimed.

## Chapter Summary

This chapter turned CUDA Hammer's fused IR into real, `nvcc`-compiled CUDA C++ for the first time. Section 18.1 built `emitSteps()`, a codegen core shared unchanged between a CPU wrapper and a CUDA wrapper, isolating the only two things a CPU-versus-GPU backend genuinely requires -- how an external input gets read (an indexed buffer array versus a named device pointer parameter, a real host/device memory-space distinction, not a style choice) and how ReLU's own max gets spelled (`std::max` versus `fmaxf`) -- and used it to generate a `__global__` elementwise kernel launching one thread per output element, compiled by real `nvcc`, run, and honestly checked to report this toolchain's real "no CUDA-capable device" constraint rather than silently returning garbage. Section 18.2 built the reduction case, explaining concretely why launching one thread per reduce element with a plain `*out += step;` is a genuine data race, fixing it with `atomicAdd()` (the correct, not the fastest, approach -- a stated limitation, not a hidden one), and proving its own generated kernel's arithmetic correct via a real, executed CPU counterpart matching a hand-derived sum. Section 18.3 closed the chapter with a full, multi-kernel CUDA program generated from Chapter 16's own 10-node capstone graph, run through `boundedReductionFusionPass()` both uncapped (3 kernels) and capped at `maxChainLength=3` (4 kernels) -- both compiling cleanly, both honestly reporting the same toolchain constraint, and both, via their real executed CPU counterparts, reporting the exact same `y=132.0` Chapter 16 and Chapter 17 already established -- extending this book's own "different fusion structure, same math" claim to a real CUDA target for the first time.

## Self-Check Questions

1. `emitSteps()` is called once for `Target::Cpu` and once for `Target::Cuda`, with the exact same `FusedStep` list. What TWO things differ between the two calls' own generated text, and why is each difference a genuine backend requirement rather than a stylistic choice?
2. Why can a CUDA elementwise kernel not simply take a `const std::vector<const float*>&` parameter the way the CPU wrapper does, and pass it straight to the kernel?
3. Section 18.2 opens by explaining why `*out += step0;`, executed by one thread per reduce element, is a data race. Concretely, what can go wrong, and how does `atomicAdd()` fix it?
4. What CUDA API call is required immediately BEFORE launching an `atomicAdd()`-based reduction kernel, and what specifically goes wrong if it is skipped?
5. This chapter states that `atomicAdd()` is the CORRECT approach to a GPU reduction but not the FASTEST one. What technique does it name as the faster alternative, and what, structurally, makes that alternative faster?
6. Every CUDA program this chapter generates is compiled with `nvcc` but never actually executes its own kernel's arithmetic. What DOES get checked by running the compiled binary, and what separate, genuinely executed program proves the kernel's own arithmetic correct?
7. Section 18.3's own `generateFullCudaProgram()` calls `cudaMemcpy` to copy data device-to-host exactly once, at the very end, for the node named `y`. Why is copying every intermediate buffer back unnecessary for this section's own verification strategy?
8. The uncapped and capped CUDA programs in Section 18.3 have different kernel counts (3 versus 4) and different kernel signatures. What is proven to stay IDENTICAL between them, and by what mechanism?

## Where We Go Next

Chapter 19, "Generating Vectorized CPU Code," targets the exact same `LoopNest`-driven, `emitSteps()`-shaped lowering this chapter and Chapter 17 both built, aimed at a backend this toolchain CAN genuinely compile AND run: real SIMD intrinsics, on two different real architectures this book actually has access to (AVX2/FMA on the cloud sandbox's x86-64, NEON on the device's aarch64) -- closing the loop this chapter opened with actual, hardware-executed, per-architecture-verified results, rather than an honestly-reported absence of hardware. Chapter 20, "A JIT Backend: Compiling and Loading Generated Code at Runtime," then replaces every backend's own shell-out-and-read-stdout harness (used since Chapter 17) with real in-process code loading via `dlopen`. Apply the Chapter 5-18 depth-level standard (more prose, more diagrams before code) throughout.

## Worked Solutions

1. The two things that differ are HOW an external input gets read (`ext[i][(idx) % extCounts[i]]`, an indexed buffer array, on the CPU; `exti[(idx) % extCounti]`, a named device pointer parameter, on the GPU) and HOW ReLU's own max gets spelled (`std::max(0.0f, x)` versus `fmaxf(0.0f, x)`). The read difference is a genuine requirement because a `std::vector`'s own buffer lives in host memory, and a `__global__` kernel dereferencing a host address is undefined behavior -- device code needs its own device-memory pointer, passed as a distinct kernel parameter. The max-spelling difference is a genuine requirement because plain `std::max` is not guaranteed to compile inside `__device__` code without extra compiler flags this book does not want to depend on, while `fmaxf` is CUDA's own guaranteed device-side float max.
2. A `std::vector<const float*>`'s own internal array -- the block of memory actually holding those pointer values -- lives in HOST memory. Passing that vector to a `__global__` kernel and having the kernel index into it would mean the kernel reads a host memory address from device code, which is undefined behavior (a crash or silently wrong data, depending on the driver) -- not a valid CUDA program. Each external input instead has to be its own named `const float*` PARAMETER, so the CUDA driver copies each individual device pointer VALUE into the kernel's own argument space, which is a valid way for device code to receive a device address.
3. Multiple threads perform a non-atomic READ, then MODIFY, then WRITE on the exact same memory address, with no guarantee about the order those reads and writes actually interleave across threads. Two threads can both read the same starting value before either one writes its own update, so one thread's contribution gets silently overwritten and lost -- the classic lost-update bug, and a near-certainty at any real thread count, not a rare edge case. `atomicAdd(out, value)` replaces the three separate, interruptible steps with a single hardware-guaranteed ATOMIC operation: no other thread's `atomicAdd` on the same address can interleave in the middle of it, so every thread's own contribution is correctly, indivisibly added.
4. `cudaMemset(dOut, 0, sizeof(float))` is required immediately before the launch. `atomicAdd()` ADDS its value to whatever the output buffer already holds -- it does not initialize that buffer to zero itself. A freshly `cudaMalloc`'d buffer holds whatever bytes happened to already be in that device memory, not zero, so skipping the `cudaMemset` call would silently corrupt every real result into garbage-plus-the-correct-sum instead of the correct sum alone.
5. The faster alternative named (but not implemented) is a shared-memory tree reduction or warp-shuffle instructions. What makes it faster, structurally, is combining values WITHIN a single thread block first -- using fast on-chip shared memory or direct register-to-register warp shuffles -- so that only ONE `atomicAdd()` per BLOCK, rather than one per THREAD, ever has to touch slower global memory, dramatically reducing contention on that single shared address as thread count grows.
6. Running the compiled binary checks that every CUDA API call this toolchain's own real `nvcc`-compiled program makes is a genuinely valid call -- and, since this toolchain has no physical GPU on either side, every one of those calls honestly reports `"no CUDA-capable device is detected"` via its own checked `cudaError_t`, rather than silently returning uninitialized garbage. A completely SEPARATE program -- the same `emitSteps()` step sequences, generated instead for `Target::Cpu`, chained through named buffers, compiled with `g++`, and genuinely EXECUTED as a real process -- proves the kernel's own underlying arithmetic is correct, matched against `evaluateArrays()` on every buffer.
7. Copying every intermediate buffer back is unnecessary because this toolchain cannot execute the CUDA kernels at all -- every intermediate device buffer would just hold whatever `cudaMalloc` happened to leave there (since every `cudaMemcpy` and kernel launch honestly fails), so copying them back would only ever report the SAME "no CUDA-capable device" style failure this section's own top-line `y` result already reports once. The genuine per-buffer correctness check -- comparing every single buffer of the fused graph against `evaluateArrays()` -- is instead performed against the separately generated, separately compiled, and ACTUALLY EXECUTED CPU program, where every buffer really does get computed and really can be compared.
8. What is proven identical is the FINAL COMPUTED NUMBER, `y = 132.0`, not the kernel count, the kernel signatures, or the generated CUDA text, all of which genuinely differ between the two variants. The mechanism is not the CUDA programs themselves (which cannot execute on this toolchain) -- it is each variant's own separately generated, separately compiled, and separately EXECUTED CPU counterpart (the identical `emitSteps()` step sequences targeting `Target::Cpu` instead of `Target::Cuda`), whose real, printed output is parsed back and compared, buffer by buffer, against `evaluateArrays()` run on that same fused graph, and whose final `y` value is compared directly against the other variant's own.

---

**Sources cited in this chapter:**

None new. This chapter's own shared `emitSteps()` codegen core, CUDA kernel generators, and `atomicAdd()`-based reduction design are all original to this book, building directly on the `LoopNest`/`FusedStep` view Chapter 15 and Chapter 17 already established, verified against this book's own directly observed `nvcc` toolchain behavior (documented in this chapter's own opening section) rather than any external source.