# 19. Generating Vectorized CPU Code

**What you will understand:** the `Isa` abstraction (`Avx2`/`Neon`) and its five small primitives -- `vecLoadOrBroadcast()`, `vecAdd()`, `vecMul()`, `vecMax()`, `vecStore()` -- that hide two real architectures' own different intrinsic names behind one generator; `canVectorizeElementwise()`, the scope check that decides whether a node's own external inputs are safe to load as a contiguous vector or a broadcast scalar, and what it deliberately does NOT handle; `vecFma()`, which hides a genuine hardware difference (AVX2's `_mm256_fmadd_ps(a,b,c)` computes `a*b+c`; NEON's `vfmaq_f32(acc,a,b)` computes `acc+a*b` -- the same three operands, a different argument order) behind a peephole optimization that folds a `Mul` step into the `Add` step immediately after it; a genuinely new vectorized-reduction technique (a running vector accumulator, collapsed to one scalar by a horizontal sum only once, after the main loop); and a full per-node dispatcher that chooses, for every node in a real fused graph, between this chapter's vector path and Chapter 18's own scalar fallback -- generated, compiled, and RUN for real, on two different real architectures this book actually has access to.

**What you need to know first:** Chapter 18's own `emitSteps()` and `Target` enum (reused unchanged for every scalar tail loop and fallback body in this chapter), Chapter 16's own `boundedReductionFusionPass()` (reused verbatim in Section 19.3), and Chapter 13's own single-consumer inlining invariant (`consumers.at(oldId) != 1` forces externalization -- the exact fact Section 19.2 leans on to prove its own FMA fold is always safe).

---

Chapter 18 targeted a platform neither machine this book runs on can actually execute -- every CUDA kernel compiled cleanly with a real `nvcc`, and every one of them honestly reported "no CUDA-capable device" the moment it actually ran. This chapter targets something narrower in scope but, for the first time in Part 4, genuinely executable end to end: real SIMD vector instructions, on the real CPU hardware both of this book's own machines actually have. The cloud sandbox this book verifies against is x86-64 and supports AVX2 with FMA; the device this book delivers to is aarch64 (Apple Silicon) and supports NEON. Both are real, physical vector instruction sets -- eight 32-bit lanes per register on AVX2, four on NEON -- and, unlike Chapter 18's own CUDA target, this chapter's own generated code does not just compile. It runs, on both machines, and its numeric result is compared against `evaluateArrays()` on both.

Before writing a single line of this chapter's own vector codegen, the same honest-toolchain-investigation discipline Chapter 18 opened with applied here too: two tiny, throwaway test programs (written, compiled, run, and deleted before this chapter's own real files were started) settled the facts this whole chapter's own primitives rest on. On the cloud sandbox, `g++ -mavx2 -mfma -dM -E` confirmed that `__AVX2__`, `__FMA__`, and `__x86_64__` are all predefined with those two flags, and a throwaway program calling `_mm256_fmadd_ps(a, b, c)` -- which computes `a*b + c` in one instruction -- compiled and ran correctly (`2*3+1 = 7.0`). On the device, `g++ -dM -E` with NO special flags at all confirmed `__ARM_NEON` and `__aarch64__` are both predefined unconditionally -- NEON is not an optional extension on aarch64 the way AVX2 is on x86-64, it is part of the mandatory baseline instruction set -- and a throwaway program calling `vfmaq_f32(c, a, b)` compiled and ran correctly too, but computing `c + a*b` (`1 + 2*3 = 7.0`): the SAME three operands as `_mm256_fmadd_ps`, but with the accumulator argument FIRST instead of last. That argument-order difference is not a curiosity -- it is the exact fact `vecFma()` in Section 19.2 exists to hide. The same throwaway investigation also surfaced `vaddvq_f32`, a NEON intrinsic that sums all four lanes of a vector in a single instruction; AVX2 has no equivalent single instruction that reduces all eight of its own lanes without reaching for AVX-512, so Section 19.3's own vectorized reduction deliberately does not use it, a stated scope limitation explained where it comes up.

```text
TWO REAL ARCHITECTURES, TWO REAL INTRINSIC SETS, ONE GENERATOR:

  cloud sandbox: x86-64, AVX2/FMA          device: aarch64 (Apple Silicon), NEON

  __m256  (8 x float, 32 bytes)            float32x4_t  (4 x float, 16 bytes)
  header: immintrin.h                        header: arm_neon.h
  compiled with: -mavx2 -mfma               compiled with: (no extra flags --
                                             NEON is the aarch64 baseline)

  _mm256_loadu_ps(p)      vld1q_f32(p)                contiguous load
  _mm256_set1_ps(x)       vdupq_n_f32(x)              scalar broadcast
  _mm256_add_ps(a,b)      vaddq_f32(a,b)               lanewise add
  _mm256_mul_ps(a,b)      vmulq_f32(a,b)               lanewise multiply
  _mm256_max_ps(a,z)      vmaxq_f32(a,z)               lanewise max (ReLU)
  _mm256_storeu_ps(p,x)   vst1q_f32(p,x)               store to memory

  _mm256_fmadd_ps(a,b,c)  vfmaq_f32(c,a,b)      SAME meaning (a*b+c), a
    = a*b + c               = c + a*b            DIFFERENT argument order --
                                                  accumulator LAST vs. FIRST

  ONE Isa enum { Avx2, Neon }, ONE set of vecXxx() functions dispatching on
  it -- every generator in this chapter calls vecAdd()/vecMul()/vecFma()/...
  and never spells an intrinsic name directly except inside those functions.

  Section 19.1 -- elementwise: vecLoadOrBroadcast()/vecAdd()/vecMul()/
                  vecMax()/vecStore(), a main vector loop plus a scalar
                  tail (n not always divisible by 8 or 4), and
                  canVectorizeElementwise() -- the scope check this whole
                  chapter's own vector path depends on

  Section 19.2 -- a real peephole optimization: vecFma() hides the
                  AVX2-vs-NEON argument-order difference, folding a Mul
                  step into the Add right after it -- one instruction
                  instead of two, proven safe by an invariant Chapter 13
                  already established, not by a constructed counterexample

  Section 19.3 -- a whole graph, dispatched per node: a genuinely new
                  vectorized-reduction technique (running accumulator,
                  horizontal sum once), a real scalar fallback for a node
                  the vector path can't safely handle, and Chapter 16's
                  own 10-node capstone graph, vectorized end to end and
                  RUN FOR REAL on both architectures
```

## 19.1 From Threads to Lanes: A Portable SIMD Elementwise Kernel Generator

### Intuition

Chapter 18's own CUDA kernel launched one THREAD per output element, all running concurrently. A SIMD vector instruction takes a different approach to the same idea: one instruction operates on SEVERAL elements at once, packed into a single wide register -- eight 32-bit floats side by side in an AVX2 `__m256`, four in a NEON `float32x4_t`. Where Chapter 18's own `emitSteps()` generated one `float stepN = ...;` line per step, this section's own generator produces one VECTOR instruction per step, each one computing eight (or four) of a node's own output elements simultaneously, advancing the loop index by the vector width instead of by one. The two real architectures this book has access to package that idea under completely different names -- `_mm256_add_ps` versus `vaddq_f32` -- but the SHAPE of the generated code is identical either way: a main loop processing whole vectors, followed by a scalar tail loop (reusing Chapter 18's own `emitSteps()`, completely unchanged) for whatever handful of elements are left over when the total count is not an exact multiple of the vector width.

### Background

Five small functions -- `vecLoadOrBroadcast()`, `vecAdd()`, `vecMul()`, `vecMax()`, `vecStore()` -- are the ONLY place this section's own generator ever spells an actual intrinsic name; every one of them takes an `Isa` (`Avx2` or `Neon`) and returns the right call as a string, and `generateVectorElementwiseFunction()` itself calls only these five, never an intrinsic directly. `vecLoadOrBroadcast()` is the one with a real decision inside it: a node's own external input is either loaded as a genuine CONTIGUOUS vector (`_mm256_loadu_ps(ptr + i)` / `vld1q_f32(ptr + i)`, reading `width` consecutive floats starting at the current loop index) or BROADCAST from a single scalar value (`_mm256_set1_ps(ptr[0])` / `vdupq_n_f32(ptr[0])`, replicating one float into every lane) -- and which of the two happens is decided once, outside the generator, by comparing that input's own element count against 1. The generated function's own external-input access pattern deliberately mirrors the CPU convention Chapter 17 and 18 already established, not Chapter 18's own CUDA convention: a single `const std::vector<const float*>& ext` parameter, indexed as `ext[0]`, `ext[1]`, and so on -- CUDA needed one NAMED pointer parameter per input because a `std::vector`'s own buffer lives in host memory and a `__global__` kernel cannot dereference it; plain CPU code has no such host/device split, so the indexed-array convention this chapter uses is both simpler and, for this backend, strictly more correct.

`canVectorizeElementwise()` is the scope check this section's own vector path -- and every later section's own dispatcher -- depends on: it returns true only when EVERY external input's own element count either equals the node's own output count (a genuine, full-width, contiguous array) or equals exactly 1 (a scalar that broadcasts cleanly against any width). This is a real, stated scope limitation, not an oversight: a genuine INTERIOR broadcast -- a trailing dimension that repeats some number of times greater than one but less than the output width, the same shape Chapter 6 first introduced -- does not fit either case, and `canVectorizeElementwise()` correctly returns false for it. Section 19.1's own worked example proves both halves of that claim: a node with inputs `a:[11]` and a scalar `b` passes the check (11 matches the output count, 1 is always a valid scalar), while a second, separate node built from `a:[3,4]` and `b:[4]` -- a real trailing-dimension broadcast, not a scalar -- correctly fails it. The worked example's own element count of 11 is deliberately NOT a multiple of either architecture's own vector width (8 or 4), forcing a genuine, non-trivial scalar tail on both machines rather than a suspiciously clean "it just happens to divide evenly" result.

```text
vecLoadOrBroadcast(): the ONE real decision this section's whole codegen rests on

  external input's own element count vs. this node's own output count:

    count == outCount   ->  CONTIGUOUS VECTOR LOAD
                             _mm256_loadu_ps(ext[i] + idx)   (AVX2, 8 lanes)
                             vld1q_f32(ext[i] + idx)         (NEON, 4 lanes)

    count == 1           ->  SCALAR BROADCAST
                             _mm256_set1_ps(ext[i][0])       (AVX2, same
                             vdupq_n_f32(ext[i][0])           value in
                                                               every lane)

    anything else         ->  canVectorizeElementwise() returns FALSE --
                              a real interior/trailing broadcast (Chapter 6's
                              own shape) is OUT OF SCOPE for this chapter's
                              own vector path; Section 19.3 shows the
                              dispatcher correctly falling back to scalar
                              code for exactly this case, not silently
                              miscompiling it

Main loop + scalar tail, n=11 elements, AVX2 (width 8) vs. NEON (width 4):

  AVX2:   i=0: process elements 0..7 (one vector instruction per step)
          i=8: 8+8=16 > 11, stop the main loop
          scalar tail: i=8,9,10  (3 elements, Chapter 18's own emitSteps())

  NEON:   i=0: process elements 0..3
          i=4: process elements 4..7
          i=8: 8+4=12 > 11, stop the main loop
          scalar tail: i=8,9,10  (3 elements -- same 3, different vector
                                   width got there by a different route)

  SAME scalar tail logic (emitSteps(), Target::Cpu, completely unchanged)
  on both architectures -- only the vector MAIN loop's own width differs.
```


```cpp
// Chapter 19: Generating Vectorized CPU Code
// 045_from_threads_to_lanes_a_portable_simd_elementwise_kernel_generator.cpp
//
// Section 19.1 -- Chapter 18 launched one CUDA THREAD per output element.
// This chapter targets a genuinely different kind of parallelism: one CPU
// INSTRUCTION processing several elements at once, via real SIMD registers
// -- 8 floats per instruction on the cloud sandbox's own AVX2, 4 floats per
// instruction on the device's own NEON. Unlike Chapter 18's CUDA target,
// this chapter's own generated code is not just compiled -- it is compiled
// AND ACTUALLY EXECUTED, for real, on two genuinely different real
// architectures this book has real access to, closing the loop Chapter 17
// opened and Chapter 18 could not close for lack of a physical GPU.
//
// This section's own vectorized codegen covers exactly two operand shapes
// -- an external input whose buffer is the SAME size as the output (a
// plain contiguous vector load) or a bare SCALAR (broadcast to every
// lane) -- a stated scope limitation, not a silent gap: a genuine
// trailing-suffix broadcast (like Chapter 6's own b:[4] against a:[3,4])
// is NOT handled by this section's own vector path, and this section
// demonstrates that limitation being CAUGHT, not silently miscompiled.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 045_from_threads_to_lanes_a_portable_simd_elementwise_kernel_generator.cpp -o 045_driver
// Run:     ./045_driver
// (This file's own main() is plain, portable C++: it detects which real
// ISA THIS machine's own compiler targets via __x86_64__/__aarch64__,
// generates a vectorized .cpp program as a string, writes it to disk, and
// shells out to compile-and-run IT with the matching real intrinsics
// header and real compiler flags for that architecture.)
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

// ==================== emitSteps() / generateCpuElementwiseFunction() (from Sections 17.2/18.1, unchanged) ====================
// Reused verbatim for exactly ONE purpose in this section: the SCALAR TAIL
// of the vector loop below needs no new logic of its own -- it is Chapter
// 18's own scalar codegen, called again, starting wherever the vector loop
// left off.

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

// ==================== Section 19.1: Isa -- the two real architectures this book actually runs on ====================

enum class Isa { Avx2, Neon };

static int vectorWidth(Isa isa) { return (isa == Isa::Avx2) ? 8 : 4; }
static std::string vecType(Isa isa) { return (isa == Isa::Avx2) ? "__m256" : "float32x4_t"; }
static std::string isaHeaderInclude(Isa isa) {
    return (isa == Isa::Avx2) ? "#include <immintrin.h>" : "#include <arm_neon.h>";
}
static std::string isaName(Isa isa) { return (isa == Isa::Avx2) ? "AVX2/FMA" : "NEON"; }

// The only two operand shapes this section's own vector codegen covers:
// a buffer the SAME size as the output (a plain contiguous load) or a bare
// SCALAR (broadcast to every lane). Anything else -- a genuine trailing-
// suffix broadcast smaller than a full lane, like Chapter 6's own b:[4]
// against a:[3,4] -- is explicitly OUT of scope; canVectorizeElementwise()
// (below) exists to CATCH that case, not silently miscompile it.
static std::string vecLoadOrBroadcast(Isa isa, int index, const std::string& baseExpr, bool isScalar) {
    // Unlike Chapter 18's CUDA kernels (which needed a NAMED pointer
    // parameter per external input -- a real host/device memory-space
    // requirement), this is plain CPU code: ext[index] is already a valid
    // const float* from the SAME std::vector<const float*> Chapter 17/18's
    // own scalar functions already take, so plain pointer arithmetic on
    // ext[index] works directly, no separate parameter needed.
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
    return (isa == Isa::Avx2) ? ("_mm256_storeu_ps(" + dst + ", " + val + ");")
                               : ("vst1q_f32(" + dst + ", " + val + ");");
}

// True only when every external input this node's own steps read is either
// the SAME element count as the output (a plain contiguous vector load) or
// a bare scalar (broadcast). A genuine trailing-suffix broadcast smaller
// than the output but larger than 1 -- which Section 17.1's own scalar
// "% extCounts[i]" trick handles fine one element at a time -- has no safe
// single vector-load translation here, so this predicate returns false and
// the caller is expected to fall back to Chapter 17/18's own scalar path.
static bool canVectorizeElementwise(const Node* n, const std::map<int, long long>& elementCounts) {
    long long outCount = elementCounts.at(n->id);
    for (const Value& in : n->inputs) {
        long long c = elementCounts.at(in.nodeId);
        if (c != outCount && c != 1) return false;
    }
    return true;
}

static std::string generateVectorElementwiseFunction(const std::string& funcName, const LoweredNode& lowered,
                                                       Isa isa, const std::vector<bool>& isScalarInput) {
    int vw = vectorWidth(isa);
    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    src += "    long long i = 0;\n";
    src += "    for (; i + " + std::to_string(vw) + " <= n; i += " + std::to_string(vw) + ") {\n";
    std::vector<std::string> stepVars;
    for (size_t s = 0; s < lowered.steps.size(); ++s) {
        const FusedStep& step = lowered.steps[s];
        auto operandExpr = [&](const FusedOperand& o) -> std::string {
            if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
            return stepVars[static_cast<size_t>(o.index)];
        };
        std::string expr;
        if (step.op == OpKind::ReLU) expr = vecMax(isa, operandExpr(step.operands[0]));
        else if (step.op == OpKind::Add) expr = vecAdd(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1]));
        else expr = vecMul(isa, operandExpr(step.operands[0]), operandExpr(step.operands[1]));
        std::string varName = "v" + std::to_string(s);
        src += "        " + vecType(isa) + " " + varName + " = " + expr + ";\n";
        stepVars.push_back(varName);
    }
    src += "        " + vecStore(isa, "out + i", stepVars.back()) + "\n";
    src += "    }\n";
    // The scalar tail: Chapter 18's own emitSteps(), unchanged, called again
    // starting from wherever the vector loop above stopped -- no new logic.
    std::string finalExpr;
    std::vector<std::string> tailLines = emitSteps(lowered.steps, false, "i", Target::Cpu, finalExpr);
    src += "    for (; i < n; ++i) {\n";
    for (const std::string& line : tailLines) src += "        " + line + "\n";
    src += "        out[i] = " + finalExpr + ";\n";
    src += "    }\n}\n";
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
    printf("=== Section 19.1: from threads to lanes -- a portable SIMD elementwise kernel generator ===\n\n");

    // n=11 on purpose: not a multiple of 8 (AVX2's own width) OR 4 (NEON's
    // own width), so BOTH architectures' own generated code has to take a
    // genuine, non-empty scalar tail -- not a convenient round number.
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addUnary(OpKind::ReLU, t1, "t2");
    Value out = g.addBinary(OpKind::Mul, t2, b, "out");
    (void)out;

    std::map<int, Shape> declared = {{a.nodeId, Shape{{11}}}, {b.nodeId, Shape{}}};
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
    printf("Fused \"out\" node: %zu steps, %zu external inputs (a:[11], b: scalar)\n\n",
           outFusedNode->fusedSteps.size(), outFusedNode->inputs.size());

    bool canVec = canVectorizeElementwise(outFusedNode, fusedElementCounts);
    printf("self-check: canVectorizeElementwise() confirms this node is in scope -- every external\n");
    printf("input is either the full output count (11) or a bare scalar (%s)\n\n", canVec ? "confirmed" : "MISMATCH");

    // ---- The stated scope limitation, CAUGHT, not silently miscompiled ----
    printf("--- What canVectorizeElementwise() correctly REFUSES ---\n\n");
    Graph gb;
    Value ba = gb.addInput("a");
    Value bb = gb.addInput("b");
    Value bout = gb.addBinary(OpKind::Add, ba, bb, "out");
    (void)bout;
    std::map<int, Shape> bDeclared = {{ba.nodeId, Shape{{3, 4}}}, {bb.nodeId, Shape{{4}}}};  // Chapter 6's own case
    std::map<int, Shape> bShapes = inferShapes(gb, bDeclared);
    std::map<int, long long> bElementCounts;
    for (const auto& kv : bShapes) bElementCounts[kv.first] = numElements(kv.second);
    const Node* boutNode = gb.node(bout.nodeId);
    bool canVecBroadcast = canVectorizeElementwise(boutNode, bElementCounts);
    printf("a:[3,4] (12 elements), b:[4] (4 elements) -- a genuine trailing-suffix broadcast, the\n");
    printf("same case Chapter 6's own diamond graph uses. canVectorizeElementwise() returns %s --\n",
           canVecBroadcast ? "true (WRONG)" : "false");
    printf("self-check: this node is correctly identified as OUT OF SCOPE for this section's own\n");
    printf("vector path (%s) -- a real compiler would fall back to Chapter 17/18's own scalar\n",
           !canVecBroadcast ? "confirmed" : "MISMATCH");
    printf("generateCpuElementwiseFunction() for exactly this node, not miscompile it.\n\n");

    // ---- Generate the SAME node's vector body for BOTH real ISAs this book runs on ----
    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);
    std::vector<bool> isScalarInput;
    for (const Value& in : outFusedNode->inputs)
        isScalarInput.push_back(fusedElementCounts.at(in.nodeId) == 1);

    std::string avx2Src = generateVectorElementwiseFunction("compute_out", outLowered, Isa::Avx2, isScalarInput);
    std::string neonSrc = generateVectorElementwiseFunction("compute_out", outLowered, Isa::Neon, isScalarInput);
    printf("--- Generated vector body, Isa::Avx2 (8 lanes/instruction) ---\n\n%s\n", avx2Src.c_str());
    printf("--- Generated vector body, Isa::Neon (4 lanes/instruction) ---\n\n%s\n", neonSrc.c_str());
    printf("self-check: same step sequence, same scalar-tail logic (reused from Chapter 18's own\n");
    printf("emitSteps() unchanged) -- only the intrinsic names, the vector width, and the type\n");
    printf("(__m256 vs. float32x4_t) differ between the two generated bodies (confirmed by reading\n");
    printf("both texts above)\n\n");

    // ---- Detect THIS machine's own real architecture and actually run it ----
#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";  // NEON is mandatory baseline on aarch64 -- no extra flags needed
#else
#error "Section 19.1's own generator targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif
    printf("--- This machine compiles as %s -- generating, compiling, and RUNNING that path for real ---\n\n",
           isaName(hostIsa).c_str());

    std::vector<float> aVals = {1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11};
    std::vector<float> bVals = {2};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}};
    auto fusedArrays = evaluateArrays(fused, in, fusedElementCounts);
    long long n = fusedElementCounts.at(outFusedNode->id);
    int vw = vectorWidth(hostIsa);
    long long vectorIters = n / vw, tailIters = n % vw;
    printf("n=%lld, vector width=%d -> %lld full vector iteration(s), %lld scalar tail element(s)\n\n",
           n, vw, vectorIters, tailIters);

    auto formatArrayLiteral = [](const std::vector<float>& v) {
        std::string out = "{";
        for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
        out += "}";
        return out;
    };
    std::string hostVectorSrc = generateVectorElementwiseFunction("compute_out", outLowered, hostIsa, isScalarInput);
    std::string prog = std::string(isaHeaderInclude(hostIsa)) + "\n#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n";
    prog += hostVectorSrc + "\n";
    prog += "int main() {\n";
    prog += "    std::vector<float> ext0 = " + formatArrayLiteral(aVals) + ";\n";
    prog += "    std::vector<float> ext1 = " + formatArrayLiteral(bVals) + ";\n";
    prog += "    std::vector<const float*> ext = {ext0.data(), ext1.data()};\n";
    prog += "    std::vector<long long> extCounts = {" + std::to_string(aVals.size()) + ", " + std::to_string(bVals.size()) + "};\n";
    prog += "    std::vector<float> out(" + std::to_string(n) + ");\n";
    prog += "    compute_out(ext, extCounts, out.data(), " + std::to_string(n) + ");\n";
    prog += "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";

    writeFile("/tmp/hammer_ch19_045.cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags +
                                                  " /tmp/hammer_ch19_045.cpp -o /tmp/hammer_ch19_045 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags) ---\n\n%s\n", isaName(hostIsa).c_str(),
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());
    printf("self-check: this machine's own real compiler accepts the generated %s intrinsics (%s)\n\n",
           isaName(hostIsa).c_str(), compileClean ? "confirmed" : "MISMATCH");

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch19_045");
    printf("--- Running the compiled binary FOR REAL on this machine's own hardware ---\n\n%s\n", runOutput.c_str());

    std::istringstream iss(runOutput);
    std::vector<float> vecVals;
    float v;
    while (iss >> v) vecVals.push_back(v);
    const std::vector<float>& expected = fusedArrays.at("out");
    bool matches = (vecVals.size() == expected.size());
    if (matches) for (size_t i = 0; i < vecVals.size(); ++i) if (std::fabs(vecVals[i] - expected[i]) > 1e-3f) matches = false;
    printf("evaluateArrays(fused).out = ");
    for (float ev : expected) printf("%.6f ", ev);
    printf("\n");
    printf("self-check: the vector-plus-tail program, ACTUALLY EXECUTED on this machine's own real\n");
    printf("%s hardware, matches evaluateArrays() exactly, element for element (%s)\n",
           isaName(hostIsa).c_str(), matches ? "confirmed" : "MISMATCH");

    bool allOk = canVec && !canVecBroadcast && compileClean && matches;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "045_from_threads_to_lanes_a_portable_simd_elementwise_kernel_generator.cpp" -o "045_from_threads_to_lanes_a_portable_simd_elementwise_kernel_generator"
./"045_from_threads_to_lanes_a_portable_simd_elementwise_kernel_generator"
```

**Output (cloud sandbox, x86-64, AVX2/FMA):**

```text
=== Section 19.1: from threads to lanes -- a portable SIMD elementwise kernel generator ===

Fused "out" node: 3 steps, 2 external inputs (a:[11], b: scalar)

self-check: canVectorizeElementwise() confirms this node is in scope -- every external
input is either the full output count (11) or a bare scalar (confirmed)

--- What canVectorizeElementwise() correctly REFUSES ---

a:[3,4] (12 elements), b:[4] (4 elements) -- a genuine trailing-suffix broadcast, the
same case Chapter 6's own diamond graph uses. canVectorizeElementwise() returns false --
self-check: this node is correctly identified as OUT OF SCOPE for this section's own
vector path (confirmed) -- a real compiler would fall back to Chapter 17/18's own scalar
generateCpuElementwiseFunction() for exactly this node, not miscompile it.

--- Generated vector body, Isa::Avx2 (8 lanes/instruction) ---

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    long long i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(ext[0] + i), _mm256_set1_ps(ext[1][0]));
        __m256 v1 = _mm256_max_ps(v0, _mm256_setzero_ps());
        __m256 v2 = _mm256_mul_ps(v1, _mm256_set1_ps(ext[1][0]));
        _mm256_storeu_ps(out + i, v2);
    }
    for (; i < n; ++i) {
        float step0 = ext[0][(i) % extCounts[0]] + ext[1][(i) % extCounts[1]];
        float step1 = std::max(0.0f, step0);
        float step2 = step1 * ext[1][(i) % extCounts[1]];
        out[i] = step2;
    }
}

--- Generated vector body, Isa::Neon (4 lanes/instruction) ---

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    long long i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v0 = vaddq_f32(vld1q_f32(ext[0] + i), vdupq_n_f32(ext[1][0]));
        float32x4_t v1 = vmaxq_f32(v0, vdupq_n_f32(0.0f));
        float32x4_t v2 = vmulq_f32(v1, vdupq_n_f32(ext[1][0]));
        vst1q_f32(out + i, v2);
    }
    for (; i < n; ++i) {
        float step0 = ext[0][(i) % extCounts[0]] + ext[1][(i) % extCounts[1]];
        float step1 = std::max(0.0f, step0);
        float step2 = step1 * ext[1][(i) % extCounts[1]];
        out[i] = step2;
    }
}

self-check: same step sequence, same scalar-tail logic (reused from Chapter 18's own
emitSteps() unchanged) -- only the intrinsic names, the vector width, and the type
(__m256 vs. float32x4_t) differ between the two generated bodies (confirmed by reading
both texts above)

--- This machine compiles as AVX2/FMA -- generating, compiling, and RUNNING that path for real ---

n=11, vector width=8 -> 1 full vector iteration(s), 3 scalar tail element(s)

--- g++ compile (real AVX2/FMA flags) ---

(no output -- clean compile)

self-check: this machine's own real compiler accepts the generated AVX2/FMA intrinsics (confirmed)

--- Running the compiled binary FOR REAL on this machine's own hardware ---

6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000 22.000000 0.000000 26.000000 

evaluateArrays(fused).out = 6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000 22.000000 0.000000 26.000000 
self-check: the vector-plus-tail program, ACTUALLY EXECUTED on this machine's own real
AVX2/FMA hardware, matches evaluateArrays() exactly, element for element (confirmed)
```

**Output (device, aarch64 Apple Silicon, NEON):**

```text
=== Section 19.1: from threads to lanes -- a portable SIMD elementwise kernel generator ===

Fused "out" node: 3 steps, 2 external inputs (a:[11], b: scalar)

self-check: canVectorizeElementwise() confirms this node is in scope -- every external
input is either the full output count (11) or a bare scalar (confirmed)

--- What canVectorizeElementwise() correctly REFUSES ---

a:[3,4] (12 elements), b:[4] (4 elements) -- a genuine trailing-suffix broadcast, the
same case Chapter 6's own diamond graph uses. canVectorizeElementwise() returns false --
self-check: this node is correctly identified as OUT OF SCOPE for this section's own
vector path (confirmed) -- a real compiler would fall back to Chapter 17/18's own scalar
generateCpuElementwiseFunction() for exactly this node, not miscompile it.

--- Generated vector body, Isa::Avx2 (8 lanes/instruction) ---

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    long long i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v0 = _mm256_add_ps(_mm256_loadu_ps(ext[0] + i), _mm256_set1_ps(ext[1][0]));
        __m256 v1 = _mm256_max_ps(v0, _mm256_setzero_ps());
        __m256 v2 = _mm256_mul_ps(v1, _mm256_set1_ps(ext[1][0]));
        _mm256_storeu_ps(out + i, v2);
    }
    for (; i < n; ++i) {
        float step0 = ext[0][(i) % extCounts[0]] + ext[1][(i) % extCounts[1]];
        float step1 = std::max(0.0f, step0);
        float step2 = step1 * ext[1][(i) % extCounts[1]];
        out[i] = step2;
    }
}

--- Generated vector body, Isa::Neon (4 lanes/instruction) ---

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    long long i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v0 = vaddq_f32(vld1q_f32(ext[0] + i), vdupq_n_f32(ext[1][0]));
        float32x4_t v1 = vmaxq_f32(v0, vdupq_n_f32(0.0f));
        float32x4_t v2 = vmulq_f32(v1, vdupq_n_f32(ext[1][0]));
        vst1q_f32(out + i, v2);
    }
    for (; i < n; ++i) {
        float step0 = ext[0][(i) % extCounts[0]] + ext[1][(i) % extCounts[1]];
        float step1 = std::max(0.0f, step0);
        float step2 = step1 * ext[1][(i) % extCounts[1]];
        out[i] = step2;
    }
}

self-check: same step sequence, same scalar-tail logic (reused from Chapter 18's own
emitSteps() unchanged) -- only the intrinsic names, the vector width, and the type
(__m256 vs. float32x4_t) differ between the two generated bodies (confirmed by reading
both texts above)

--- This machine compiles as NEON -- generating, compiling, and RUNNING that path for real ---

n=11, vector width=4 -> 2 full vector iteration(s), 3 scalar tail element(s)

--- g++ compile (real NEON flags) ---

(no output -- clean compile)

self-check: this machine's own real compiler accepts the generated NEON intrinsics (confirmed)

--- Running the compiled binary FOR REAL on this machine's own hardware ---

6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000 22.000000 0.000000 26.000000

evaluateArrays(fused).out = 6.000000 0.000000 10.000000 0.000000 14.000000 0.000000 18.000000 0.000000 22.000000 0.000000 26.000000
self-check: the vector-plus-tail program, ACTUALLY EXECUTED on this machine's own real
NEON hardware, matches evaluateArrays() exactly, element for element (confirmed)
```


!!! note "Why ext[index], not a named ext0/ext1 parameter"
    An earlier draft of `vecLoadOrBroadcast()` generated CUDA-style named-pointer references (`ext0[i]`, `ext1[0]`) -- a direct copy of Chapter 18's own naming convention, applied without re-deriving WHY that convention existed. It failed to compile: the enclosing function's own signature takes a single `const std::vector<const float*>& ext` parameter, the CPU convention Chapter 17 and 18 both already used for scalar CPU code, and no `ext0`/`ext1` variable is ever declared. The fix is not a workaround -- it is the more correct design for this backend: Chapter 18's own named-parameter requirement existed specifically because a `__global__` CUDA kernel cannot dereference a host-memory `std::vector`'s own buffer, a real host/device address-space distinction that simply does not apply to plain CPU code. Indexing into the vector (`ext[0] + i`, valid pointer arithmetic on a `const float*`) is both simpler to generate and correctly reflects that this backend has no such split.

## 19.2 Fusing Multiply and Add: A Real Peephole Optimization, Vectorized

### Intuition

Section 19.1's own generator emits one vector instruction per `FusedStep`, unconditionally -- structurally correct, but real hardware offers something genuinely cheaper for one very common pattern: a multiply immediately followed by an add that consumes it. Both of this book's own real architectures have a single instruction for exactly that -- AVX2/FMA's `_mm256_fmadd_ps` and NEON's `vfmaq_f32` each compute `a*b + c` in ONE instruction instead of two -- and this section adds a genuine peephole optimization that recognizes a `Mul` step directly followed by an `Add` step consuming it, and emits one fused instruction instead of two. This is not a cosmetic rewrite: the generated code for a folded pattern is measurably shorter, one vector instruction where there were two, and it is a real, checkable instance of the same "recognize a pattern, emit less code" idea Chapter 11's own algebraic simplification pass and Chapter 13's own elementwise fusion already both practiced, now applied at the instruction-selection level instead of the graph level.

### Background

The one genuine obstacle to this fold is SAFETY: folding `Mul` step `M` into `Add` step `A` only makes sense if `M`'s own separately-computed value is never needed anywhere ELSE -- if some other step also read `M`'s own result, this fold would still have to compute it as its own standalone value, and the fold would save nothing. This section proves that condition holds not by constructing an artificial unsafe case and defending against it, but by tracing it back to an invariant Chapter 13 already established, five chapters earlier: `resolveIntoGroup()`'s own `mustBeExternal` check includes `consumers.at(oldId) != 1` -- any node read by more than one other node in the ORIGINAL graph is always externalized, forced OUT of the fused group entirely, before it could ever be shared by two steps within one group's own step list. The consequence is exactly the safety condition this fold needs: within any ONE fused group, every step except the group's own final output is read by AT MOST one later step, guaranteed by logic that already existed for an entirely different reason. `computeStepUseCounts()` exists to make that invariant CHECKABLE in code -- counting how many times each step is actually referenced -- not because the fold needs to defend against a case that can actually arise from this book's own fusion passes; its own use-count check (`useCounts[s-1] != 1`) always evaluates to true here, confirmed rather than assumed.

`vecFma()` is where the AVX2-versus-NEON argument-order difference gets hidden: one function, one call site, meaning "compute `mulLhs * mulRhs + addend`" -- and two genuinely different generated instructions underneath, `_mm256_fmadd_ps(mulLhs, mulRhs, addend)` on AVX2 (multiplicands first, accumulator last) and `vfmaq_f32(addend, mulLhs, mulRhs)` on NEON (accumulator first, multiplicands last). `generateVectorElementwiseFunctionFma()` runs in two passes over a group's own steps: the first pass decides which `Mul` steps fold into the `Add` immediately after them (four conditions, all checked, not assumed -- the step is an `Add`; it is not the first step; the immediately preceding step is a `Mul`; that `Mul`'s own use count is exactly 1 and the `Add` genuinely references it as one of its own operands), and the second pass emits code, skipping every folded-away `Mul`'s own line entirely and emitting one `vecFma()` call for the `Add` that absorbed it. Two small test graphs prove the fold fires exactly when it should and nowhere else: a genuine `Mul`-then-`Add` pattern (`t1 = Mul(a,b)`, `out = Add(t1,c)`) folds to exactly one instruction, while a `ReLU`-then-`Add` pattern -- no `Mul` anywhere in it -- correctly produces zero folds, since the fold rule only ever fires when the step immediately before an `Add` is specifically a `Mul`.

```text
_mm256_fmadd_ps(a, b, c) = a*b + c        vfmaq_f32(acc, a, b) = acc + a*b

  SAME three operands, meaning "multiply a by b, add c" either way --
  DIFFERENT argument position for the accumulator (last on AVX2, first on
  NEON). vecFma(isa, mulLhs, mulRhs, addend) hides this behind one call.

Two raw steps -> one folded instruction (t1=Mul(a,b), out=Add(t1,c)):

  WITHOUT the fold (Section 19.1's own generator):
    v0 = vecMul(a, b)              two vector instructions
    v1 = vecAdd(v0, c)

  WITH the fold (this section):
    v1 = vecFma(a, b, c)           ONE vector instruction -- v0's own line
                                    is never separately emitted at all

  The pattern that correctly does NOT fold (r1=ReLU(a), out2=Add(r1,b)):
    v0 = vecMax(a, zero)           ReLU, not Mul -- the fold rule's own
    v1 = vecAdd(v0, b)             pattern-match (previous step == Mul)
                                    never matches, 0 folds, both steps kept

Why the fold is ALWAYS safe here (not just in this example):

  Chapter 13's resolveIntoGroup():  mustBeExternal includes
                                     consumers.at(oldId) != 1

  -> any node read by >1 consumer is forced OUT of the group before fusion
  -> within ONE group's own steps, every non-final step has EXACTLY 1 reader
  -> a folded Mul step's own value is, by construction, never needed
     anywhere else -- proven by an invariant that already existed for
     shape-safety reasons, not discovered by searching for counterexamples
```

```cpp
// Chapter 19: Generating Vectorized CPU Code
// 046_fusing_multiply_and_add_a_real_peephole_optimization_vectorized.cpp
//
// Section 19.2 -- Section 19.1's own vector codegen emits one instruction
// per FusedStep, unconditionally. But real hardware -- both of this book's
// own real targets -- offers a genuinely cheaper instruction for a very
// common pattern: multiply, then immediately add. AVX2/FMA's own
// _mm256_fmadd_ps and NEON's own vfmaq_f32 each compute a*b+c in ONE
// instruction instead of two, and this section adds a real peephole
// optimization to Section 19.1's own generator that finds a Mul step
// immediately followed by an Add step consuming it, and emits ONE fused
// instruction instead of two separate ones -- a genuine, measurable
// instruction-count reduction, not a cosmetic rewrite.
//
// The interesting question a peephole fusion like this always raises is
// SAFETY: folding Mul step M into Add step A only makes sense if M's own
// separately-computed value is never needed anywhere else -- if it were,
// this section would have to keep computing it anyway, and the fold would
// save nothing. This section proves that safety condition holds here not
// by testing it against a constructed counterexample (this book's own
// fusion passes make an actual unsafe case impossible to construct in the
// first place -- explained below), but by tracing it back to an invariant
// Chapter 13 established five chapters ago.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 046_fusing_multiply_and_add_a_real_peephole_optimization_vectorized.cpp -o 046_driver
// Run:     ./046_driver
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
//
// The invariant this whole section leans on lives HERE, unchanged since
// Chapter 13: `mustBeExternal` includes `consumers.at(oldId) != 1`. Any
// node read by more than one other node in the ORIGINAL graph is always
// externalized -- forced OUT of the group, never inlined as a PriorStep --
// before it could ever be shared by two steps within one fused group's own
// step list. The consequence: within any ONE group's own fusedSteps,
// every step except the group's own final output is read by AT MOST one
// later step. That is exactly the safety condition this section's own FMA
// fold needs -- and it comes for free, not from new logic in this file.

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

// ==================== emitSteps() (from Section 18.1, unchanged -- reused for the scalar tail) ====================

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

// ==================== Isa / vector primitives (from Section 19.1, unchanged) ====================

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
// Section 19.1's own canVectorizeElementwise() scope check isn't repeated
// here -- both test graphs below are plain elementwise groups the same
// shape as Section 19.1's own worked example, so there is nothing new to
// gate. It returns unchanged in Section 19.3's capstone, where a graph with
// a genuine broadcast reappears.

// ==================== Section 19.2: the FMA fold ====================

// x86's own _mm256_fmadd_ps(a, b, c) computes a*b + c directly. NEON's own
// vfmaq_f32(acc, a, b) computes acc + a*b -- the SAME three operands, in a
// DIFFERENT argument order (accumulator first). This helper hides that
// difference the same way Chapter 18's own maxExpr() hid std::max vs.
// fmaxf: one call site, one meaning ("mulLhs*mulRhs + addend"), two real
// argument orders underneath.
static std::string vecFma(Isa isa, const std::string& mulLhs, const std::string& mulRhs, const std::string& addend) {
    if (isa == Isa::Avx2) return "_mm256_fmadd_ps(" + mulLhs + ", " + mulRhs + ", " + addend + ")";
    return "vfmaq_f32(" + addend + ", " + mulLhs + ", " + mulRhs + ")";
}

// Counts how many times each step is read by a LATER step (PriorStep) or
// by the group's own final returned value. Given Chapter 13's own
// single-consumer inlining invariant (explained above the fusion pass),
// this always evaluates to exactly 1 for every non-final step in a real
// fused group -- so this function exists to make that invariant CHECKABLE,
// not because the fold below needs to defend against a case that can
// actually arise from this book's own passes.
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
        // the group's own final value (the last step) is its own "use"
        useCounts[stepCount - 1]++;
    }
    return useCounts;
}

static bool stepUsesOperand(const FusedStep& step, OperandKind kind, int index) {
    for (const FusedOperand& o : step.operands) if (o.kind == kind && o.index == index) return true;
    return false;
}

// Extends Section 19.1's own generateVectorElementwiseFunction() with
// exactly one new rule: when step s is Add and its immediately preceding
// step s-1 is a Mul used by NOTHING except step s, emit ONE vecFma() line
// for step s and skip step s-1's own line entirely -- the Mul's own
// operands get read directly into the FMA instead of being materialized
// as their own separate vector variable.
static std::string generateVectorElementwiseFunctionFma(const std::string& funcName, const LoweredNode& lowered,
                                                          Isa isa, const std::vector<bool>& isScalarInput,
                                                          int& foldCount, int& rawStepCount) {
    const std::vector<FusedStep>& steps = lowered.steps;
    std::vector<int> useCounts = computeStepUseCounts(steps, false);
    rawStepCount = static_cast<int>(steps.size());
    foldCount = 0;

    // Pass 1: decide which Mul steps get folded into the Add immediately after them.
    std::vector<bool> foldedAway(steps.size(), false);
    for (size_t s = 0; s < steps.size(); ++s) {
        if (steps[s].op != OpKind::Add || s == 0) continue;
        if (steps[s - 1].op != OpKind::Mul) continue;
        if (useCounts[s - 1] != 1) continue;  // always true here, per the invariant above -- checked, not assumed
        if (!stepUsesOperand(steps[s], OperandKind::PriorStep, static_cast<int>(s - 1))) continue;
        foldedAway[s - 1] = true;
        foldCount++;
    }

    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
                       "const std::vector<long long>& extCounts, float* out, long long n) {\n";
    int vw = vectorWidth(isa);
    src += "    long long i = 0;\n";
    src += "    for (; i + " + std::to_string(vw) + " <= n; i += " + std::to_string(vw) + ") {\n";

    std::vector<std::string> stepVars(steps.size());
    auto operandExpr = [&](const FusedOperand& o) -> std::string {
        if (o.kind == OperandKind::ExternalInput) return vecLoadOrBroadcast(isa, o.index, "i", isScalarInput[o.index]);
        return stepVars[static_cast<size_t>(o.index)];
    };
    // Pass 2: emit.
    for (size_t s = 0; s < steps.size(); ++s) {
        if (foldedAway[s]) continue;  // this Mul's own line is skipped -- folded into the Add right after it
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
    printf("=== Section 19.2: fusing multiply and add -- a real peephole optimization, vectorized ===\n\n");

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    // t1 = Mul(a, b), out = Add(t1, c) -- a genuine multiply-then-add
    // pattern. n=9: one full vector iteration on either ISA (8 or 4 wide)
    // plus a real remainder, same discipline as Section 19.1.
    Graph g;
    Value a = g.addInput("a");
    Value b = g.addInput("b");
    Value c = g.addInput("c");
    Value t1 = g.addBinary(OpKind::Mul, a, b, "t1");
    Value out = g.addBinary(OpKind::Add, t1, c, "out");
    (void)out;
    std::map<int, Shape> declared = {{a.nodeId, Shape{{9}}}, {b.nodeId, Shape{}}, {c.nodeId, Shape{}}};
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
    printf("Fused \"out\" node: %zu raw steps (Mul, then Add) -- a fold candidate\n\n", outFusedNode->fusedSteps.size());

    LoweredNode outLowered = lowerNode(outFusedNode, fusedShapes, fusedElementCounts);
    std::vector<bool> isScalarInput;
    for (const Value& in : outFusedNode->inputs) isScalarInput.push_back(fusedElementCounts.at(in.nodeId) == 1);

    int foldCount = 0, rawStepCount = 0;
    std::string foldedSrc = generateVectorElementwiseFunctionFma("compute_out", outLowered, hostIsa, isScalarInput,
                                                                    foldCount, rawStepCount);
    printf("--- Generated vector body WITH the FMA fold (%s) ---\n\n%s\n", isaName(hostIsa).c_str(), foldedSrc.c_str());
    printf("self-check: %d raw step(s) (Mul, Add) generate exactly 1 vector instruction inside the\n", rawStepCount);
    printf("main loop -- %d fold(s) applied, the Mul's own line never separately emitted (%s)\n\n",
           foldCount, foldCount == 1 ? "confirmed" : "MISMATCH");

    // ---- A pattern that should NOT fold: ReLU immediately before Add (no Mul involved) ----
    Graph g2;
    Value a2 = g2.addInput("a");
    Value b2 = g2.addInput("b");
    Value r1 = g2.addUnary(OpKind::ReLU, a2, "r1");
    Value out2 = g2.addBinary(OpKind::Add, r1, b2, "out2");
    (void)out2;
    std::map<int, Shape> declared2 = {{a2.nodeId, Shape{{9}}}, {b2.nodeId, Shape{}}};
    std::map<int, Shape> shapes2 = inferShapes(g2, declared2);
    std::map<int, long long> elementCounts2;
    for (const auto& kv : shapes2) elementCounts2[kv.first] = numElements(kv.second);
    FusionResult fusedResult2 = reductionFusionPass(g2, {});
    std::map<int, Shape> fusedShapes2;
    std::map<int, long long> fusedElementCounts2;
    for (const auto& n : fusedResult2.graph.nodes()) {
        int oldId = fusedResult2.representativeOldId.at(n->id);
        fusedShapes2[n->id] = shapes2.at(oldId);
        fusedElementCounts2[n->id] = elementCounts2.at(oldId);
    }
    const Node* out2FusedNode = nullptr;
    for (const auto& n : fusedResult2.graph.nodes()) if (n->debugName == "out2") out2FusedNode = n.get();
    LoweredNode out2Lowered = lowerNode(out2FusedNode, fusedShapes2, fusedElementCounts2);
    std::vector<bool> isScalarInput2;
    for (const Value& in : out2FusedNode->inputs) isScalarInput2.push_back(fusedElementCounts2.at(in.nodeId) == 1);
    int foldCount2 = 0, rawStepCount2 = 0;
    generateVectorElementwiseFunctionFma("compute_out2", out2Lowered, hostIsa, isScalarInput2, foldCount2, rawStepCount2);
    printf("--- A pattern that correctly does NOT fold: ReLU(a) then Add(that, b) -- no Mul at all ---\n\n");
    printf("self-check: %d raw step(s) (ReLU, Add), %d fold(s) applied -- the fold rule only ever\n",
           rawStepCount2, foldCount2);
    printf("fires on an Add whose immediately preceding step is a Mul, and correctly leaves this\n");
    printf("ReLU-then-Add pattern alone (%s)\n\n", foldCount2 == 0 ? "confirmed" : "MISMATCH");

    // ---- Compile and run the FOLDED version for real, on this host ----
    std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> bVals = {2};
    std::vector<float> cVals = {10};
    std::map<std::string, std::vector<float>> in = {{"a", aVals}, {"b", bVals}, {"c", cVals}};
    auto fusedArrays = evaluateArrays(fused, in, fusedElementCounts);
    long long n = fusedElementCounts.at(outFusedNode->id);

    auto formatArrayLiteral = [](const std::vector<float>& v) {
        std::string out = "{";
        for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
        out += "}";
        return out;
    };
    std::string prog = std::string(isaHeaderInclude(hostIsa)) + "\n#include <cstdio>\n#include <vector>\n#include <algorithm>\n\n";
    prog += foldedSrc + "\n";
    prog += "int main() {\n";
    prog += "    std::vector<float> ext0 = " + formatArrayLiteral(aVals) + ";\n";
    prog += "    std::vector<float> ext1 = " + formatArrayLiteral(bVals) + ";\n";
    prog += "    std::vector<float> ext2 = " + formatArrayLiteral(cVals) + ";\n";
    prog += "    std::vector<const float*> ext = {ext0.data(), ext1.data(), ext2.data()};\n";
    prog += "    std::vector<long long> extCounts = {" + std::to_string(aVals.size()) + ", " +
            std::to_string(bVals.size()) + ", " + std::to_string(cVals.size()) + "};\n";
    prog += "    std::vector<float> out(" + std::to_string(n) + ");\n";
    prog += "    compute_out(ext, extCounts, out.data(), " + std::to_string(n) + ");\n";
    prog += "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";

    writeFile("/tmp/hammer_ch19_046.cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags +
                                                  " /tmp/hammer_ch19_046.cpp -o /tmp/hammer_ch19_046 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags, folded FMA body) ---\n\n%s\n", isaName(hostIsa).c_str(),
           compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());

    std::string runOutput = runShellCaptureAll("/tmp/hammer_ch19_046");
    printf("--- Running the folded FMA program FOR REAL on this machine's own %s hardware ---\n\n%s\n",
           isaName(hostIsa).c_str(), runOutput.c_str());

    std::istringstream iss(runOutput);
    std::vector<float> vecVals;
    float v;
    while (iss >> v) vecVals.push_back(v);
    const std::vector<float>& expected = fusedArrays.at("out");
    bool matches = (vecVals.size() == expected.size());
    if (matches) for (size_t i = 0; i < vecVals.size(); ++i) if (std::fabs(vecVals[i] - expected[i]) > 1e-3f) matches = false;
    printf("evaluateArrays(fused).out = ");
    for (float ev : expected) printf("%.6f ", ev);
    printf("\n");
    printf("self-check: a single fmadd/vfma instruction computes the identical result an unfused\n");
    printf("mul+add pair would -- confirmed against evaluateArrays() (%s)\n", matches ? "confirmed" : "MISMATCH");

    bool allOk = compileClean && matches && (foldCount == 1) && (foldCount2 == 0);
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "046_fusing_multiply_and_add_a_real_peephole_optimization_vectorized.cpp" -o "046_fusing_multiply_and_add_a_real_peephole_optimization_vectorized"
./"046_fusing_multiply_and_add_a_real_peephole_optimization_vectorized"
```

**Output (cloud sandbox, x86-64, AVX2/FMA):**

```text
=== Section 19.2: fusing multiply and add -- a real peephole optimization, vectorized ===

Fused "out" node: 2 raw steps (Mul, then Add) -- a fold candidate

--- Generated vector body WITH the FMA fold (AVX2/FMA) ---

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    long long i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v1 = _mm256_fmadd_ps(_mm256_loadu_ps(ext[0] + i), _mm256_set1_ps(ext[1][0]), _mm256_set1_ps(ext[2][0]));  // fused: v0 folded in
        _mm256_storeu_ps(out + i, v1);
    }
    for (; i < n; ++i) {
        float step0 = ext[0][(i) % extCounts[0]] * ext[1][(i) % extCounts[1]];
        float step1 = step0 + ext[2][(i) % extCounts[2]];
        out[i] = step1;
    }
}

self-check: 2 raw step(s) (Mul, Add) generate exactly 1 vector instruction inside the
main loop -- 1 fold(s) applied, the Mul's own line never separately emitted (confirmed)

--- A pattern that correctly does NOT fold: ReLU(a) then Add(that, b) -- no Mul at all ---

self-check: 2 raw step(s) (ReLU, Add), 0 fold(s) applied -- the fold rule only ever
fires on an Add whose immediately preceding step is a Mul, and correctly leaves this
ReLU-then-Add pattern alone (confirmed)

--- g++ compile (real AVX2/FMA flags, folded FMA body) ---

(no output -- clean compile)

--- Running the folded FMA program FOR REAL on this machine's own AVX2/FMA hardware ---

12.000000 14.000000 16.000000 18.000000 20.000000 22.000000 24.000000 26.000000 28.000000 

evaluateArrays(fused).out = 12.000000 14.000000 16.000000 18.000000 20.000000 22.000000 24.000000 26.000000 28.000000 
self-check: a single fmadd/vfma instruction computes the identical result an unfused
mul+add pair would -- confirmed against evaluateArrays() (confirmed)
```

**Output (device, aarch64 Apple Silicon, NEON):**

```text
=== Section 19.2: fusing multiply and add -- a real peephole optimization, vectorized ===

Fused "out" node: 2 raw steps (Mul, then Add) -- a fold candidate

--- Generated vector body WITH the FMA fold (NEON) ---

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out, long long n) {
    long long i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v1 = vfmaq_f32(vdupq_n_f32(ext[2][0]), vld1q_f32(ext[0] + i), vdupq_n_f32(ext[1][0]));  // fused: v0 folded in
        vst1q_f32(out + i, v1);
    }
    for (; i < n; ++i) {
        float step0 = ext[0][(i) % extCounts[0]] * ext[1][(i) % extCounts[1]];
        float step1 = step0 + ext[2][(i) % extCounts[2]];
        out[i] = step1;
    }
}

self-check: 2 raw step(s) (Mul, Add) generate exactly 1 vector instruction inside the
main loop -- 1 fold(s) applied, the Mul's own line never separately emitted (confirmed)

--- A pattern that correctly does NOT fold: ReLU(a) then Add(that, b) -- no Mul at all ---

self-check: 2 raw step(s) (ReLU, Add), 0 fold(s) applied -- the fold rule only ever
fires on an Add whose immediately preceding step is a Mul, and correctly leaves this
ReLU-then-Add pattern alone (confirmed)

--- g++ compile (real NEON flags, folded FMA body) ---

(no output -- clean compile)

--- Running the folded FMA program FOR REAL on this machine's own NEON hardware ---

12.000000 14.000000 16.000000 18.000000 20.000000 22.000000 24.000000 26.000000 28.000000

evaluateArrays(fused).out = 12.000000 14.000000 16.000000 18.000000 20.000000 22.000000 24.000000 26.000000 28.000000
self-check: a single fmadd/vfma instruction computes the identical result an unfused
mul+add pair would -- confirmed against evaluateArrays() (confirmed)
```


!!! warning "[COMMON TRAP] assuming FMA intrinsics share one argument order"
    `_mm256_fmadd_ps(a, b, c)` and `vfmaq_f32(acc, a, b)` compute the exact same mathematical result -- `a*b + c` either way -- but a direct, unexamined port of one architecture's own call to the other silently swaps which value ends up multiplied and which ends up added. Reading NEON's own intrinsic reference is what actually caught this: `vfmaq_f32` documents its first parameter as the value being accumulated INTO, not a third multiplicand appended at the end the way AVX2's own signature reads. `vecFma()` exists specifically so every call site in this chapter spells the fold the same way -- `vecFma(isa, mulLhs, mulRhs, addend)` -- regardless of which architecture is generating code, with the argument reordering handled in exactly one place.

## 19.3 A Whole Graph, Vectorized: Dispatch, Reduction, and a Real Fallback

### Intuition

Sections 19.1 and 19.2 each vectorized ONE node, chosen by hand, in isolation. A real backend has to look at an ENTIRE fused graph and decide, node by node, which of three things to generate: the FMA-fold-aware vector body from Section 19.2, a genuinely new VECTORIZED REDUCTION body this section adds, or -- for a node whose own shapes fall outside `canVectorizeElementwise()`'s own stated scope -- Chapter 18's own plain scalar body, generated, compiled, and run for real rather than merely described as a fallback that would theoretically work. This section builds that dispatcher, proves the fallback path is genuinely correct (not just "not vectorized") by compiling and running a real broadcast case through it, and then runs Chapter 16's own 10-node capstone graph -- the SAME graph Section 17.3 already made real on a CPU loop nest and Section 18.3 already made real (as far as compilation goes) on a GPU -- through the dispatcher end to end, compiling and EXECUTING the whole thing for real on both of this book's own real architectures.

### Background

A `FusedReduction` group's own steps are, structurally, the exact same per-element step sequence Sections 19.1 and 19.2 already know how to vectorize, plus one trailing `Sum` step that folds every one of those per-element RESULTS down to a single scalar. Vectorizing it needs exactly one new idea: accumulate each lane's own running total into a VECTOR accumulator across the main loop -- `acc = vecAdd(acc, this iteration's own vector result)` -- instead of storing each lane's own result separately, and only collapse that accumulator down to one float, a "horizontal sum," once, after the loop ends, not once per element. `generateVectorReductionFunction()`'s own horizontal sum uses the simplest technique genuinely portable across both architectures: store the accumulator's own lanes out to a small array (`vecStore()`, the same primitive Section 19.1 already built) and add them up with a short scalar loop. NEON offers a real, faster alternative here -- `vaddvq_f32(acc)` sums all four lanes in a single instruction -- but AVX2 has no equivalent single instruction that reduces all eight of its own lanes without reaching for AVX-512, so this section deliberately uses the slower-but-uniform array-extraction approach on BOTH backends rather than special-casing NEON's own faster path; a production compiler would dispatch to `vaddvq_f32` on NEON and keep array-extraction only where AVX2 needs it. Chapter 16's own capstone graph, once fused with `boundedReductionFusionPass(maxChainLength=3)`, happens to produce a reduction node (`s`) whose own `Sum` step reads directly from an external input with no per-element steps ahead of it -- the simplest possible case for this new machinery, but it is the SAME code path a longer reduce chain would take.

`generateFunctionForNode()` is the whole dispatcher, and it really is this small: a reduction node (`Sum` or `FusedReduction`) always takes the new vectorized-reduction path; anything else takes Section 19.2's own FMA-fold-aware vector path when `canVectorizeElementwise()` allows it, and Chapter 18's own plain scalar path otherwise. Before running it against a real graph, this section proves the scalar branch is not a theoretical safety net by constructing a genuine trailing-dimension broadcast (`a:[3,4]`, `b:[4]`, `out = Add(a,b)` -- 12 elements against 4, matching neither the output count nor a scalar) and running it all the way through: `canVectorizeElementwise()` correctly returns false, the dispatcher correctly chooses `SCALAR-FALLBACK`, and the generated scalar function is compiled and run for real, matching `evaluateArrays()` exactly -- "not vectorized" and "wrong" are different things, and this proves it concretely rather than leaving it asserted. The dispatcher then runs against Chapter 16's own 10-node graph, fused the same way Section 17.3 and 18.3 both fused it (`maxChainLength=3`, six materialized nodes: `t3` with 3 raw steps, `t5` with 2, the reduction `s` with 1, and `y` with 2): every one of the four non-leaf nodes takes a VECTOR path (three FMA-fold elementwise groups plus the one vectorized reduction, zero scalar fallbacks, since nothing in this particular graph's own shapes falls outside this chapter's scope), three separate FMA folds fire (inside `t3`, `t5`, and `y`, one each), and the whole generated program -- one real vector function per node, chained through named buffers exactly like Chapter 18's own CPU driver -- compiles and RUNS on both architectures, reporting the same `y[0] = 132.0` Chapter 16, 17, and 18 have all already established, this time computed by real vector instructions on real hardware rather than a serial loop or an honestly-uncheckable GPU kernel.

```text
Vectorized reduction: a running vector accumulator, collapsed ONCE

  acc = zero-vector
  i = 0
  main loop (width w): acc = vecAdd(acc, this iteration's own vector value)
                        i += w
  (loop ends -- accumulator holds w PARTIAL sums, not the final answer yet)

  horizontal sum (portable, both architectures):
    lanes[w] = vecStore(acc)             extract every lane to an array
    hsum = lanes[0] + lanes[1] + ... + lanes[w-1]   plain scalar addition

  scalar tail (Chapter 18's own emitSteps(), isReduction=true, unchanged):
    for remaining i: hsum += this element's own value

  out[0] = hsum

  (NEON's own vaddvq_f32(acc) would do the horizontal sum in ONE
  instruction instead of a w-way scalar loop -- named here as a real,
  faster alternative, not implemented, since AVX2 has no single-instruction
  equivalent without AVX-512 -- the same "correct, not fastest" honesty
  Chapter 18's own atomicAdd() discussion already practiced)

Chapter 16's own 10-node graph, fused (maxChainLength=3), dispatched:

  a       b
  |       |
  +---+---+
      |
  t1,t2,t3 fuse (SizeCap boundary) into t3   -> VECTOR+FMA (1 fold: the
      |                                          trailing Mul+Add inside
  t4,t5 fuse separately into t5              -> VECTOR+FMA  its own 3 steps)
      |                                         (1 fold)
  +---+---+
  |       |
  s       y2
  (FusedReduction)                           -> VECTOR-REDUCTION (new
  |       |                                      this section)
  +---+---+
      |
      y                                      -> VECTOR+FMA (1 fold: s + t5*b)

  4 nodes, 4 vector functions, 0 scalar fallbacks, 3 total FMA folds --
  compiled and RUN FOR REAL on both architectures, reporting y[0]=132.0
```

```cpp
// Chapter 19: Generating Vectorized CPU Code
// 047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback.cpp
//
// Section 19.3 -- the capstone. Sections 19.1 and 19.2 each vectorized ONE
// node at a time, chosen by hand. A real backend has to look at an entire
// fused graph and decide, node by node, which of three things to generate:
// the FMA-fold-aware vector elementwise body (19.1 + 19.2), a genuinely new
// vectorized REDUCTION body this section adds, or -- when a node's own
// shapes don't fit this backend's scope, exactly as canVectorizeElementwise()
// already checks for -- the plain scalar body Chapter 18 already knows how
// to generate. This section builds that per-node dispatch, proves the
// fallback path is real (not just a returned "false") by compiling and
// running a genuinely non-vectorizable broadcast case, and then runs
// Chapter 16's own 10-node capstone graph -- the SAME graph Section 17.3
// and Section 18.3 already made real, on a CPU loop nest and then on a GPU
// -- through the dispatcher, compiling and executing the WHOLE thing for
// real on both of this book's own real architectures (AVX2/FMA and NEON).
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback.cpp -o 047_driver
// Run:     ./047_driver
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

// ==================== emitSteps() (from Section 18.1, unchanged -- the scalar backend, still used for
//                      every scalar tail loop AND for the whole-function scalar fallback) ====================

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

// ---- Section 18.1's own scalar elementwise generator: the fallback path
// for any node canVectorizeElementwise() below rejects. ----
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

// ==================== Isa / vector primitives (from Section 19.1, unchanged) ====================

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

// This section is where canVectorizeElementwise() finally earns its keep:
// every node in this file's own dispatcher runs through it for real, and a
// node it rejects gets the scalar fallback above -- generated, compiled,
// and run, not just described.
static bool canVectorizeElementwise(const Node* n, const std::map<int, long long>& elementCounts) {
    long long outCount = elementCounts.at(n->id);
    for (const Value& in : n->inputs) {
        long long c = elementCounts.at(in.nodeId);
        if (c != outCount && c != 1) return false;
    }
    return true;
}

// ==================== Section 19.2's own FMA fold (unchanged) ====================

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
static std::string generateVectorElementwiseFunctionFma(const std::string& funcName, const LoweredNode& lowered,
                                                          Isa isa, const std::vector<bool>& isScalarInput,
                                                          int& foldCount, int& rawStepCount) {
    const std::vector<FusedStep>& steps = lowered.steps;
    std::vector<int> useCounts = computeStepUseCounts(steps, false);
    rawStepCount = static_cast<int>(steps.size());
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

    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
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

// ==================== Section 19.3: vectorized reduction ====================
//
// A FusedReduction group's own steps are, structurally, the SAME per-element
// step sequence Section 19.1/19.2 already vectorize, plus one trailing Sum
// step that folds every one of those per-element RESULTS into a single
// scalar. Vectorizing it needs exactly one new idea: accumulate each lane's
// own running total into a VECTOR accumulator across the main loop, instead
// of storing per-lane results -- and only collapse that accumulator down to
// one float ("horizontal sum") once, after the loop ends, not once per
// element.
//
// The horizontal sum below uses the simplest technique that is genuinely
// portable across BOTH of this book's own real ISAs: store the
// accumulator's own lanes out to a small array and add them up in plain
// scalar code. NEON offers a real single-instruction alternative here --
// vaddvq_f32(acc) sums all four lanes in one instruction -- but AVX2 has no
// equivalent single instruction that reduces all eight lanes without
// reaching for AVX-512, so this section deliberately uses the
// slower-but-uniform array-extraction approach on BOTH backends rather than
// special-casing NEON's own faster path. A production compiler would
// dispatch to vaddvq_f32 on NEON and keep the array-extraction fallback
// only for AVX2.
static std::string generateVectorReductionFunction(const std::string& funcName, const LoweredNode& lowered,
                                                     Isa isa, const std::vector<bool>& isScalarInput) {
    const std::vector<FusedStep>& steps = lowered.steps;
    size_t stepCount = steps.size() - 1;  // every step except the trailing Sum
    long long reduceExtent = lowered.nest.loops.back().extent;
    int vw = vectorWidth(isa);

    std::string src = "void " + funcName + "(const std::vector<const float*>& ext, "
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

// ==================== Section 19.3: the per-node dispatcher ====================
//
// One function decides, per node, which of three generators produced its
// body: a reduction node always takes the vectorized-reduction path above;
// an elementwise node takes the FMA-fold-aware vector path from Section
// 19.2 when canVectorizeElementwise() allows it, and Chapter 18's own plain
// scalar path otherwise. No other logic -- this IS the whole dispatcher.
static bool isReductionNode(const Node* n) { return n->op == OpKind::Sum || n->op == OpKind::FusedReduction; }

enum class Backend { VectorFma, VectorReduction, ScalarFallback };

static std::string generateFunctionForNode(const std::string& funcName, const Node* n, const LoweredNode& lowered,
                                            Isa isa, const std::map<int, long long>& elementCounts,
                                            Backend& backendUsed, int& foldCount) {
    foldCount = 0;
    std::vector<bool> isScalarInput;
    for (const Value& in : n->inputs) isScalarInput.push_back(elementCounts.at(in.nodeId) == 1);

    if (isReductionNode(n)) {
        backendUsed = Backend::VectorReduction;
        return generateVectorReductionFunction(funcName, lowered, isa, isScalarInput);
    }
    if (canVectorizeElementwise(n, elementCounts)) {
        backendUsed = Backend::VectorFma;
        int rawStepCount = 0;
        return generateVectorElementwiseFunctionFma(funcName, lowered, isa, isScalarInput, foldCount, rawStepCount);
    }
    backendUsed = Backend::ScalarFallback;
    return generateCpuElementwiseFunction(funcName, lowered);
}
static const char* backendName(Backend b) {
    return b == Backend::VectorFma ? "VECTOR+FMA" : b == Backend::VectorReduction ? "VECTOR-REDUCTION" : "SCALAR-FALLBACK";
}

// ==================== Whole-graph driver, dispatched per node (mirrors Section 18.3's own
//                      generateFullCpuProgram(), with generateFunctionForNode() in place of
//                      the always-scalar generateCpuFunctionForNode()) ====================

static std::string formatLiteralArray(const std::vector<float>& v) {
    std::string out = "{";
    for (size_t i = 0; i < v.size(); ++i) { if (i) out += ", "; out += std::to_string(v[i]) + "f"; }
    out += "}";
    return out;
}

static std::string generateFullVectorizedProgram(const Graph& g, const std::map<int, Shape>& shapes,
                                                   const std::map<int, long long>& elementCounts,
                                                   const std::map<std::string, std::vector<float>>& inputArrays,
                                                   Isa isa, std::vector<std::pair<std::string, Backend>>& dispatchLog,
                                                   int& totalFolds) {
    TopoResult topo = topologicalSort(g);
    std::string prog = std::string(isaHeaderInclude(isa)) +
                        "\n#include <cstdio>\n#include <vector>\n#include <map>\n#include <string>\n#include <algorithm>\n\n";

    std::map<int, std::string> funcNameById;
    totalFolds = 0;
    for (int id : topo.order) {
        const Node* n = g.node(id);
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        std::string funcName = "compute_" + n->debugName;
        funcNameById[id] = funcName;
        LoweredNode lowered = lowerNode(n, shapes, elementCounts);
        Backend backendUsed; int foldCount = 0;
        prog += generateFunctionForNode(funcName, n, lowered, isa, elementCounts, backendUsed, foldCount) + "\n";
        dispatchLog.push_back({n->debugName, backendUsed});
        totalFolds += foldCount;
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
static bool arraysMatch(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-2f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

int main() {
    printf("=== Section 19.3: a whole graph, vectorized dispatch, reduction, and a real fallback ===\n\n");

#if defined(__x86_64__)
    Isa hostIsa = Isa::Avx2;
    std::string extraFlags = " -mavx2 -mfma";
#elif defined(__aarch64__)
    Isa hostIsa = Isa::Neon;
    std::string extraFlags = "";
#else
#error "targets only x86_64 (AVX2/FMA) and aarch64 (NEON)."
#endif

    bool allOk = true;

    // ---------------------------------------------------------------
    // Part 1: prove the fallback is REAL. a genuine trailing-dimension
    // broadcast -- a:[3,4], b:[4], out = Add(a,b) -- is exactly the shape
    // canVectorizeElementwise() was always documented to reject (count(a)=12,
    // count(b)=4: neither equals the other nor is it a scalar). This isn't
    // a hypothetical: the dispatcher below hits it, chooses the scalar
    // path, and that scalar path is compiled and run for real, matching
    // evaluateArrays() exactly -- "not vectorized" and "wrong" are
    // different things, and this proves it.
    // ---------------------------------------------------------------
    printf("--- Part 1: a genuine broadcast the vector path can't safely handle ---\n\n");
    {
        Graph gb;
        Value ba = gb.addInput("a");
        Value bb = gb.addInput("b");
        Value bout = gb.addBinary(OpKind::Add, ba, bb, "out");
        std::map<int, Shape> declaredB = {{ba.nodeId, Shape{{3, 4}}}, {bb.nodeId, Shape{{4}}}};
        std::map<int, Shape> shapesB = inferShapes(gb, declaredB);
        std::map<int, long long> countsB;
        for (const auto& kv : shapesB) countsB[kv.first] = numElements(kv.second);

        const Node* outNode = gb.node(bout.nodeId);
        bool canVec = canVectorizeElementwise(outNode, countsB);
        printf("a:[3,4] (12 elements), b:[4] (4 elements) -- canVectorizeElementwise(out) = %s\n",
               canVec ? "true" : "false");
        printf("self-check: a real trailing-dimension broadcast is correctly rejected by the vector\n");
        printf("path's own scope check (%s)\n\n", !canVec ? "confirmed" : "MISMATCH");

        LoweredNode boutLowered = lowerNode(outNode, shapesB, countsB);
        Backend backendUsed; int foldCount = 0;
        std::string fnSrc = generateFunctionForNode("compute_out", outNode, boutLowered, hostIsa, countsB, backendUsed, foldCount);
        printf("dispatcher chose: %s\n\n", backendName(backendUsed));

        std::vector<float> aVals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<float> bVals = {100, 200, 300, 400};
        std::map<std::string, std::vector<float>> inB = {{"a", aVals}, {"b", bVals}};
        auto expectedB = evaluateArrays(gb, inB, countsB);

        std::string prog = "#include <cstdio>\n#include <vector>\n\n" + fnSrc +
                            "\nint main() {\n    std::vector<float> ext0 = " + formatLiteralArray(aVals) +
                            ";\n    std::vector<float> ext1 = " + formatLiteralArray(bVals) +
                            ";\n    std::vector<const float*> ext = {ext0.data(), ext1.data()};\n"
                            "    std::vector<long long> extCounts = {12, 4};\n"
                            "    std::vector<float> out(12);\n"
                            "    compute_out(ext, extCounts, out.data(), 12);\n"
                            "    for (float v : out) printf(\"%.6f \", v);\n    printf(\"\\n\");\n    return 0;\n}\n";
        writeFile("/tmp/hammer_ch19_047_fallback.cpp", prog);
        std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2 /tmp/hammer_ch19_047_fallback.cpp "
                                                      "-o /tmp/hammer_ch19_047_fallback 2>&1");
        bool compileClean = compileLog.empty();
        std::string runOutput = runShellCaptureAll("/tmp/hammer_ch19_047_fallback");
        std::istringstream iss(runOutput);
        std::vector<float> got;
        float v;
        while (iss >> v) got.push_back(v);
        bool matches = arraysMatch(got, expectedB.at("out"));
        printf("scalar fallback, compiled and run for real: %s. output %s evaluateArrays() (%s)\n\n",
               compileClean ? "clean compile" : "COMPILE FAILED", matches ? "matches" : "does NOT match",
               (compileClean && matches) ? "confirmed" : "MISMATCH");
        allOk = allOk && compileClean && matches && !canVec && (backendUsed == Backend::ScalarFallback);
    }

    // ---------------------------------------------------------------
    // Part 2: Chapter 16's own 10-node capstone graph, reused verbatim --
    // the same graph Section 17.3 made real on a CPU loop nest and Section
    // 18.3 made real on a GPU. This time every fused node is dispatched
    // through generateFunctionForNode(), compiled, and run for real on
    // THIS machine's own real vector hardware.
    // ---------------------------------------------------------------
    printf("--- Part 2: Chapter 16's 10-node graph, vectorized end to end ---\n\n");
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
    printf("fused graph (maxChainLength=3, same cap Section 17.3/18.3 used): ");
    for (const auto& n : fused.nodes()) {
        printf("%s", n->debugName.c_str());
        if (n->op == OpKind::FusedElementwise) printf("(FE,%zust)", n->fusedSteps.size());
        else if (n->op == OpKind::FusedReduction) printf("(FR,%zust)", n->fusedSteps.size());
        printf(" ");
    }
    printf("\n\n");

    std::vector<std::pair<std::string, Backend>> dispatchLog;
    int totalFolds = 0;
    std::string prog = generateFullVectorizedProgram(fused, fusedShapes, fusedElementCounts, inputArrays,
                                                       hostIsa, dispatchLog, totalFolds);
    printf("per-node dispatch (%s):\n", isaName(hostIsa).c_str());
    for (const auto& entry : dispatchLog) printf("  %-4s -> %s\n", entry.first.c_str(), backendName(entry.second));
    printf("\nself-check: every fused node in this graph -- 3 elementwise groups plus the\n");
    printf("reduction -- is genuinely vectorizable (no broadcast beyond a scalar anywhere\n");
    printf("in this graph), so the dispatcher chose VECTOR paths for all %zu nodes, 0\n", dispatchLog.size());
    printf("scalar fallbacks (%s)\n\n", (dispatchLog.size() > 0 && std::all_of(dispatchLog.begin(), dispatchLog.end(),
           [](const std::pair<std::string, Backend>& e) { return e.second != Backend::ScalarFallback; }))
           ? "confirmed" : "MISMATCH");
    printf("self-check: %d total FMA fold(s) applied across the 3 elementwise groups --\n", totalFolds);
    printf("t3's own ReLU-then-Mul-then-Add folds its trailing Mul+Add, t5's own Mul-then-Add\n");
    printf("folds, and y's own Mul-then-Add (s + t5*b) folds too, for exactly 3 (%s)\n\n",
           totalFolds == 3 ? "confirmed" : "MISMATCH");

    std::string cpuStem = "/tmp/hammer_ch19_047_capstone";
    writeFile(cpuStem + ".cpp", prog);
    std::string compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2" + extraFlags + " " + cpuStem +
                                                  ".cpp -o " + cpuStem + " 2>&1");
    bool compileClean = compileLog.empty();
    printf("--- g++ compile (real %s flags, %zu generated functions) ---\n\n%s\n", isaName(hostIsa).c_str(),
           dispatchLog.size(), compileClean ? "(no output -- clean compile)\n" : compileLog.c_str());

    std::string runOutput = runShellCaptureAll(cpuStem);
    printf("--- Running the whole vectorized graph FOR REAL on this machine's own %s hardware ---\n\n%s\n",
           isaName(hostIsa).c_str(), runOutput.c_str());
    auto generatedArrays = parseNamedBuffers(runOutput);

    bool allNodesMatch = true;
    for (const auto& n : fused.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        bool ok = generatedArrays.count(n->debugName) && arraysMatch(generatedArrays.at(n->debugName),
                                                                       evaluateArrays(fused, inputArrays, fusedElementCounts).at(n->debugName));
        if (!ok) allNodesMatch = false;
    }
    const std::vector<float>& yGenerated = generatedArrays.at("y");
    const std::vector<float>& yExpected = origArrays.at("y");
    bool yMatches = arraysMatch(yGenerated, yExpected);
    printf("evaluateArrays(original).y = ");
    for (float v : yExpected) printf("%.6f ", v);
    printf("\ngenerated (vectorized, %zu functions dispatched, compiled and run for real).y = ", dispatchLog.size());
    for (float v : yGenerated) printf("%.6f ", v);
    printf("\n\n");
    printf("self-check: y[0] = %.6f -- the SAME number Section 17.3's own CPU loop nest and\n", yGenerated.empty() ? 0.0f : yGenerated[0]);
    printf("Section 18.3's own CPU reference both reported for this exact graph, now produced\n");
    printf("by real vector instructions on real %s hardware (%s)\n\n", isaName(hostIsa).c_str(),
           (!yGenerated.empty() && std::fabs(yGenerated[0] - 132.0f) < 1e-2f) ? "confirmed" : "MISMATCH");
    printf("self-check: every generated buffer (not just y) matches evaluateArrays() on the\n");
    printf("SAME fused graph -- the vector path and the interpreter agree at every node, not\n");
    printf("just at the output (%s)\n", allNodesMatch ? "confirmed" : "MISMATCH");

    allOk = allOk && compileClean && yMatches && allNodesMatch && (totalFolds == 3) &&
            std::all_of(dispatchLog.begin(), dispatchLog.end(),
                        [](const std::pair<std::string, Backend>& e) { return e.second != Backend::ScalarFallback; });

    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback.cpp" -o "047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback"
./"047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback"
```

**Output (cloud sandbox, x86-64, AVX2/FMA):**

```text
=== Section 19.3: a whole graph, vectorized dispatch, reduction, and a real fallback ===

--- Part 1: a genuine broadcast the vector path can't safely handle ---

a:[3,4] (12 elements), b:[4] (4 elements) -- canVectorizeElementwise(out) = false
self-check: a real trailing-dimension broadcast is correctly rejected by the vector
path's own scope check (confirmed)

dispatcher chose: SCALAR-FALLBACK

scalar fallback, compiled and run for real: clean compile. output matches evaluateArrays() (confirmed)

--- Part 2: Chapter 16's 10-node graph, vectorized end to end ---

a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000 

fused graph (maxChainLength=3, same cap Section 17.3/18.3 used): a b t3(FE,3st) t5(FE,2st) s(FR,1st) y(FE,2st) 

per-node dispatch (AVX2/FMA):
  t3   -> VECTOR+FMA
  t5   -> VECTOR+FMA
  s    -> VECTOR-REDUCTION
  y    -> VECTOR+FMA

self-check: every fused node in this graph -- 3 elementwise groups plus the
reduction -- is genuinely vectorizable (no broadcast beyond a scalar anywhere
in this graph), so the dispatcher chose VECTOR paths for all 4 nodes, 0
scalar fallbacks (confirmed)

self-check: 3 total FMA fold(s) applied across the 3 elementwise groups --
t3's own ReLU-then-Mul-then-Add folds its trailing Mul+Add, t5's own Mul-then-Add
folds, and y's own Mul-then-Add (s + t5*b) folds too, for exactly 3 (confirmed)

--- g++ compile (real AVX2/FMA flags, 4 generated functions) ---

(no output -- clean compile)

--- Running the whole vectorized graph FOR REAL on this machine's own AVX2/FMA hardware ---

a: 1.000000 -2.000000 3.000000 -4.000000 5.000000 -6.000000 7.000000 -8.000000
b: 2.000000
t3: 4.000000 2.000000 8.000000 2.000000 12.000000 2.000000 16.000000 2.000000
t5: 10.000000 6.000000 18.000000 6.000000 26.000000 6.000000 34.000000 6.000000
s: 112.000000
y: 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000

evaluateArrays(original).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000 
generated (vectorized, 4 functions dispatched, compiled and run for real).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000 

self-check: y[0] = 132.000000 -- the SAME number Section 17.3's own CPU loop nest and
Section 18.3's own CPU reference both reported for this exact graph, now produced
by real vector instructions on real AVX2/FMA hardware (confirmed)

self-check: every generated buffer (not just y) matches evaluateArrays() on the
SAME fused graph -- the vector path and the interpreter agree at every node, not
just at the output (confirmed)
```

**Output (device, aarch64 Apple Silicon, NEON):**

```text
=== Section 19.3: a whole graph, vectorized dispatch, reduction, and a real fallback ===

--- Part 1: a genuine broadcast the vector path can't safely handle ---

a:[3,4] (12 elements), b:[4] (4 elements) -- canVectorizeElementwise(out) = false
self-check: a real trailing-dimension broadcast is correctly rejected by the vector
path's own scope check (confirmed)

dispatcher chose: SCALAR-FALLBACK

scalar fallback, compiled and run for real: clean compile. output matches evaluateArrays() (confirmed)

--- Part 2: Chapter 16's 10-node graph, vectorized end to end ---

a = [1,-2,3,-4,5,-6,7,-8], b = [2]. evaluateArrays(original).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000

fused graph (maxChainLength=3, same cap Section 17.3/18.3 used): a b t3(FE,3st) t5(FE,2st) s(FR,1st) y(FE,2st)

per-node dispatch (NEON):
  t3   -> VECTOR+FMA
  t5   -> VECTOR+FMA
  s    -> VECTOR-REDUCTION
  y    -> VECTOR+FMA

self-check: every fused node in this graph -- 3 elementwise groups plus the
reduction -- is genuinely vectorizable (no broadcast beyond a scalar anywhere
in this graph), so the dispatcher chose VECTOR paths for all 4 nodes, 0
scalar fallbacks (confirmed)

self-check: 3 total FMA fold(s) applied across the 3 elementwise groups --
t3's own ReLU-then-Mul-then-Add folds its trailing Mul+Add, t5's own Mul-then-Add
folds, and y's own Mul-then-Add (s + t5*b) folds too, for exactly 3 (confirmed)

--- g++ compile (real NEON flags, 4 generated functions) ---

(no output -- clean compile)

--- Running the whole vectorized graph FOR REAL on this machine's own NEON hardware ---

a: 1.000000 -2.000000 3.000000 -4.000000 5.000000 -6.000000 7.000000 -8.000000
b: 2.000000
t3: 4.000000 2.000000 8.000000 2.000000 12.000000 2.000000 16.000000 2.000000
t5: 10.000000 6.000000 18.000000 6.000000 26.000000 6.000000 34.000000 6.000000
s: 112.000000
y: 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000

evaluateArrays(original).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000
generated (vectorized, 4 functions dispatched, compiled and run for real).y = 132.000000 124.000000 148.000000 124.000000 164.000000 124.000000 180.000000 124.000000

self-check: y[0] = 132.000000 -- the SAME number Section 17.3's own CPU loop nest and
Section 18.3's own CPU reference both reported for this exact graph, now produced
by real vector instructions on real NEON hardware (confirmed)

self-check: every generated buffer (not just y) matches evaluateArrays() on the
SAME fused graph -- the vector path and the interpreter agree at every node, not
just at the output (confirmed)
```


!!! note "What this chapter proved, and what stayed a stated limitation"
    Every generated function in this chapter -- elementwise, FMA-folded, reduction, and scalar fallback alike -- was compiled AND executed for real, on both of this book's own real architectures, closing a loop Chapter 18 could only leave honestly open (a compiled CUDA program that could not actually run). What stayed a stated, deliberate limitation: `canVectorizeElementwise()` only ever handles a node whose own external inputs are either full-width or scalar, correctly refusing a genuine interior/trailing broadcast rather than silently miscompiling it -- proven concretely in Section 19.3, not just asserted; and the horizontal sum in Section 19.3's own vectorized reduction deliberately uses the slower, portable array-extraction technique on both architectures rather than NEON's own faster single-instruction `vaddvq_f32`, the same "correct, not fastest" honesty Chapter 18's own `atomicAdd()` discussion already practiced for a different reason.

## Chapter Summary

This chapter generated real, vector-instruction CPU code for the first time, on two genuinely different real architectures this book has access to -- and, unlike every earlier CUDA-targeting chapter, ACTUALLY RAN every single one of them. Section 19.1 built the `Isa` abstraction and its five vector primitives (`vecLoadOrBroadcast()`, `vecAdd()`, `vecMul()`, `vecMax()`, `vecStore()`), hiding AVX2's and NEON's own different intrinsic names behind one generator that emits a vector main loop plus a scalar tail (reusing Chapter 18's own `emitSteps()` unchanged), gated by `canVectorizeElementwise()` -- a real, stated scope check that only ever accepts a full-width or scalar external input, correctly rejecting a genuine interior broadcast. Section 19.2 added a real peephole optimization: `vecFma()` hides a genuine hardware difference (AVX2's accumulator-last argument order versus NEON's accumulator-first) behind one call, and a two-pass fold-detection generator recognizes a `Mul` step immediately followed by a consuming `Add` and emits one fused instruction instead of two -- proven safe not by testing against a constructed counterexample, but by tracing the safety condition back to Chapter 13's own five-chapters-old single-consumer inlining invariant. Section 19.3 closed the chapter with a full per-node dispatcher: a genuinely new vectorized-reduction technique (a running vector accumulator, collapsed to one scalar by a portable horizontal sum), a real scalar fallback proven correct by compiling and running an actual broadcast case through it, and Chapter 16's own 10-node capstone graph, vectorized end to end (three FMA-fold elementwise groups, one vectorized reduction, zero scalar fallbacks, three total folds) and genuinely executed on both the cloud sandbox's AVX2/FMA and the device's NEON, reporting the same `y[0]=132.0` Chapter 16, 17, and 18 have all already established -- this time computed by real vector instructions on real hardware, not a serial loop or an honestly-uncheckable GPU kernel.

## Self-Check Questions

1. `vecLoadOrBroadcast()` makes exactly one real decision. What is it, and what does `canVectorizeElementwise()` check to decide, ahead of time, which branch applies for a given external input?
2. Why does this chapter's own generated code index external inputs as `ext[0]`, `ext[1]`, ... rather than using Chapter 18's own named-parameter convention (`ext0`, `ext1`, ...)?
3. `_mm256_fmadd_ps(a, b, c)` and `vfmaq_f32(acc, a, b)` compute the same mathematical result. What, concretely, differs between their own argument orders, and what function hides that difference?
4. Section 19.2's own FMA fold needs one safety condition to hold: a folded `Mul` step's own value must never be needed anywhere else. What invariant, from which earlier chapter, guarantees that condition always holds for a real fused group -- and how does this chapter make that guarantee CHECKABLE rather than just assumed?
5. What is the one new idea a vectorized REDUCTION needs beyond a vectorized elementwise loop, and what does this chapter's own horizontal sum deliberately choose NOT to use, even though a faster alternative exists on one of the two architectures?
6. Section 19.3 constructs a graph with a real trailing-dimension broadcast (`a:[3,4]`, `b:[4]`) before running the main capstone graph. What does this prove, and why is it not enough to simply show that `canVectorizeElementwise()` returns false for it?
7. Chapter 16's own 10-node graph, fused with `maxChainLength=3`, produces four materialized nodes. How many of them take a vector path versus a scalar fallback in this chapter's own dispatcher, and why?
8. How many total FMA folds fire across Chapter 16's own capstone graph in Section 19.3, and in which three nodes?

## Where We Go Next

Chapter 20, "A JIT Backend: Compiling and Loading Generated Code at Runtime," replaces every backend's own shell-out-and-read-stdout harness -- used, unchanged in spirit, since Chapter 17 first wrote generated C++ to a temp file and invoked a real compiler as a subprocess -- with real in-process code loading via `dlopen`: generated code compiled to a shared library and called directly, as a function pointer, without ever spawning a second process or parsing printed text back out of its stdout. Apply the Chapter 5-19 depth-level standard (more prose, more diagrams before code) throughout.

## Worked Solutions

1. The one real decision is whether a given external input gets loaded as a CONTIGUOUS VECTOR (`_mm256_loadu_ps`/`vld1q_f32`, reading `width` consecutive floats starting at the current loop index) or BROADCAST from a single scalar (`_mm256_set1_ps`/`vdupq_n_f32`, replicating one value into every lane). `canVectorizeElementwise()` decides which applies, ahead of time, by comparing that input's own element count against the node's own output count (equal -> contiguous vector) and against 1 (equal -> scalar broadcast); any other count makes the whole node ineligible for this chapter's own vector path.
2. Chapter 18's own named-parameter convention (`ext0`, `ext1`, ...) existed for a real, physical reason specific to CUDA: a `__global__` kernel cannot dereference a host-memory `std::vector`'s own buffer, so each device pointer had to arrive as its own distinct kernel parameter. Plain CPU code has no such host/device address-space split, so this chapter reuses the simpler CPU convention Chapter 17 and 18 both already established for scalar code -- a single `const std::vector<const float*>& ext` parameter, indexed at codegen time.
3. `_mm256_fmadd_ps(a, b, c)` computes `a*b + c` -- the two multiplicands first, the accumulator last. `vfmaq_f32(acc, a, b)` computes `acc + a*b` -- the accumulator FIRST, the two multiplicands after it. Same three operands, same mathematical result, different argument POSITIONS. `vecFma(isa, mulLhs, mulRhs, addend)` hides this behind one call meaning "multiply mulLhs by mulRhs, add addend," reordering the arguments correctly for whichever `Isa` is being generated.
4. The guarantee comes from Chapter 13's own `resolveIntoGroup()`, specifically its `mustBeExternal` check, which includes `consumers.at(oldId) != 1`: any node read by more than one other node in the original graph is always externalized -- forced OUT of a fused group -- before fusion could ever let it be shared by two steps within one group's own step list. The consequence is that, within any one fused group, every non-final step has exactly one reader, which is exactly the safety condition the FMA fold needs. This chapter makes it checkable, not just assumed, via `computeStepUseCounts()`, which actually counts each step's own references and confirms the count is 1 before folding, rather than relying on the invariant silently.
5. The one new idea is accumulating each iteration's own per-element result into a running VECTOR accumulator across the main loop, instead of storing per-lane results separately, and collapsing that accumulator to one scalar ("horizontal sum") only ONCE, after the loop ends. This chapter's own horizontal sum deliberately does NOT use NEON's own `vaddvq_f32`, a single instruction that sums all four lanes at once -- it uses the slower but portable array-extraction-and-add technique on both architectures, since AVX2 has no equivalent single instruction without AVX-512.
6. It proves the scalar fallback path is genuinely CORRECT, not merely "not vectorized" -- the generated scalar function for that broadcast case is actually compiled and run, and its output is checked against `evaluateArrays()`, not just assumed to work because the vector path was skipped. Showing only that `canVectorizeElementwise()` returns false would prove the scope check fires correctly, but nothing about whether the code the dispatcher falls back to actually computes the right answer; compiling and running it is what closes that gap.
7. All four materialized nodes (`t3`, `t5`, the reduction `s`, and `y`) take a VECTOR path -- three via the FMA-fold elementwise generator and one via the new vectorized-reduction generator -- with zero scalar fallbacks. This is because every external input in this particular graph is either full-width (matching its own node's output count) or a true scalar (`b`, count 1); nothing in Chapter 16's own capstone graph contains the kind of interior/trailing broadcast that would make `canVectorizeElementwise()` return false.
8. Three total FMA folds fire, one each inside `t3`, `t5`, and `y`. `t3`'s own three raw steps are ReLU, then a Mul immediately followed by a consuming Add, which folds. `t5`'s own two raw steps are a Mul immediately followed by a consuming Add, which folds. `y`'s own two raw steps are a Mul (`t5 * b`) immediately followed by an Add that consumes it (`s + that product`), which also folds -- the reduction node `s` itself does not participate in this count, since it takes the separate vectorized-reduction path, not the FMA-fold elementwise path.

---

**Sources cited in this chapter:**

None new. This chapter's own `Isa` abstraction, vector primitives, FMA-fold peephole optimization, and vectorized-reduction technique are all original to this book, verified against this book's own directly observed AVX2/FMA and NEON toolchain behavior (documented in this chapter's own opening section) rather than any external source.