# 1. The Cost of Eager Execution

<!-- hammer:figure:opener-1:start -->
<figure class="chapter-figure" markdown="1">

[![The Cost of Eager Execution](../images/openers/01.svg?v=b86f687b676a)](../images/openers/01.svg?v=b86f687b676a)

<figcaption>Chapter roadmap. Open any diagram for a full-size view.</figcaption>
</figure>
<!-- hammer:figure:opener-1:end -->

!!! note "Timing provenance"
    The NVIDIA blog's 9.6 microseconds is end-to-end latency of a particular synchronized V100 kernel, including its 2.9-microsecond GPU execution. It is not pure launch overhead or a universal device constant. The host example scales this historical latency as an illustrative model; fused latency depends on the fused work and is not measured here. [NVIDIA source](https://developer.nvidia.com/blog/cuda-graphs/).


**What you will understand:** eager execution can incur repeated launch overhead and logical intermediate-array traffic. The examples count logical accesses and separately scale a cited historical synchronized-kernel latency; neither is a fresh GPU performance measurement.

**What you need to know first:** basic C++ (loops, `std::vector`, functions) and the idea that a tensor operation -- an elementwise add, a matrix multiply, an activation function -- reads some input data and produces some output data. No CUDA or compiler background is assumed; that starts in Chapter 3.

---

Every tensor framework a reader has used runs a chain of operations by executing them one at a time: allocate an output, compute it fully, move to the next operation, using that output as the next input. This is called eager execution, and it is not a design mistake -- it is what makes a framework interactive and easy to debug, since every intermediate value is a real, inspectable object the moment it's produced. But "the moment it's produced" is exactly the phrase that hides this chapter's whole argument: producing an intermediate value eagerly means writing it to memory in full, immediately, whether or not anything outside the current chain will ever look at it. This chapter makes that cost concrete and countable, in two independent ways, before Chapter 2 gets to what a compiler pipeline actually does about it.

```text
EAGER (op-at-a-time, chain length 3):

  input --> +------+ --> tmp1 --> +------+ --> tmp2 --> +------+ --> output
            | op 1 |              | op 2 |              | op 3 |
            +------+              +------+              +------+

  every --> touching tmp1 or tmp2 above is a full array read or a full
  array write, in real memory: 2 memory round trips per operation, and a
  separate kernel launch per operation, for a chain of any length N.

FUSED (compiled, chain length 3):

  input --> +--------------------------+ --> output
            | op 1 -> op 2 -> op 3     |
            +--------------------------+

  the --> between op 1, op 2, op 3 inside the single box is a value held
  in a register: it never becomes a separate array, never touches memory,
  and the whole chain is one kernel launch, regardless of N.
```

## 1.1 What Eager Execution Actually Does, Step by Step

<!-- hammer:figure:1.1:start -->
<figure class="chapter-figure" markdown="1">

[![Eager execution materializes intermediates](../images/concepts/01-01.svg?v=eefa4fb58b25)](../images/concepts/01-01.svg?v=eefa4fb58b25)

<figcaption>Each materialized stage adds logical array reads and writes.</figcaption>
</figure>
<!-- hammer:figure:1.1:end -->

### Intuition

Picture a courier who has three packages to deliver to the same address, one after another. Instead of loading all three into the van at once, the courier drives to the warehouse, loads package one, drives to the address, drops it off, drives back to the warehouse, loads package two, drives to the address again, and so on. Each trip is a real, full round trip regardless of how small the package is. Eager execution is that courier: each operation in a chain is its own full round trip to memory, even when the "package" -- the intermediate result -- was never going anywhere except into the very next operation.

```text
op 1: read input (full array in memory) --> compute --> write tmp1 (full array)
op 2: read tmp1  (full array in memory) --> compute --> write tmp2 (full array)
op 3: read tmp2  (full array in memory) --> compute --> write output (full array)

3 operations --> 3 separate full-array reads + 3 separate full-array writes
= 6 real memory touches, for a chain that only needed to touch memory
twice: once to read the original input, once to write the final output.
```

### Background

The code below computes a real N-operation elementwise chain (alternating bias-add, scale, and ReLU, so a chain of any length exercises all three) two genuinely different ways over the exact same input array: EAGER, which allocates a brand-new full array for every operation's output, and FUSED, which computes the entire chain per element in a single pass with no intermediate array at all. Both modes are genuinely compiled and run; their final outputs are compared element-for-element (bit-identical is expected here, since both modes apply the exact same sequence of elementary operations in the exact same order -- there is no reduction or summation involved, so there is no floating-point associativity question the way there would be for a sum). The number of bytes each mode actually reads and writes is counted directly from real array sizes as the program runs, then checked against a closed-form formula.

```cpp
// Chapter 1: The Cost of Eager Execution
// 001_eager_vs_fused_memory_traffic.cpp
//
// Section 1.1 -- what "eager execution" actually does, step by step, and
// Section 1.3 -- the real memory-traffic cost that follows from it.
//
// A real N-operation elementwise chain (add a constant, multiply by a
// constant, clamp at zero -- i.e. a bias-add, a scale, and a ReLU,
// repeated) is computed two genuinely different ways over the exact same
// input data:
//   - EAGER: each operation reads a full input array and writes a brand
//     new full output array, exactly the way an op-at-a-time framework
//     materializes every intermediate result.
//   - FUSED: a single pass over the data computes every operation in the
//     chain per element, in registers, and only ever reads the original
//     input once and writes the final output once.
// Both modes are genuinely executed and their outputs are compared
// element-for-element (bit-identical is expected and checked -- this is
// the same sequence of elementary float ops applied in the same order
// either way, so there is no reduction-order question here the way
// there is for a sum or a collective). The real byte counts each mode
// actually reads and writes are counted directly from array sizes, not
// timed and not estimated -- and checked against a closed-form formula
// for the number of chained operations.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 001_eager_vs_fused_memory_traffic.cpp -o 001_eager_vs_fused_memory_traffic
// Run:     ./001_eager_vs_fused_memory_traffic
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

// One elementwise operation in the chain: alternates bias-add, scale,
// and ReLU, so a chain of any length N exercises all three.
static float applyOp(int opIndex, float x) {
    switch (opIndex % 3) {
        case 0: return x + 0.5f;               // bias-add
        case 1: return x * 1.25f;               // scale
        default: return std::max(0.0f, x);      // ReLU
    }
}

struct ByteCounts {
    uint64_t bytesRead = 0;
    uint64_t bytesWritten = 0;
};

// EAGER: op i reads the FULL array produced by op i-1 and writes a FULL
// new array -- a real heap allocation per op, exactly mirroring how a
// framework in eager mode materializes every intermediate tensor.
static ByteCounts runEager(const std::vector<float>& input, int numOps,
                            std::vector<float>& finalOut) {
    ByteCounts counts;
    std::vector<float> current = input;
    for (int op = 0; op < numOps; ++op) {
        std::vector<float> next(current.size());
        for (size_t i = 0; i < current.size(); ++i) {
            next[i] = applyOp(op, current[i]);
        }
        counts.bytesRead += current.size() * sizeof(float);
        counts.bytesWritten += next.size() * sizeof(float);
        current = std::move(next);
    }
    finalOut = current;
    return counts;
}

// FUSED: one pass. The original input is read exactly once; the final
// output is written exactly once. Every intermediate value between ops
// lives in a single local float (a register in any real optimized
// build), never in a separate heap array.
static ByteCounts runFused(const std::vector<float>& input, int numOps,
                            std::vector<float>& finalOut) {
    ByteCounts counts;
    finalOut.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float value = input[i];
        for (int op = 0; op < numOps; ++op) {
            value = applyOp(op, value);
        }
        finalOut[i] = value;
    }
    counts.bytesRead = input.size() * sizeof(float);
    counts.bytesWritten = finalOut.size() * sizeof(float);
    return counts;
}

int main() {
    printf("=== Section 1.1 / 1.3: eager vs. fused execution of the same op chain ===\n\n");

    const size_t arraySize = 1000; // small on purpose -- this chapter's claim is about
                                    // byte-COUNT scaling with chain length, not raw size.
    std::vector<float> input(arraySize);
    for (size_t i = 0; i < arraySize; ++i) {
        input[i] = static_cast<float>(i) * 0.01f - 3.0f; // spans negative and positive
    }
    const uint64_t arrayBytes = arraySize * sizeof(float);

    printf("Array size: %zu float32 elements (%llu bytes)\n\n", arraySize,
           (unsigned long long)arrayBytes);

    printf("%-8s %-22s %-22s %-14s %-10s\n", "num_ops", "eager_bytes_moved",
           "fused_bytes_moved", "reduction_x", "outputs_match");

    bool allFormulaMatches = true;
    bool allOutputsMatch = true;

    for (int numOps = 1; numOps <= 8; ++numOps) {
        std::vector<float> eagerOut, fusedOut;
        ByteCounts eager = runEager(input, numOps, eagerOut);
        ByteCounts fused = runFused(input, numOps, fusedOut);

        uint64_t eagerTotal = eager.bytesRead + eager.bytesWritten;
        uint64_t fusedTotal = fused.bytesRead + fused.bytesWritten;

        // Closed-form check: eager reads+writes a full array on every one
        // of the numOps operations (2 * arrayBytes per op); fused reads
        // the input once and writes the output once, regardless of chain
        // length (2 * arrayBytes total, always).
        uint64_t predictedEager = static_cast<uint64_t>(numOps) * 2ULL * arrayBytes;
        uint64_t predictedFused = 2ULL * arrayBytes;
        bool formulaMatches = (predictedEager == eagerTotal) && (predictedFused == fusedTotal);
        allFormulaMatches = allFormulaMatches && formulaMatches;

        bool outputsMatch = (eagerOut.size() == fusedOut.size());
        for (size_t i = 0; outputsMatch && i < eagerOut.size(); ++i) {
            if (eagerOut[i] != fusedOut[i]) outputsMatch = false;
        }
        allOutputsMatch = allOutputsMatch && outputsMatch;

        double reduction = static_cast<double>(eagerTotal) / static_cast<double>(fusedTotal);
        printf("%-8d %-22llu %-22llu %-14.2f %-10s\n", numOps,
               (unsigned long long)eagerTotal, (unsigned long long)fusedTotal,
               reduction, outputsMatch ? "yes" : "NO");
    }

    printf("\nself-check: every row's counted bytes match the closed-form formula\n");
    printf("eager(N) = N * 2 * array_bytes, fused = 2 * array_bytes (%s); eager and\n",
           allFormulaMatches ? "confirmed" : "MISMATCH");
    printf("fused outputs are bit-identical at every chain length (%s)\n",
           allOutputsMatch ? "confirmed" : "MISMATCH");

    return (allFormulaMatches && allOutputsMatch) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 001_eager_vs_fused_memory_traffic.cpp -o 001_eager_vs_fused_memory_traffic
./001_eager_vs_fused_memory_traffic
```

**Output:**

```text
=== Section 1.1 / 1.3: eager vs. fused execution of the same op chain ===

Array size: 1000 float32 elements (4000 bytes)

num_ops  eager_bytes_moved      fused_bytes_moved      reduction_x    outputs_match
1        8000                   8000                   1.00           yes
2        16000                  8000                   2.00           yes
3        24000                  8000                   3.00           yes
4        32000                  8000                   4.00           yes
5        40000                  8000                   5.00           yes
6        48000                  8000                   6.00           yes
7        56000                  8000                   7.00           yes
8        64000                  8000                   8.00           yes

self-check: every row's counted bytes match the closed-form formula
eager(N) = N * 2 * array_bytes, fused = 2 * array_bytes (confirmed); eager and
fused outputs are bit-identical at every chain length (confirmed)
```

!!! warning "[COMMON TRAP] Assuming an intermediate result is 'free' because it's 'just in memory'"
    It's easy to think of `tmp1` and `tmp2` above as bookkeeping rather than real cost, since they never appear in the framework's own output. They are not free: each one is a real heap allocation, a real full-array write, and a real full-array read by the next operation. A chain that looks like one logical computation in source code is, under eager execution, exactly as many real memory round trips as it has operations.

## 1.2 Why Fixed Per-Kernel-Launch Overhead Multiplies With More Operations

<!-- hammer:figure:1.2:start -->
<figure class="chapter-figure" markdown="1">

[![Launch overhead is a per-launch cost](../images/concepts/01-02.svg?v=24a9f3b02b21)](../images/concepts/01-02.svg?v=24a9f3b02b21)

<figcaption>A fixed launch-cost model is illustrative; real timings require a GPU.</figcaption>
</figure>
<!-- hammer:figure:1.2:end -->

### Intuition

A tollbooth charges the same fixed toll no matter how much cargo is in the vehicle passing through. A truck carrying one crate and a truck carrying a hundred crates both stop, both pay, both wait for the gate. Launching a kernel -- asking the hardware to actually start executing a block of code -- has exactly this shape: a real, mostly-fixed cost paid at the moment of launch, independent of how much or how little work that kernel actually does. A chain of three small operations pays that toll three times; one fused operation covering the same work pays it once.

```text
EAGER: 3 kernel launches, each pays the fixed per-launch toll
  [launch] --> [launch] --> [launch]
   9.6us         9.6us         9.6us     =  28.8us total, paid 3 times

FUSED: 1 kernel launch, the toll is paid once
  [launch]
   9.6us                                 =   9.6us total, paid once
```

(9.6us is NVIDIA's measured per-kernel *wall-clock* time, not pure overhead --
of that, NVIDIA's own post reports roughly 2.9us is real kernel execution, so
the fixed launch-and-sync cost this chapter is arguing about is closer to the
remaining ~6.7us. The chapter uses the full 9.6us figure throughout because
that is the real, cited, end-to-end cost actually paid per launch -- overhead
included, not subtracted out.)

### Background

The timing input is NVIDIA's historical V100 result: 9.6 microseconds per individually synchronized kernel, including 2.9 microseconds of GPU execution. This is end-to-end latency, not launch overhead alone. The table scales that latency for repeated equal-cost kernels and contrasts a hypothetical one-kernel baseline; a fused kernel performs different work, so its latency must be measured independently. Eager CUDA operations can be queued asynchronously on a stream; per-operation host synchronization is a worst-case reference pattern, not a requirement of eager execution.

```cpp
// Chapter 1: The Cost of Eager Execution
// 002_launch_overhead_and_synthesis.cpp
//
// Section 1.2 -- cited synchronized per-kernel latency multiplied by chain length --
// and a closing synthesis combining it with File 1's real byte-traffic
// model.
//
// This file never fabricates a timing number. The one number here that
// is not directly counted by this program is NVIDIA's own real, cited,
// published end-to-end synchronized per-kernel latency (9.6 microseconds
// per kernel, "including overheads", measured with a CPU wallclock timer
// around per-kernel-synchronized launches) from NVIDIA's own "Getting
// Started with CUDA Graphs" developer blog post -- used here exactly the
// way Appendix D of this book's own sibling volume already used a real
// cited hardware bandwidth figure: as an input to a real, stated formula,
// not as a re-measurement performed by this program. Every other number
// below is genuinely computed arithmetic on that cited constant, or a
// genuinely counted byte total reproduced from File 1's own closed-form
// result.
//
// Source: NVIDIA Technical Blog, "Getting Started with CUDA Graphs",
// https://developer.nvidia.com/blog/cuda-graphs/ -- "divide by
// NSTEP*NKERNEL which gives 9.6us per kernel (including overheads)".
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 002_launch_overhead_and_synthesis.cpp -o 002_launch_overhead_and_synthesis
// Run:     ./002_launch_overhead_and_synthesis
#include <cstdio>
#include <cstdint>

int main() {
    printf("=== Section 1.2: cited synchronized per-kernel latency, multiplied by chain length ===\n\n");

    // Real, cited constant -- NOT measured by this program. See the
    // header comment above for the exact source and quote.
    const double NVIDIA_CITED_SYNCHRONIZED_LATENCY_US = 9.6;

    printf("NVIDIA's own cited figure (CUDA Graphs blog post): %.1f us per synchronized kernel,\n",
           NVIDIA_CITED_SYNCHRONIZED_LATENCY_US);
    printf("measured end to end with per-kernel host synchronization -- a worst-case\n");
    printf("reference pattern; eager submission need not host-synchronize per operation.\n");
    printf("The cited 9.6 us INCLUDES 2.9 us of GPU kernel execution; it is not pure launch overhead.\n");
    printf("This illustrative model assumes unchanged per-kernel latency, not a fusion speed prediction.\n");
    printf("A fused kernel replaces the whole chain with exactly ONE launch, regardless\n");
    printf("of how many operations were fused into it.\n\n");

    printf("%-8s %-24s %-24s %-12s\n", "num_ops", "eager_model_us",
           "one_kernel_model_us", "reduction_x");

    for (int numOps = 1; numOps <= 8; ++numOps) {
        double eagerOverheadUs = static_cast<double>(numOps) * NVIDIA_CITED_SYNCHRONIZED_LATENCY_US;
        double fusedOverheadUs = 1.0 * NVIDIA_CITED_SYNCHRONIZED_LATENCY_US;
        double reduction = eagerOverheadUs / fusedOverheadUs;
        printf("%-8d %-24.2f %-24.2f %-12.2f\n", numOps, eagerOverheadUs, fusedOverheadUs, reduction);
    }

    printf("\n=== Synthesis: a concrete 4-operation chain on a 1,000,000-element tensor ===\n\n");

    const int numOps = 4;
    const uint64_t numElements = 1000000;
    const uint64_t arrayBytes = numElements * sizeof(float);

    // Byte-traffic side: reproduces File 1's own closed-form result
    // (eager(N) = N * 2 * array_bytes, fused = 2 * array_bytes) at this
    // chapter's own concrete example size -- not re-simulated here, the
    // formula was already checked against genuinely counted bytes in
    // File 1 for N = 1..8.
    uint64_t eagerBytes = static_cast<uint64_t>(numOps) * 2ULL * arrayBytes;
    uint64_t fusedBytes = 2ULL * arrayBytes;

    // Launch-overhead side: the real cited constant applied to this
    // example's own chain length.
    double eagerOverheadUs = static_cast<double>(numOps) * NVIDIA_CITED_SYNCHRONIZED_LATENCY_US;
    double fusedOverheadUs = NVIDIA_CITED_SYNCHRONIZED_LATENCY_US;

    printf("Chain length: %d operations. Tensor: %llu float32 elements (%.2f MiB).\n\n",
           numOps, (unsigned long long)numElements, arrayBytes / (1024.0 * 1024.0));
    printf("%-30s %-15s %-15s\n", "", "eager", "fused");
    printf("%-30s %-15.2f %-15.2f  (us, from NVIDIA's cited synchronized-kernel figure)\n",
           "synchronized latency model", eagerOverheadUs, fusedOverheadUs);
    printf("%-30s %-15.2f %-15.2f  (MiB, logically counted from array sizes)\n",
           "memory traffic", eagerBytes / (1024.0 * 1024.0), fusedBytes / (1024.0 * 1024.0));

    printf("\nThese two reductions are reported SEPARATELY and are not combined into a\n");
    printf("single predicted wall-clock speedup -- doing that would require an assumed\n");
    printf("memory bandwidth and an assumed relationship between overhead and traffic\n");
    printf("time, which this book's own standing discipline does not fabricate. What is\n");
    printf("real here is that a compiler that fuses this chain removes %dx of the launch\n",
           numOps);
    printf("modeled synchronized-latency cost AND %dx of logical memory traffic, from two independent mechanisms --\n",
           numOps);

    bool launchReductionCorrect = (eagerOverheadUs / fusedOverheadUs) == static_cast<double>(numOps);
    bool trafficReductionCorrect = (eagerBytes / fusedBytes) == static_cast<uint64_t>(numOps);
    printf("self-check: synchronized-latency model ratio equals the chain length exactly (%s),\n",
           launchReductionCorrect ? "confirmed" : "MISMATCH");
    printf("memory-traffic reduction equals the chain length exactly (%s)\n",
           trafficReductionCorrect ? "confirmed" : "MISMATCH");

    return (launchReductionCorrect && trafficReductionCorrect) ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 002_launch_overhead_and_synthesis.cpp -o 002_launch_overhead_and_synthesis
./002_launch_overhead_and_synthesis
```

**Output:**

```text
=== Section 1.2: cited synchronized per-kernel latency, multiplied by chain length ===

NVIDIA's own cited figure (CUDA Graphs blog post): 9.6 us per synchronized kernel,
measured end to end with per-kernel host synchronization -- a worst-case
reference pattern; eager submission need not host-synchronize per operation.
The cited 9.6 us INCLUDES 2.9 us of GPU kernel execution; it is not pure launch overhead.
This illustrative model assumes unchanged per-kernel latency, not a fusion speed prediction.
A fused kernel replaces the whole chain with exactly ONE launch, regardless
of how many operations were fused into it.

num_ops  eager_model_us           one_kernel_model_us      reduction_x
1        9.60                     9.60                     1.00
2        19.20                    9.60                     2.00
3        28.80                    9.60                     3.00
4        38.40                    9.60                     4.00
5        48.00                    9.60                     5.00
6        57.60                    9.60                     6.00
7        67.20                    9.60                     7.00
8        76.80                    9.60                     8.00

=== Synthesis: a concrete 4-operation chain on a 1,000,000-element tensor ===

Chain length: 4 operations. Tensor: 1000000 float32 elements (3.81 MiB).

                               eager           fused
synchronized latency model     38.40           9.60             (us, from NVIDIA's cited synchronized-kernel figure)
memory traffic                 30.52           7.63             (MiB, logically counted from array sizes)

These two reductions are reported SEPARATELY and are not combined into a
single predicted wall-clock speedup -- doing that would require an assumed
memory bandwidth and an assumed relationship between overhead and traffic
time, which this book's own standing discipline does not fabricate. What is
real here is that a compiler that fuses this chain removes 4x of the launch
modeled synchronized-latency cost AND 4x of logical memory traffic, from two independent mechanisms --
self-check: synchronized-latency model ratio equals the chain length exactly (confirmed),
memory-traffic reduction equals the chain length exactly (confirmed)
```

!!! warning "[COMMON TRAP] Treating launch overhead as negligible because it's a few microseconds"
    Launch overhead can matter for sequences of short kernels. The cited 9.6-microsecond figure includes execution and synchronization, so multiplying it counts that entire historical latency, not isolated launch overhead. Async stream submission, CUDA Graphs, fusion, kernel work, and device characteristics change the result; the table is an illustration, not a predicted speedup.

## 1.3 Memory Traffic Is the Dominant Cost for Memory-Bound Chains

<!-- hammer:figure:1.3:start -->
<figure class="chapter-figure" markdown="1">

[![Fusion removes intermediate traffic](../images/concepts/01-03.svg?v=481e8513ef2d)](../images/concepts/01-03.svg?v=481e8513ef2d)

<figcaption>Logical traffic savings are not a measured speedup.</figcaption>
</figure>
<!-- hammer:figure:1.3:end -->

### Intuition

Think of a relay race where, instead of runners handing the baton directly to each other, every handoff is routed through a separate judge's table off to the side: runner one carries the baton to the table, sets it down, runner two picks it up from the table, carries it to the next table, and so on. The baton travels the same total distance either way, but every off-to-the-side trip is pure overhead that direct hand-to-hand passing would never pay. File 1's own closed-form result already proved this chapter's version of that overhead exactly: eager execution's memory traffic grows linearly with chain length, while fused execution's memory traffic never grows past the one real read and one real write the computation actually needed.

```text
memory traffic moved (in units of one full array, read+write), by chain length N:

N=1   | --                    ( 2 units)
N=2   | ----                  ( 4 units)
N=4   | --------              ( 8 units)
N=8   | ----------------      (16 units)
fused | --                    ( 2 units, constant, regardless of N)
```

### Background

Section 1.1's own File 1 already contains this section's real code and real, locked output -- the `eager_bytes_moved` and `fused_bytes_moved` columns of that same table are this section's own claim, verified the same way: genuinely counted bytes from real array sizes, checked against the closed-form formulas `eager(N) = N * 2 * array_bytes` and `fused = 2 * array_bytes` for every chain length from 1 to 8. Nothing new needs to be compiled here; what's new is reading that table as a statement about memory traffic specifically, independent of Section 1.2's separate launch-overhead argument. The two costs are genuinely independent mechanisms: a hypothetical kernel-launch mechanism with zero overhead would still leave the memory-traffic column exactly as it is, and a hypothetical infinite-bandwidth memory system would still leave the launch-overhead column exactly as it is. A real compiler has to attack both, which is exactly what Part 3's fusion passes do.

!!! warning "[COMMON TRAP] Assuming heavier per-element compute automatically hides memory cost"
    A chain of operations that do more arithmetic per element -- not just add and multiply, but something like a real activation function's exponential -- does not automatically escape this chapter's argument. It only escapes it once the amount of *compute* per byte moved is high enough that the hardware is busy computing rather than waiting on memory; this ratio has a name (arithmetic intensity) and a real quantitative test (the roofline model), and Chapter 12 builds both from scratch. Until then, assume a chain of small elementwise operations is memory-bound, because for the specific ops this chapter uses -- add, multiply, ReLU -- it genuinely is.

## Chapter Summary

Eager execution can materialize intermediate arrays and issue multiple launches. The logical byte model shows how eliminating those materializations can reduce array traffic; real HBM traffic depends on caching and implementation. The separate timing illustration scales NVIDIA's historical synchronized-kernel latency, including execution. It neither measures pure launch overhead nor predicts the latency of a fused kernel. Host synchronization after each operation is not required by eager execution: operations on the same CUDA stream preserve dependencies while the CPU can enqueue work asynchronously.

## Self-Check Questions

1. In File 1's EAGER mode, why does op 2 count as a full memory read even though its input (`tmp1`) was written by op 1 just one loop iteration earlier?
2. File 1 checks that EAGER and FUSED produce bit-identical output, not merely close output. Why is bit-identical the right expectation here, when the sibling multi-GPU book found real 1-bit mismatches between differently-ordered reductions?
3. What does `fused(N) = 2 * array_bytes` (constant, independent of N) mean physically about where an intermediate value between fused operations actually lives while the program runs?
4. File 2 uses NVIDIA's cited 9.6us figure rather than a number this book measured itself. What specifically about this book's own authoring environment makes measuring that number directly impossible?
5. NVIDIA's blog post also reports a lower, 3.8us-per-kernel figure once synchronization is moved from after every single kernel to after every batch of kernels. Why does File 2 use the stricter, per-kernel-synchronized 9.6us figure instead, given that a lower number was also available?
6. Section 1.2's closing text refuses to combine the launch-overhead reduction and the memory-traffic reduction into one predicted speedup number. What additional, unverified assumption would that combination require?
7. Why does the [COMMON TRAP] in Section 1.3 say a chain of operations with heavier per-element compute "does not automatically escape" this chapter's memory-traffic argument, rather than saying it never applies to such a chain?
8. If a chain had 20 operations instead of 8, what would File 1's closed-form formula predict for the memory-traffic reduction ratio, without running the program again?

## Where We Go Next

Chapter 1 established two real, countable costs that eager execution pays and a compiler can remove. Chapter 2 steps back from CUDA Hammer specifically and asks what a compiler pipeline looks like in general -- the vocabulary (IR, pass, lowering, codegen) that every later chapter in this book, and every real compiler discussed in Part 6, is built out of.

## Worked Solutions

1. Op 2's input is `tmp1`, a `std::vector<float>` that op 1 finished writing and returned. Even though it was produced "just one loop iteration earlier" in the program's own source-code order, at the hardware level it is a fully materialized array sitting in memory (heap-allocated, potentially evicted from cache by the time op 2 runs on a large array) -- op 2 has no way to reach into op 1's registers, because op 1's function call already returned and those registers are gone. Reading it back is a genuine, full memory read, regardless of how recently it was written.
2. The multi-GPU book's 1-bit mismatches came from *reordering a reduction* -- summing the same values in a different grouping, which floating-point addition is not associative under. File 1 never reduces or reorders anything: both EAGER and FUSED apply the exact same sequence of elementary operations (add, multiply, max) to each element, in the exact same order, one element at a time. There is no different grouping to create a mismatch, so bit-identical output is not just expected, it is the only correct outcome -- a mismatch would indicate a real bug in one of the two implementations.
3. It means the value lives in a CPU register (or, in the compiled binary, possibly briefly on the stack, but never in a separately heap-allocated array with its own address that the next operation has to look up and read). The `value` variable inside `runFused`'s inner loop is reused in place across all `numOps` iterations of the inner `for` loop -- it is one scalar, not N arrays.
4. This book's own authoring environment (stated in Getting Started) has a real, complete `nvcc` toolchain on the cloud sandbox but no physical NVIDIA GPU on either of its two authoring machines. Measuring a real kernel-launch overhead number requires actually launching a kernel on real hardware and timing it; `nvcc`-compiled code can be built here, but there is no device to run it on and produce a real timing measurement.
5. It is a deliberately pessimistic reference pattern with synchronization after every kernel. Eager execution does not require this: the host may queue dependent operations on one CUDA stream without synchronizing after each. NVIDIA's 3.8-microsecond result illustrates reduced exposed overhead when synchronization is less frequent. Both values include kernel execution, so neither is an isolated launch-overhead measurement.
6. It would require an assumed memory bandwidth (to convert "MB of traffic" into a time) and an assumed relationship between how launch-overhead time and memory-traffic time actually combine on real hardware (do they overlap, or add serially, and by how much) -- neither of which this book has a real measurement for, since it has no physical GPU to measure either on.
7. Because the memory-traffic argument's applicability depends on a ratio (arithmetic intensity: how much compute happens per byte moved), not on the mere presence of "heavier" compute. A chain could add meaningfully more compute per element and still be far below the ratio needed to become compute-bound rather than memory-bound -- "heavier" alone doesn't say where a specific chain actually lands on that ratio, which is exactly why Chapter 12 builds a real, quantitative test (the roofline model) instead of relying on a qualitative impression.
8. `eager(20) = 20 * 2 * array_bytes` and `fused = 2 * array_bytes` regardless of N, so the reduction ratio is `eager(20) / fused = 20`. The formula (already checked against genuinely counted bytes for N = 1 through 8 in File 1's own output) predicts a 20x reduction at N = 20, with no need to rerun the program to know that.

---

**Sources cited in this chapter:**

- NVIDIA Technical Blog, ["Getting Started with CUDA Graphs"](https://developer.nvidia.com/blog/cuda-graphs/) -- historical V100 end-to-end per-kernel latency with and without per-kernel synchronization; these values include GPU execution.
