# 12. Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels

**What you will understand:** arithmetic intensity -- the ratio of total FLOPs to total bytes moved -- and why that ratio, not either quantity alone, determines whether a kernel's runtime is limited by how fast a GPU can compute or by how fast it can move data; the roofline model, which turns that ratio into a concrete classification using one real GPU's own published peak numbers; and, measured on both a hand-picked chain and a real `Graph` from this book's own IR, exactly how much fusion raises that ratio -- the quantitative case Part 3's actual fusion passes exist to act on.

**What you need to know first:** Chapter 1's memory-traffic model (`eager(N) = N * 2 * array_bytes` vs. `fused = 2 * array_bytes`), Chapter 4's `Value`/`Node`/`Graph` and `topologicalSort()`, Chapter 6's `Shape` and `inferShapes()`, and Chapter 9's "the last node is the graph's own output" convention.

---

Chapter 1 counted bytes: an eager chain of `K` operations over `N` elements moves `K * 2 * N * array_bytes` in total, against a single fused kernel's `2 * N * array_bytes`. That comparison was honest as far as it went, but it stopped at a ratio of totals -- "fusion moves `K` times less data" -- without ever asking whether either number, on its own, says anything about how fast either version actually runs on real hardware. Chapter 1's own Worked Solution #7 named the gap directly: a chain could do meaningfully more compute per byte and still fall nowhere near the point where that extra compute actually changes which resource -- the compute engine or the memory bus -- is the one holding the kernel back. This chapter builds the quantitative test that question needs: arithmetic intensity, and the roofline model that classifies it against one real GPU's own published limits.

This is Part 3's own opening chapter, and it plays the same role for operator fusion that Chapter 1 played for the whole book: establishing, with real counted and cited evidence, exactly what the next several chapters are trying to save. Chapters 13 through 16 build the actual fusion passes -- merging elementwise chains, fusing around reductions, loop fusion and tiling, and where fusion has to stop. None of that machinery is justified yet without first answering the question this chapter asks: fusion changes *where* and *how often* data crosses memory, but by how much, and does it actually matter?

```text
WHAT THIS CHAPTER ADDS TO CHAPTER 1's OWN ARGUMENT:

  Chapter 1 (byte totals):              Chapter 12 (a ratio, then a threshold):

    eager(N)  = K * 2 * N * bytes         AI_unfused = totalFLOPs / eager(N)
    fused     = 2 * N * bytes             AI_fused   = totalFLOPs / fused
    "fusion moves K times less data"      "fusion's ratio crosses a REAL GPU's
                                            own ridge point; the byte totals
                                            alone never said whether that
                                            mattered"

  PART 3 -- OPERATOR FUSION:
    12. Why Fusion Matters (this chapter)   -- quantifies what there is to save
    13. Elementwise Fusion                  -- the first real fusion PASS
    14. Reduction Fusion
    15. Loop Fusion and Tiling
    16. Fusion Boundaries                   -- where fusion has to stop
```

## 12.1 Arithmetic Intensity: The Ratio That Predicts Performance

### Intuition

Picture two warehouse workers doing the exact same total job: moving 1,000 boxes from a loading dock to a shelf. The first worker carries one box at a time -- walk to the dock, pick up a box, walk to the shelf, set it down, walk back -- a thousand round trips for a thousand boxes. The second worker uses a hand truck that holds 50 boxes at once -- twenty round trips, same thousand boxes moved, same total "work" in the sense of boxes shelved. Counting only "boxes shelved" (the total work) says nothing about who finishes first; counting only "round trips" (the total travel) says nothing about it either, since a worker who somehow shelved zero boxes would need zero trips. What actually predicts who finishes first is the *ratio* -- boxes shelved per round trip. That ratio is what arithmetic intensity measures for a kernel: not total FLOPs, not total bytes moved, but FLOPs *per byte*, computed as `totalFLOPs / totalBytesMoved`.

### Background

Chapter 1's own chain -- `K` elementwise operations applied one after another to an `N`-element array -- is the same chain this section reuses, now measured as a ratio instead of a pair of totals. Two things need counting for any version of this chain: the total arithmetic (how many floating-point operations it performs, treating every elementwise op as one FLOP per output element, the same simplification Chapter 1 used to avoid needing to distinguish op types for its own byte count) and the total data movement (how many bytes actually cross to and from memory, not how many bytes merely exist in some array).

```text
UNFUSED (K separate kernels) vs. FUSED (one kernel), for a chain of length K
over N elements:

  UNFUSED:  kernel 1        kernel 2               kernel K
            read N -+       read N -+              read N -+
            write N-+->mem  write N-+->mem   ...   write N-+->mem
            (2N bytes)      (2N bytes)               (2N bytes)

            totalFLOPs = K*N        totalBytes = K * 2N * bytesPerElement
            AI_unfused = (K*N) / (K*2N*bytesPerElement) = 1 / (2*bytesPerElement)
                         ^^^ the K cancels out completely -- constant, for any K

  FUSED:    one kernel, x lives in a register between every step
            read N (the FIRST input) -+                        +- write N (the LAST output)
                                       |   K steps, all in-reg  |
                                       +------------------------+
            (2N bytes total, regardless of K)

            totalFLOPs = K*N         totalBytes = 2N * bytesPerElement
            AI_fused   = (K*N) / (2N*bytesPerElement) = K / (2*bytesPerElement)
                         ^^^ grows LINEARLY with K -- no cancellation this time
```

With 4-byte float elements, `AI_unfused = 1/8 = 0.125` FLOPs/byte -- for *any* chain length `K` at all, because every extra operation in the unfused version brings its own extra `2N` bytes of traffic along with it, in exact lockstep with its own extra `N` FLOPs. `AI_fused = K/8`, which keeps climbing as `K` grows, because a fused chain's traffic never grows past its first read and its last write no matter how many operations happen in between. File 024 below computes both formulas directly (no simulation, no timing -- these are closed-form ratios) across a range of chain lengths and confirms, by direct computation rather than by asserting it, that the unfused ratio never moves while the fused ratio never stops climbing.

```cpp
// Chapter 12: Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels
// 024_arithmetic_intensity_of_an_unfused_vs_fused_chain.cpp
//
// Section 12.1 -- arithmetic intensity, the RATIO (not either total by
// itself) that actually predicts whether a kernel's runtime is limited
// by how fast it can compute or by how fast it can move data.
//
// Chapter 1 counted TOTAL bytes moved by an eager/unfused chain of
// length K against a single fused kernel's own total and stopped there
// (eager(N) = N * 2 * array_bytes vs. fused = 2 * array_bytes). This
// chapter divides each of those same totals by the chain's own total
// FLOP count, producing arithmetic intensity: FLOPs per byte moved.
// Chapter 1's own Worked Solution #7 named exactly this ratio as the
// missing piece needed to say anything quantitative about where a chain
// actually lands, rather than just how much traffic it moves -- this is
// that quantitative test.
//
// Every op in the chain below is treated as costing exactly 1 FLOP per
// output element -- a deliberate simplification (a real ReLU is a
// compare-and-select, not a floating add or multiply) made for the same
// reason Chapter 1 didn't need to distinguish op types for its own
// memory-traffic count: the point here is the SHAPE of the ratio
// (constant vs. growing with chain length), not any one op's exact cost.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 024_arithmetic_intensity_of_an_unfused_vs_fused_chain.cpp -o 024_arithmetic_intensity_of_an_unfused_vs_fused_chain
// Run:     ./024_arithmetic_intensity_of_an_unfused_vs_fused_chain
#include <cstdio>
#include <cmath>
#include <vector>

// One float32 element -- matching Chapter 1's own 4-byte-per-element
// assumption, so this chapter's numbers stay directly comparable to it.
static constexpr double kBytesPerElement = 4.0;

// A chain of K unary elementwise ops over N elements each
// (y1 = op(x), y2 = op(y1), ..., yK = op(y_{K-1})):
//
//   UNFUSED (K separate kernels): kernel i reads its own N-element input
//   and writes its own N-element output from/to memory -- 2*N elements
//   moved, K separate times, once per kernel launch.
//     totalFLOPs = K * N                              (the work itself)
//     totalBytes = K * 2 * N * bytesPerElement         (K round trips to memory)
//
//   FUSED (one kernel): only the chain's very first input is ever read
//   from memory and only its very last output is ever written back --
//   every intermediate y_i lives in a register for the one instruction
//   that produces AND immediately consumes it, never touching memory.
//     totalFLOPs = K * N                              (identical work)
//     totalBytes = 2 * N * bytesPerElement             (ONE round trip, regardless of K)
static double unfusedArithmeticIntensity(long long K, long long N) {
    double totalFlops = static_cast<double>(K) * static_cast<double>(N);
    double totalBytes = static_cast<double>(K) * 2.0 * static_cast<double>(N) * kBytesPerElement;
    return totalFlops / totalBytes;
}

static double fusedArithmeticIntensity(long long K, long long N) {
    double totalFlops = static_cast<double>(K) * static_cast<double>(N);
    double totalBytes = 2.0 * static_cast<double>(N) * kBytesPerElement;
    return totalFlops / totalBytes;
}

int main() {
    printf("=== Section 12.1: arithmetic intensity of an unfused vs. a fused chain ===\n\n");

    const long long N = 1'000'000;  // element count -- cancels out of the ratio entirely, shown for several K
    std::vector<long long> testChainLengths = {1, 2, 4, 8, 16, 32, 64, 100, 101, 128, 1000, 1'000'000};

    printf("  %-10s %-24s %-24s\n", "K", "AI_unfused (FLOPs/byte)", "AI_fused (FLOPs/byte)");
    double firstUnfusedAI = unfusedArithmeticIntensity(testChainLengths[0], N);
    bool unfusedAIConstant = true;
    bool fusedAIStrictlyIncreasing = true;
    double prevFusedAI = -1.0;
    for (long long K : testChainLengths) {
        double aiUnfused = unfusedArithmeticIntensity(K, N);
        double aiFused = fusedArithmeticIntensity(K, N);
        printf("  %-10lld %-24.6f %-24.6f\n", K, aiUnfused, aiFused);
        if (std::fabs(aiUnfused - firstUnfusedAI) > 1e-9) unfusedAIConstant = false;
        if (aiFused <= prevFusedAI) fusedAIStrictlyIncreasing = false;
        prevFusedAI = aiFused;
    }

    printf("\nself-check: AI_unfused is EXACTLY %.6f FLOPs/byte for every K tested, independent of\n", firstUnfusedAI);
    printf("chain length and of N (%s)\n", unfusedAIConstant ? "confirmed" : "MISMATCH");
    printf("self-check: AI_fused is strictly increasing as K grows (%s)\n",
           fusedAIStrictlyIncreasing ? "confirmed" : "MISMATCH");
    printf("\nBoth totals grow with K -- but their RATIO behaves completely differently: unfused\n");
    printf("traffic grows exactly as fast as unfused work does, so the ratio never moves; fused\n");
    printf("traffic stays flat while fused work keeps growing, so the ratio climbs without bound.\n");

    bool allOk = unfusedAIConstant && fusedAIStrictlyIncreasing;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 024_arithmetic_intensity_of_an_unfused_vs_fused_chain.cpp -o 024_arithmetic_intensity_of_an_unfused_vs_fused_chain
./024_arithmetic_intensity_of_an_unfused_vs_fused_chain
```

**Output:**

```text
=== Section 12.1: arithmetic intensity of an unfused vs. a fused chain ===

  K          AI_unfused (FLOPs/byte)  AI_fused (FLOPs/byte)   
  1          0.125000                 0.125000                
  2          0.125000                 0.250000                
  4          0.125000                 0.500000                
  8          0.125000                 1.000000                
  16         0.125000                 2.000000                
  32         0.125000                 4.000000                
  64         0.125000                 8.000000                
  100        0.125000                 12.500000               
  101        0.125000                 12.625000               
  128        0.125000                 16.000000               
  1000       0.125000                 125.000000              
  1000000    0.125000                 125000.000000           

self-check: AI_unfused is EXACTLY 0.125000 FLOPs/byte for every K tested, independent of
chain length and of N (confirmed)
self-check: AI_fused is strictly increasing as K grows (confirmed)

Both totals grow with K -- but their RATIO behaves completely differently: unfused
traffic grows exactly as fast as unfused work does, so the ratio never moves; fused
traffic stays flat while fused work keeps growing, so the ratio climbs without bound.
```

!!! note "A ratio needs a threshold before it means anything"
    `AI_fused` climbing from `0.125` to `125000` FLOPs/byte as `K` grows from 1 to 1,000,000 is a real, computed fact -- but nothing so far says whether any particular value on that climb actually matters. A ratio only becomes an answer once it is compared against something. Section 12.2 supplies that something: a real GPU's own ridge point.

## 12.2 The Roofline Model: A Real GPU's Ridge Point

### Intuition

Picture the warehouse again, but now with two hard limits instead of one worker's technique: the loading dock can only hand off boxes at some fixed maximum rate (bytes per second, off a real memory bus), and the shelving crew can only shelve boxes at some fixed maximum rate too (FLOPs per second, off a real compute engine), and neither limit can be exceeded no matter how the work is organized. A crew doing very few boxes per round trip (low arithmetic intensity) will always be waiting on the next delivery from the dock -- bound by the dock's rate, no matter how fast they can shelve. A crew doing enormous numbers of boxes per round trip (high arithmetic intensity) will eventually be shelving continuously, bound only by their own top shelving speed, with the dock's rate no longer the limiting factor at all. The roofline model draws exactly this as two joined lines on a graph of achievable performance vs. arithmetic intensity: a sloped line (rate limited by the dock, i.e. memory bandwidth) meeting a flat line (rate limited by the crew's own top speed, i.e. peak compute) at one specific point -- the ridge point.

### Background

The roofline model is real, published work: Samuel Williams, Andrew Waterman, and David Patterson, ["Roofline: An Insightful Visual Performance Model for Multicore Architectures,"](https://dl.acm.org/doi/10.1145/1498765.1498785) *Communications of the ACM*, April 2009. It plots a machine's *achievable* performance (FLOPs/second) against a kernel's arithmetic intensity (FLOPs/byte) as two joined lines: a diagonal line, for low arithmetic intensity, where achievable performance is capped by `arithmeticIntensity * peakBandwidth` (you cannot compute faster than data arrives); and a flat line, for high arithmetic intensity, capped by the machine's own `peakFLOPs` outright (you cannot compute faster than the hardware's top speed, no matter how much data is already on hand). NERSC's own roofline documentation describes arithmetic intensity, in the same terms this book has already been using it, as ["the ratio of total floating-point operations (FLOPs) performed by a given code or code section, to the total data movement (Bytes) required to support those FLOPs,"](https://docs.nersc.gov/tools/performance/roofline/) and names the point where those two lines meet the machine's own "machine balance" point -- this book calls it the ridge point, the more common name for the same quantity.

```text
THE ROOFLINE SHAPE (achievable performance vs. arithmetic intensity, log-log axes):

  achievable
  FLOPs/sec
      ^
      |                                    ________________  -- flat roofline:
      |                                   /                    capped at peakFLOPs,
      |                                  /                      REGARDLESS of AI
      |                                 /
      |                                /   -- RIDGE POINT: peakFLOPs / peakBandwidth
      |                               /        (in FLOPs per byte)
      |                              /
      |                        ____ /    -- sloped roofline:
      |                  _____/               capped at AI * peakBandwidth,
      |            _____/                     GROWS as AI grows
      |      _____/
      +-----/------------------------------------------------->
                        arithmetic intensity (FLOPs/byte)

  LEFT of the ridge point:  memory-bound  -- more bandwidth would help; more
                                              raw compute speed would NOT
  RIGHT of the ridge point: compute-bound -- more raw compute speed would
                                              help; more bandwidth would NOT
```

The ridge point itself is just `peakFLOPs / peakBandwidth`, expressed in FLOPs per byte -- the exact arithmetic intensity at which the sloped line's own formula, `AI * peakBandwidth`, first reaches `peakFLOPs`. Below that AI, `AI * peakBandwidth < peakFLOPs`, so the memory bus runs out of data to feed the compute engine before the compute engine ever reaches its own top speed -- memory-bound. At or above that AI, the compute engine is already the tighter constraint -- compute-bound. File 025 below reuses Section 12.1's two arithmetic-intensity functions completely unchanged and computes a real ridge point from one real, cited GPU's own published numbers: NVIDIA's A100 (40GB PCIe) datasheet states FP32 peak throughput of 19.5 TFLOPS and memory bandwidth of 1,555 GB/s.

```cpp
// Chapter 12: Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels
// 025_the_roofline_model_and_a_real_gpus_ridge_point.cpp
//
// Section 12.2 -- the roofline model's ridge point, computed here from
// one real GPU's own published peak numbers, that turns Section 12.1's
// arithmetic-intensity ratio into a concrete memory-bound-or-compute-
// bound classification -- the roofline model itself is real, published
// work: Samuel Williams, Andrew Waterman, and David Patterson,
// "Roofline: An Insightful Visual Performance Model for Multicore
// Architectures," Communications of the ACM, April 2009.
//
// Reuses Section 12.1's unfusedArithmeticIntensity()/fusedArithmeticIntensity()
// completely unchanged -- this section's only new idea is a THRESHOLD to
// compare that ratio against.
//
// Real external numbers used below, cited to NVIDIA's own published A100
// datasheet (40GB PCIe SKU): FP32 peak throughput 19.5 TFLOPS, memory
// bandwidth 1,555 GB/s. The ridge point -- the arithmetic intensity at
// which a kernel exactly saturates BOTH the compute engine and the
// memory bus at once -- is peak FLOPs/s divided by peak bytes/s, computed
// here from those two cited numbers rather than looked up as a separate
// published figure.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 025_the_roofline_model_and_a_real_gpus_ridge_point.cpp -o 025_the_roofline_model_and_a_real_gpus_ridge_point
// Run:     ./025_the_roofline_model_and_a_real_gpus_ridge_point
#include <cstdio>
#include <cmath>
#include <vector>

// ==================== Section 12.1's arithmetic intensity (unchanged) ====================

static constexpr double kBytesPerElement = 4.0;

static double unfusedArithmeticIntensity(long long K, long long N) {
    double totalFlops = static_cast<double>(K) * static_cast<double>(N);
    double totalBytes = static_cast<double>(K) * 2.0 * static_cast<double>(N) * kBytesPerElement;
    return totalFlops / totalBytes;
}

static double fusedArithmeticIntensity(long long K, long long N) {
    double totalFlops = static_cast<double>(K) * static_cast<double>(N);
    double totalBytes = 2.0 * static_cast<double>(N) * kBytesPerElement;
    return totalFlops / totalBytes;
}

// ==================== Section 12.2: a real GPU's ridge point (new) ====================
//
// Cited directly to NVIDIA's own A100 datasheet (40GB PCIe SKU):
//   FP32 peak throughput: 19.5 TFLOPS  = 19.5e12 FLOPs/second
//   Memory bandwidth:     1,555 GB/s   = 1.555e12 bytes/second
static constexpr double kA100Fp32PeakFlopsPerSec = 19.5e12;
static constexpr double kA100BandwidthBytesPerSec = 1555.0e9;

static double ridgePointFlopsPerByte() {
    return kA100Fp32PeakFlopsPerSec / kA100BandwidthBytesPerSec;
}

static const char* classify(double arithmeticIntensity, double ridgePoint) {
    return (arithmeticIntensity >= ridgePoint) ? "compute-bound" : "memory-bound";
}

int main() {
    printf("=== Section 12.2: classifying Section 12.1's chains against a real GPU's ridge point ===\n\n");

    double ridge = ridgePointFlopsPerByte();
    printf("NVIDIA A100 (40GB PCIe) ridge point = %.4f TFLOPS / %.4f GB/s = %.4f FLOPs/byte\n",
           kA100Fp32PeakFlopsPerSec / 1e12, kA100BandwidthBytesPerSec / 1e9, ridge);

    const long long N = 1'000'000;
    std::vector<long long> testChainLengths = {1, 2, 4, 8, 16, 32, 64, 100, 101, 128, 1000, 1'000'000};

    printf("\n  %-10s %-16s %-16s %-16s %-16s\n", "K", "AI_unfused", "class", "AI_fused", "class");
    bool unfusedAlwaysMemoryBound = true;
    bool someFusedIsComputeBound = false;
    double firstUnfusedAI = unfusedArithmeticIntensity(testChainLengths[0], N);
    for (long long K : testChainLengths) {
        double aiUnfused = unfusedArithmeticIntensity(K, N);
        double aiFused = fusedArithmeticIntensity(K, N);
        const char* cu = classify(aiUnfused, ridge);
        const char* cf = classify(aiFused, ridge);
        printf("  %-10lld %-16.6f %-16s %-16.6f %-16s\n", K, aiUnfused, cu, aiFused, cf);
        if (aiUnfused >= ridge) unfusedAlwaysMemoryBound = false;
        if (aiFused >= ridge) someFusedIsComputeBound = true;
    }

    printf("\nself-check: the unfused chain is memory-bound at EVERY tested K, including K=1,000,000\n");
    printf("(its AI never grows) (%s)\n", unfusedAlwaysMemoryBound ? "confirmed" : "MISMATCH");
    printf("self-check: at least one tested K makes the FUSED chain compute-bound (%s)\n",
           someFusedIsComputeBound ? "confirmed" : "MISMATCH");

    // Smallest integer K at which the fused chain crosses the ridge point:
    // AI_fused(K) = K / (2 * bytesPerElement) >= ridge  =>  K >= ridge * 2 * bytesPerElement
    long long crossoverK = static_cast<long long>(std::ceil(ridge * 2.0 * kBytesPerElement));
    bool crossoverKIsComputeBound = fusedArithmeticIntensity(crossoverK, N) >= ridge;
    bool oneLessIsNotComputeBound = fusedArithmeticIntensity(crossoverK - 1, N) < ridge;
    printf("\nSmallest fused chain length that crosses this GPU's ridge point: K = %lld\n", crossoverK);
    printf("self-check: fusedAI(K=%lld) is compute-bound (%s), fusedAI(K=%lld) is NOT (%s)\n",
           crossoverK, crossoverKIsComputeBound ? "confirmed" : "MISMATCH",
           crossoverK - 1, oneLessIsNotComputeBound ? "confirmed" : "MISMATCH");
    printf("No UNFUSED chain of any length ever crosses this ridge point -- its arithmetic\n");
    printf("intensity is fixed at %.6f FLOPs/byte forever, by construction.\n", firstUnfusedAI);

    bool allOk = unfusedAlwaysMemoryBound && someFusedIsComputeBound &&
                 crossoverKIsComputeBound && oneLessIsNotComputeBound;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 025_the_roofline_model_and_a_real_gpus_ridge_point.cpp -o 025_the_roofline_model_and_a_real_gpus_ridge_point
./025_the_roofline_model_and_a_real_gpus_ridge_point
```

**Output:**

```text
=== Section 12.2: classifying Section 12.1's chains against a real GPU's ridge point ===

NVIDIA A100 (40GB PCIe) ridge point = 19.5000 TFLOPS / 1555.0000 GB/s = 12.5402 FLOPs/byte

  K          AI_unfused       class            AI_fused         class           
  1          0.125000         memory-bound     0.125000         memory-bound    
  2          0.125000         memory-bound     0.250000         memory-bound    
  4          0.125000         memory-bound     0.500000         memory-bound    
  8          0.125000         memory-bound     1.000000         memory-bound    
  16         0.125000         memory-bound     2.000000         memory-bound    
  32         0.125000         memory-bound     4.000000         memory-bound    
  64         0.125000         memory-bound     8.000000         memory-bound    
  100        0.125000         memory-bound     12.500000        memory-bound    
  101        0.125000         memory-bound     12.625000        compute-bound   
  128        0.125000         memory-bound     16.000000        compute-bound   
  1000       0.125000         memory-bound     125.000000       compute-bound   
  1000000    0.125000         memory-bound     125000.000000    compute-bound   

self-check: the unfused chain is memory-bound at EVERY tested K, including K=1,000,000
(its AI never grows) (confirmed)
self-check: at least one tested K makes the FUSED chain compute-bound (confirmed)

Smallest fused chain length that crosses this GPU's ridge point: K = 101
self-check: fusedAI(K=101) is compute-bound (confirmed), fusedAI(K=100) is NOT (confirmed)
No UNFUSED chain of any length ever crosses this ridge point -- its arithmetic
intensity is fixed at 0.125000 FLOPs/byte forever, by construction.
```

!!! note "The unfused chain never crosses the ridge point, at ANY length"
    This is the sharpest way to state this chapter's whole argument: an *unfused* elementwise chain is memory-bound on this real GPU no matter how long it gets -- adding more operations adds proportionally more traffic right along with the extra work, so the ratio never moves. A *fused* chain of the same operations eventually becomes compute-bound, at a real, computed threshold (`K = 101` on this specific GPU) -- not because the work changed, but because fusion is the only one of the two versions whose traffic stops growing while its work keeps growing. This is precisely the gap Part 3's fusion passes exist to close.

## 12.3 Measuring Fusion's Effect on CUDA Hammer's Own Graph

### Intuition

Sections 12.1 and 12.2 both worked from one convenient, hand-picked shape: a straight chain, where every node has exactly one consumer and exactly one producer. A real tensor graph is rarely that tidy -- Chapter 4's own very first example graph, the diamond built in File 005, has a node (`t1`) with *two* consumers. An unfused kernel for each of `t1`'s two consumers has to read `t1` from memory separately, once per consumer -- a second, different kind of wasted traffic that a straight chain's analysis never exercises at all. Measuring fusion's benefit on a real `Graph`, not just a formula, means confronting that case directly.

### Background

This section reuses Chapter 4's exact diamond graph (`a`, `b` = inputs; `t1 = Add(a, b)`; `t2 = Mul(t1, a)`; `t3 = ReLU(t1)`; `out = Add(t2, t3)`) and Chapter 6's own declared leaf shapes for it (`a`: `[3, 4]`, `b`: `[4]`, broadcasting at the very first `Add`), run through Chapter 6's `inferShapes()` completely unchanged to get every node's real element count. From there, counting bytes moved is a direct walk of the graph, not a formula: an *unfused* kernel exists per non-leaf node, reading every one of that node's own operands from memory and writing its own output back; a *fused* kernel reads each **distinct** leaf exactly once, no matter how many downstream nodes consume it, and writes only the graph's own designated output (Chapter 9's "the last node is the output" convention) exactly once.

```text
WHY A SHARED NODE COSTS MORE THAN A CHAIN'S ANALYSIS ALONE WOULD SUGGEST:

  UNFUSED:  a  -->[kernel t1]--> t1 -->[kernel t2]--> t2 -+
            |                    |                        |->[kernel out]--> out
            +----(read AGAIN)----+-->[kernel t3]--> t3 ---+
                                  ^
                          t1 read from memory TWICE:
                          once for t2's kernel, once for t3's kernel

  FUSED:    a --+                                                 +--> out
            b --+--> [ONE kernel: t1, t2, t3, out all in registers]
                       t1 computed ONCE, used TWICE, never
                       written to or read back from memory at all
```

`totalFLOPs` is identical either way -- the graph performs the exact same four operations (`t1`, `t2`, `t3`, `out`) regardless of how they're scheduled into kernels; fusion changes only how often data crosses memory, never how much arithmetic happens, the same point Chapter 1 made about its own chain. File 026 below counts both totals directly from the graph's own inferred shapes and computes both arithmetic intensities, classified against the same real A100 ridge point Section 12.2 already computed.

```cpp
// Chapter 12: Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels
// 026_measuring_fusions_effect_on_cuda_hammers_own_graph.cpp
//
// Section 12.3 -- everything Sections 12.1 and 12.2 derived by formula,
// on a hand-picked straight-line CHAIN, measured here instead on a real
// CUDA Hammer Graph -- specifically the exact diamond graph Chapter 4's
// File 005 built by hand and Chapter 6's File 009 already inferred
// shapes over. A diamond is a strictly harder case than a chain: node
// 'a' and node 't1' each have TWO consumers, so an unfused kernel for
// each of their consumers reads that shared value from memory AGAIN,
// while a fused kernel reads it from memory exactly ONCE no matter how
// many downstream ops consume it -- fusion's benefit on a real DAG comes
// from eliminating both kinds of repeated traffic (repeated-tensor-in-a-
// chain AND shared-value-with-multiple-consumers), not just the first.
//
// Reuses Chapter 4's Value/Node/Graph and topologicalSort() (Kahn's
// algorithm) and Chapter 6's Shape/broadcastShapes()/inferShapes()
// completely unchanged -- this file's only new idea is walking that
// already-inferred shape table to count bytes moved, the same way
// Chapter 9's deadCodeEliminationPass() walked node->inputs to count
// liveness instead of shape.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 026_measuring_fusions_effect_on_cuda_hammers_own_graph.cpp -o 026_measuring_fusions_effect_on_cuda_hammers_own_graph
// Run:     ./026_measuring_fusions_effect_on_cuda_hammers_own_graph
#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <deque>
#include <algorithm>
#include <stdexcept>

// ==================== Value / Node / Graph (from Chapter 4, unchanged) ====================

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

// ==================== Topological sort (from Chapter 4's File 006, unchanged) ====================

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

// ==================== Shape / broadcastShapes / inferShapes (from Chapter 6, unchanged) ====================

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

static long long numElements(const Shape& s) {
    long long n = 1;
    for (int d : s.dims) n *= d;
    return n;
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
        std::string context = "node %" + std::to_string(id) + " (" + n->debugName + ")";
        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            shapes[id] = declaredShapes.at(id);
        } else if (n->op == OpKind::Add || n->op == OpKind::Mul) {
            shapes[id] = broadcastShapes(shapes.at(n->inputs[0].nodeId), shapes.at(n->inputs[1].nodeId), context);
        } else {  // ReLU -- shape-preserving
            shapes[id] = shapes.at(n->inputs[0].nodeId);
        }
    }
    return shapes;
}

// ==================== Section 12.2's ridge point (from File 025, unchanged) ====================

static constexpr double kBytesPerElement = 4.0;
static constexpr double kA100Fp32PeakFlopsPerSec = 19.5e12;
static constexpr double kA100BandwidthBytesPerSec = 1555.0e9;
static double ridgePointFlopsPerByte() { return kA100Fp32PeakFlopsPerSec / kA100BandwidthBytesPerSec; }
static const char* classify(double ai, double ridge) { return (ai >= ridge) ? "compute-bound" : "memory-bound"; }

// ==================== Section 12.3: counting bytes moved, unfused vs. fused ====================
//
// UNFUSED: each non-leaf node is its own kernel -- it reads every one of
// its own operands from memory (even if some other node already read
// that SAME operand a moment ago) and writes its own output back to
// memory. A shared value like 'a' or 't1' therefore gets re-read from
// memory once per consumer.
static long long unfusedBytesMoved(const Graph& g, const std::map<int, Shape>& shapes) {
    long long totalElements = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;  // a leaf has nothing to compute
        for (const Value& in : n->inputs) {
            totalElements += numElements(shapes.at(in.nodeId));  // read each operand from memory
        }
        totalElements += numElements(shapes.at(n->id));  // write this node's own output
    }
    return static_cast<long long>(static_cast<double>(totalElements) * kBytesPerElement);
}

// FUSED: a single kernel reads each DISTINCT leaf (Input/Const) from
// memory exactly once -- no matter how many downstream nodes consume it
// -- and writes only the graph's own designated output (Chapter 9's
// "last node is the output" convention) back to memory exactly once.
// Every intermediate value (t1, t2, t3 here) lives in a register for the
// one fused kernel body and never touches memory at all.
static long long fusedBytesMoved(const Graph& g, const std::map<int, Shape>& shapes) {
    long long leafElements = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) {
            leafElements += numElements(shapes.at(n->id));
        }
    }
    const Node* outputNode = g.nodes().back().get();  // Chapter 9's own convention
    long long outputElements = numElements(shapes.at(outputNode->id));
    return static_cast<long long>(static_cast<double>(leafElements + outputElements) * kBytesPerElement);
}

// Total FLOPs is IDENTICAL either way -- fusion changes where and how
// often data crosses memory, never how much arithmetic the graph does.
static long long totalFlops(const Graph& g, const std::map<int, Shape>& shapes) {
    long long total = 0;
    for (const auto& n : g.nodes()) {
        if (n->op == OpKind::Input || n->op == OpKind::Const) continue;
        total += numElements(shapes.at(n->id));  // one FLOP per output element, same simplification as File 024
    }
    return total;
}

int main() {
    printf("=== Section 12.3: Chapter 4's diamond graph, with Chapter 6's own declared shapes ===\n\n");

    // The exact diamond from Chapter 4's File 005 / Chapter 6's File 009:
    //   a, b = Input, Input
    //   t1   = Add(a, b)
    //   t2   = Mul(t1, a)      <- t1 used again here
    //   t3   = ReLU(t1)        <- and again here: t1 has TWO consumers
    //   out  = Add(t2, t3)
    Graph g;
    Value a  = g.addInput("a");
    Value b  = g.addInput("b");
    Value t1 = g.addBinary(OpKind::Add, a, b, "t1");
    Value t2 = g.addBinary(OpKind::Mul, t1, a, "t2");
    Value t3 = g.addUnary(OpKind::ReLU, t1, "t3");
    Value out = g.addBinary(OpKind::Add, t2, t3, "out");
    (void)out;

    // Same declared leaf shapes as Chapter 6's own File 009 -- 'a' is
    // [3, 4], 'b' is [4] and broadcasts against it at the very first Add.
    std::map<int, Shape> declared = {
        {a.nodeId, Shape{{3, 4}}},
        {b.nodeId, Shape{{4}}},
    };
    std::map<int, Shape> shapes = inferShapes(g, declared);

    for (const auto& n : g.nodes()) {
        printf("  %%%d = %s(%s): shape %s (%lld elements)\n", n->id, opKindStr(n->op).c_str(),
               n->debugName.c_str(), shapeStr(shapes.at(n->id)).c_str(), numElements(shapes.at(n->id)));
    }

    long long flops = totalFlops(g, shapes);
    long long unfusedBytes = unfusedBytesMoved(g, shapes);
    long long fusedBytes = fusedBytesMoved(g, shapes);

    printf("\nTotal FLOPs (identical either way, fusion doesn't change the arithmetic): %lld\n", flops);
    printf("Bytes moved, UNFUSED (each of t1/t2/t3/out is its own kernel):  %lld\n", unfusedBytes);
    printf("Bytes moved, FUSED   (one kernel, each leaf read once, 'out' written once): %lld\n", fusedBytes);

    bool fusedMovesFewerBytes = fusedBytes < unfusedBytes;
    printf("\nself-check: the fused kernel moves strictly fewer bytes than the unfused version (%s)\n",
           fusedMovesFewerBytes ? "confirmed" : "MISMATCH");

    double aiUnfused = static_cast<double>(flops) / static_cast<double>(unfusedBytes);
    double aiFused = static_cast<double>(flops) / static_cast<double>(fusedBytes);
    double ridge = ridgePointFlopsPerByte();

    printf("\nAI_unfused = %lld / %lld bytes = %.6f FLOPs/byte  (%s)\n", flops, unfusedBytes, aiUnfused,
           classify(aiUnfused, ridge));
    printf("AI_fused   = %lld / %lld bytes = %.6f FLOPs/byte  (%s)\n", flops, fusedBytes, aiFused,
           classify(aiFused, ridge));
    printf("(ridge point, same A100 numbers as Section 12.2: %.4f FLOPs/byte)\n", ridge);

    bool fusedAIHigher = aiFused > aiUnfused;
    printf("\nself-check: AI_fused is strictly higher than AI_unfused on this real DAG, not just on\n");
    printf("File 024's hand-picked chain (%s) -- fusion's benefit here comes from BOTH eliminating\n",
           fusedAIHigher ? "confirmed" : "MISMATCH");
    printf("chain-style intermediate traffic (t1->t2, t1->t3) AND reading a SHARED value ('a', 't1')\n");
    printf("from memory once instead of once per consumer.\n");

    printf("\nBoth land in the same '%s' classification here -- this diamond is tiny (12 elements\n",
           classify(aiUnfused, ridge));
    printf("per node), far below the K=101-per-chain scale Section 12.2 found necessary to cross this\n");
    printf("GPU's ridge point at all. The ratio improvement is real and measured either way; reaching\n");
    printf("compute-bound territory in practice needs the far larger tensors Part 3's actual fusion\n");
    printf("passes (starting next chapter) will be applied to, not a change to this result.\n");

    bool allOk = fusedMovesFewerBytes && fusedAIHigher;
    return allOk ? 0 : 1;
}
```

**Compile and run:**

```bash
g++ -std=c++17 -Wall -Wextra -O2 026_measuring_fusions_effect_on_cuda_hammers_own_graph.cpp -o 026_measuring_fusions_effect_on_cuda_hammers_own_graph
./026_measuring_fusions_effect_on_cuda_hammers_own_graph
```

**Output:**

```text
=== Section 12.3: Chapter 4's diamond graph, with Chapter 6's own declared shapes ===

  %0 = Input(a): shape [3, 4] (12 elements)
  %1 = Input(b): shape [4] (4 elements)
  %2 = Add(t1): shape [3, 4] (12 elements)
  %3 = Mul(t2): shape [3, 4] (12 elements)
  %4 = ReLU(t3): shape [3, 4] (12 elements)
  %5 = Add(out): shape [3, 4] (12 elements)

Total FLOPs (identical either way, fusion doesn't change the arithmetic): 48
Bytes moved, UNFUSED (each of t1/t2/t3/out is its own kernel):  496
Bytes moved, FUSED   (one kernel, each leaf read once, 'out' written once): 112

self-check: the fused kernel moves strictly fewer bytes than the unfused version (confirmed)

AI_unfused = 48 / 496 bytes = 0.096774 FLOPs/byte  (memory-bound)
AI_fused   = 48 / 112 bytes = 0.428571 FLOPs/byte  (memory-bound)
(ridge point, same A100 numbers as Section 12.2: 12.5402 FLOPs/byte)

self-check: AI_fused is strictly higher than AI_unfused on this real DAG, not just on
File 024's hand-picked chain (confirmed) -- fusion's benefit here comes from BOTH eliminating
chain-style intermediate traffic (t1->t2, t1->t3) AND reading a SHARED value ('a', 't1')
from memory once instead of once per consumer.

Both land in the same 'memory-bound' classification here -- this diamond is tiny (12 elements
per node), far below the K=101-per-chain scale Section 12.2 found necessary to cross this
GPU's ridge point at all. The ratio improvement is real and measured either way; reaching
compute-bound territory in practice needs the far larger tensors Part 3's actual fusion
passes (starting next chapter) will be applied to, not a change to this result.
```

!!! warning "A higher ratio is not automatically a different classification"
    `AI_fused` (`0.428571`) is roughly 4.4x higher than `AI_unfused` (`0.096774`) on this exact graph -- a real, measured improvement -- but both numbers still land on the memory-bound side of this GPU's ridge point (`12.5402`). Fusion's benefit is real at any scale; whether that benefit is enough to change a kernel's classification depends on how far the *original* ratio was from the ridge point to begin with, which Section 12.2's own `K = 101` crossover already showed can take a genuinely large amount of fused work to reach. A toy 12-element diamond graph was never going to cross that threshold -- Part 3's actual fusion passes will be applied to tensors many orders of magnitude larger than this one.

## Chapter Summary

This chapter opened Part 3 by building the quantitative test Chapter 1's own byte-counting argument was missing: arithmetic intensity, the ratio of total FLOPs to total bytes moved, rather than either total alone. Section 12.1 showed that an unfused chain's arithmetic intensity is a fixed constant (`1 / (2 * bytesPerElement)`) regardless of chain length, while a fused chain's arithmetic intensity grows linearly with it -- the same underlying formulas Chapter 1 used for its own byte totals, now expressed as a ratio that behaves completely differently under growth. Section 12.2 introduced the roofline model (Williams, Waterman, and Patterson's real, published 2009 work) and computed a real ridge point -- `peakFLOPs / peakBandwidth` -- from one real, cited GPU's own numbers (NVIDIA's A100 datasheet), finding by direct computation that an unfused chain never crosses that ridge point at any length, while a fused chain crosses it at a specific, computed threshold (`K = 101` on this GPU). Section 12.3 moved from a hand-picked chain to a real `Graph` -- Chapter 4's own diamond, reusing Chapter 6's `inferShapes()` unchanged -- and measured fusion's benefit directly from the graph's structure rather than from a formula, confirming the same ratio improvement holds on a DAG with a genuinely shared node, even though this particular toy graph is too small to cross into compute-bound territory itself.

## Self-Check Questions

1. Chapter 1 already computed `eager(N)` and `fused` as byte totals. What new quantity does this chapter compute from those same totals, and why does the chapter argue that quantity matters more than either total alone?
2. Why is `AI_unfused` for an elementwise chain exactly `1 / (2 * bytesPerElement)`, regardless of how long the chain is?
3. Why does `AI_fused` keep growing as chain length `K` grows, when `AI_unfused` does not?
4. What two real, cited numbers does Section 12.2 use to compute the A100's ridge point, and what does dividing one by the other actually represent?
5. Section 12.2 finds that an unfused chain is memory-bound at every tested chain length, including `K = 1,000,000`. Why does adding more operations never change that classification?
6. Section 12.3 measures fusion's benefit on a real DAG (Chapter 4's diamond) rather than a straight chain. What kind of wasted traffic does the diamond's shared node (`t1`) expose that a straight chain's own analysis never exercises?
7. In Section 12.3's output, `AI_fused` is roughly 4.4x higher than `AI_unfused`, yet both are still classified `memory-bound`. Is this a contradiction? Why or why not?
8. What is `totalFlops` for the fused version of a graph, compared to the unfused version of that exact same graph, and why?

## Where We Go Next

This chapter established, with real counted and cited evidence, exactly what there is to save: fusion doesn't change how much arithmetic a graph performs, only how much of that arithmetic's supporting data ever has to cross memory, and that difference is large enough, at a real and computed scale, to change which resource -- the memory bus or the compute engine -- actually limits a kernel's performance on real hardware. Chapter 13, "Elementwise Fusion," builds Part 3's first real fusion pass: a `TransformPass`, plugging into the exact same Chapter 8 `PassManager` infrastructure every pass since Chapter 9 has used unmodified, that identifies chains of elementwise operations in CUDA Hammer's own graph and merges them into a single fused node -- turning this chapter's quantitative argument for fusion into an actual, executable transformation.

## Worked Solutions

1. This chapter computes arithmetic intensity: `totalFLOPs / totalBytesMoved`, a single ratio rather than two separate totals. The chapter argues the ratio matters more because neither total alone says anything about achievable performance on real hardware -- a kernel's runtime is limited by whichever of "how fast can data arrive" or "how fast can the compute engine work" is the tighter constraint, and only a ratio of the two totals, compared against a real machine's own limits (Section 12.2's ridge point), can say which one that is.
2. Every unfused kernel in the chain contributes exactly `N` FLOPs and exactly `2*N*bytesPerElement` bytes of traffic (reading its own input, writing its own output). Both of those quantities scale by the exact same factor (`K`, the number of kernels) as the chain grows, so `K` cancels out of the ratio `(K*N) / (K*2*N*bytesPerElement)` completely, leaving `1 / (2*bytesPerElement)` -- a constant that never depended on `K` in the first place.
3. A fused chain's total FLOPs still grows with `K` (`K*N`, identical work to the unfused version), but its total bytes moved does NOT grow with `K` -- it stays fixed at `2*N*bytesPerElement`, because only the very first input is ever read from memory and only the very last output is ever written, regardless of how many operations happen registers-only in between. Growing numerator over a fixed denominator is exactly what makes `AI_fused = K / (2*bytesPerElement)` climb linearly with `K`.
4. NVIDIA's own published A100 datasheet (40GB PCIe SKU): FP32 peak throughput, 19.5 TFLOPS, and memory bandwidth, 1,555 GB/s. Dividing peak FLOPs/second by peak bytes/second gives the ridge point in FLOPs per byte -- the exact arithmetic intensity at which a kernel would need to operate to keep both the compute engine and the memory bus simultaneously at their own absolute peak rates, with neither one idle waiting on the other.
5. Because `AI_unfused` is a constant, `1 / (2*bytesPerElement)`, that never depends on chain length at all (Question 2's own answer) -- and a value that never moves can never cross a fixed threshold like the ridge point, no matter how large `K` becomes. Classification depends entirely on where the ratio sits relative to the ridge point, and this particular ratio sits in exactly the same place regardless of `K`.
6. An unfused kernel for each of `t1`'s two consumers (`t2`'s kernel and `t3`'s kernel) has to read `t1` from memory separately -- `t1` gets read from memory TWICE in the unfused version, once per consumer, even though it was only computed once. A straight chain, where every node has exactly one consumer, never exercises this: its only wasted traffic comes from writing then immediately re-reading each intermediate value exactly once. A fused kernel reads `t1` from memory zero times either way (it never leaves a register), so the diamond's extra "shared value, multiple consumers" cost is exclusively an unfused-version cost, not something a fused kernel ever pays.
7. Not a contradiction -- a higher ratio and an unchanged classification are both real facts about this graph at the same time. Classification depends on where a ratio sits relative to a fixed threshold (the ridge point), not on how much the ratio improved. This graph's `AI_unfused` (`0.096774`) was extremely far below the ridge point (`12.5402`) to begin with; even a genuine 4.4x improvement to `AI_fused` (`0.428571`) is still far short of that threshold. Section 12.2's own `K = 101` result already showed how much fused work it actually takes to close a gap that size -- a 12-element toy graph was never going to get there.
8. They are identical. Fusion is purely a scheduling decision about how many kernels a graph's computation is split into and how often intermediate results touch memory -- it never adds, removes, or changes what arithmetic operation any node in the graph performs. `totalFlops()` in File 026 is computed once, from the graph's own node list and inferred shapes, and used for BOTH the fused and unfused arithmetic-intensity calculations, because the underlying computation genuinely is the same either way.

---

**Sources cited in this chapter:**

- Samuel Williams, Andrew Waterman, and David Patterson, ["Roofline: An Insightful Visual Performance Model for Multicore Architectures,"](https://dl.acm.org/doi/10.1145/1498765.1498785) *Communications of the ACM*, Vol. 52, No. 4, April 2009 -- the roofline model and ridge-point concept used throughout Section 12.2.
- NERSC, ["Roofline Performance Model"](https://docs.nersc.gov/tools/performance/roofline/) -- quoted directly for its definition of arithmetic intensity and its "machine balance point" naming of the ridge point.
- NVIDIA, ["NVIDIA A100 Tensor Core GPU Datasheet"](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) -- the 19.5 TFLOPS FP32 peak throughput and 1,555 GB/s memory bandwidth figures (40GB PCIe SKU) used to compute the ridge point in Sections 12.2 and 12.3.
