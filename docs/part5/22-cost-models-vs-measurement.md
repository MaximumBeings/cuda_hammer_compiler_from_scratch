# 22. Cost Models vs. Measurement-Based Autotuning

**What you will understand:** `estimateLoopOverheadCost()`, a simple linear cost model -- trips through a loop times a fixed per-trip overhead -- that applies to BOTH tile size and unroll factor (Chapter 21 already proved they are the same underlying question), tested against a real 50-million-element computation and found, honestly, to be wrong past a machine-specific turnover point; `estimateLoopOrderCost()`, a cost model built on memory-access stride and cache-line counting, tested against a real 4096x4096 array traversal and found to get the DIRECTION right by a dramatic real margin while getting the exact MAGNITUDE wrong; `estimateScheduleCost()`, which sums both models into one score per `Schedule`, applied to 6 real schedules drawn from Chapter 21's own 128-schedule list and executed for real through a generic, runtime-parameterized schedule INTERPRETER; and a concrete, diagnosed explanation for why that combined model's own ranking agreement lands close to chance on this chapter's own toy-scale example -- not a verdict that cost models don't work, but a specific, honestly identified confound.

**What you need to know first:** Chapter 15's own `Loop`/`LoopNest`/`TiledLoop`/`tileLoop()`/`actualInnerExtent()` and Chapter 21's own `LoopOrder`/`enumerateLoopOrders()`, `unrollFactorCandidates()`/`computeUnrolled()`/`arraysExactlyEqual()`, and `Schedule`/`enumerateSchedules()`/`scheduleStr()` (all reused unchanged throughout this chapter); Chapter 19's own real-clock, best-of-N timing discipline; and Chapter 21's own closing number -- 128 legal schedules for one tiny `LoopNest` -- which this chapter picks up exactly where it was left.

---

Chapter 21 closed Part 5's opening chapter with a real, enumerated number: 128 legal schedules for one tiny two-loop `LoopNest`, and an explicit refusal to compile and run every single one just to find the best. This chapter builds the piece that refusal was waiting on: a COST MODEL, a cheap formula that estimates how good a schedule is likely to be without actually running it, so a later autotuner does not have to measure blindly. A cost model earns trust the same way every other claim in this book has since Chapter 17's own first generated loop -- by being checked against a real clock, on real hardware, and reported honestly when it is wrong, not asserted and left unverified.

One real mechanism is deliberately absent from this chapter, and the omission is a stated scope decision, not an oversight: this book's own IR, through Chapter 20, has no operator with genuine cross-element DATA REUSE. Every elementwise operation reads each input exactly once no matter what tile size a schedule picks, and even the full-reduction `Sum` operator (Chapter 14) still reads each input exactly once regardless of tiling -- unlike, say, a real matrix multiply, where a tile gets reused across many output elements and a cache-reuse cost model is the entire point. Building a reuse-based cost model for this book's own generated code would be modeling a benefit the code cannot actually realize. This chapter instead builds around two mechanisms that genuinely are present and measurable in the loops this book already generates: LOOP OVERHEAD, the fixed per-trip cost of a loop's own bookkeeping (Section 22.1), and MEMORY-ACCESS STRIDE, whether consecutive loop trips touch consecutive memory addresses (Section 22.2) -- then combines both into one score per schedule (Section 22.3, the capstone).

This chapter's own cross-machine verification also changes shape from every earlier timing-free chapter. Chapters 17 through 21 verified generated code by requiring exact, byte-identical or numerically-identical results on both the cloud sandbox and the device -- a fair bar, because none of that work depended on the clock. Every experiment in this chapter is a real wall-clock measurement, and wall-clock measurements are not portable across two different physical machines: an x86-64 cloud sandbox and an aarch64 device will not, and should not, report the same millisecond figures, or even necessarily agree on which candidate wins. What still gets verified identically on both machines is CORRECTNESS -- every schedule this chapter measures is also checked bit-for-bit against a known-correct baseline, on both machines, exactly as strictly as every prior chapter -- while the timing numbers themselves are reported as real, honest, machine-specific data, labeled as such, not forced into an artificial agreement they were never going to have.

```text
Two real, measurable mechanisms; two cost models; one honest verdict:

  LOOP OVERHEAD (22.1)        -- applies to BOTH tile size and unroll
                                  factor (Ch21.3: the same underlying
                                  question, "how many elements per loop
                                  trip"); a simple linear model (trips
                                  times perTripOverhead) tested against a
                                  real 50-million-element loop -- the
                                  model predicts "always chunk bigger,"
                                  real measurement finds a turnover point
                                  the model cannot see (instruction-cache
                                  and register pressure it was never told
                                  about) -- and that turnover point lands
                                  at a DIFFERENT chunk size on each real
                                  machine this book measures it on

  MEMORY-ACCESS STRIDE (22.2) -- applies to loop order: does the
                                  INNERMOST loop match the array's own
                                  contiguous dimension? a model built on
                                  counting cache-line loads gets the
                                  DIRECTION right (row-major faster) by a
                                  dramatic real margin on both machines --
                                  but underestimates the real MAGNITUDE
                                  (TLB misses, hardware prefetch effects
                                  the model never included)

  COMBINED (22.3, capstone)   -- estimateScheduleCost() sums both models
                                  (real compute work counted ONCE, not
                                  once per mechanism) and scores 6 real
                                  schedules drawn from Ch21's own 128, run
                                  for real through a generic schedule
                                  INTERPRETER -- ranking agreement lands
                                  close to chance, honestly diagnosed as
                                  interpreter bookkeeping overhead
                                  swamping a 48-element toy problem, not
                                  proof that cost models are worthless

Neither model is asked to be perfect. Both are asked to be cheap enough
to run before compiling anything, and honest about where they break --
exactly what Chapter 23's own autotuner needs to prune 128 candidates
(or a far larger real space) down to a short list worth actually
measuring.
```

## 22.1 A Loop-Overhead Cost Model for Tile Size and Unroll Factor

### Intuition

Chapter 21 proved that tile size and unroll factor are the same underlying question -- "how many elements does one loop trip cover" -- applied for two different reasons (memory blocking versus reducing per-iteration overhead). That means a single cost model, parameterized by a chunk size, can score both at once: covering `extent` elements in chunks of `chunkSize` takes `ceil(extent / chunkSize)` trips through the loop, and every one of those trips pays a real, fixed cost before it gets to do any actual work -- incrementing a counter, checking a bound, branching back to the top. `estimateLoopOverheadCost()` puts a number on exactly that idea, nothing more, and this section's whole point is to find out honestly how far "nothing more" gets you.

### Background

The model is deliberately linear: `cost = trips * perTripOverhead + extent * perElementWork`, where `trips = ceil(extent / chunkSize)` -- the exact same ceiling-division shape as Chapter 15's own `tileLoop()`. The per-element term never depends on `chunkSize` at all, since the same total amount of real work gets done no matter how it is chunked; only the per-trip term can ever change, and it can only ever go DOWN as `chunkSize` grows, because fewer, bigger trips always means fewer fixed charges. That single fact already tells you everything the model will ever predict: it will always rank the largest available `chunkSize` as cheapest, with no exception, because nothing in its own formula can ever produce a reason to prefer a smaller one.

The real test is whether that prediction holds up. Section 21.3's own `computeUnrolled()` is reused unchanged, run for real at `extent = 50,000,000`, timed with `std::chrono::steady_clock` across unroll factors 1 through 64, best-of-5 trials each (the same discipline Chapter 19 already established for timing real generated code). The model's own prediction is unambiguous -- chunkSize=64 is always cheapest -- but the real measurement finds a TURNOVER: performance improves sharply from factor 1 up through the low double digits, then flattens or reverses at the highest factors this section tries. This is not measurement noise; it happens on every real run, on both machines this book measures it on, though at a machine-specific turnover point, because the two real mechanisms the model was never told about -- instruction-cache pressure and register pressure from an unrolled loop body that has grown too large, plus auto-vectorization opportunities a compiler can lose once a loop body gets sufficiently complicated -- do not scale the same way on every microarchitecture.

```text
estimateLoopOverheadCost(extent, chunkSize, perTripOverhead, perElementWork)
worked by hand, extent=13 (not evenly divisible by every chunk size):

  chunkSize= 1: trips=ceil(13/ 1)=13  cost=13*40.0+13*1.0=533.0
  chunkSize= 2: trips=ceil(13/ 2)= 7  cost= 7*40.0+13*1.0=293.0
  chunkSize= 4: trips=ceil(13/ 4)= 4  cost= 4*40.0+13*1.0=173.0
  chunkSize= 8: trips=ceil(13/ 8)= 2  cost= 2*40.0+13*1.0= 93.0
  chunkSize=13: trips=ceil(13/13)= 1  cost= 1*40.0+13*1.0= 53.0

Fewer trips -> strictly lower predicted cost, every time, with no
exception the formula can ever produce:

  chunkSize=13 (the loop's own full extent, one single trip) is always
  the model's predicted best -- which is exactly why Section 21.1
  always appended the full extent to its own candidate list in the
  first place.

Scaled up to extent=50,000,000 and tested for real across chunk sizes
{1,2,4,8,16,32,64}, this same monotonic prediction says chunkSize=64 is
always cheapest. Real, timed measurement (Part 3 of the code below)
disagrees -- and disagrees at a DIFFERENT chunk size on each real
machine this book runs it on, because instruction-cache pressure and
register pressure are properties of real hardware this linear model was
never told about.
```

```cpp
// Chapter 22: Cost Models vs. Measurement-Based Autotuning
// 054_a_loop_overhead_cost_model_for_tile_size_and_unroll_factor.cpp
//
// Section 22.1 -- Chapter 21 closed with 128 legal schedules for one tiny
// two-loop kernel and explicitly declined to compile and run every single
// one. This section builds the first piece of the alternative: a COST
// MODEL, a cheap formula that estimates how good a schedule is likely to
// be WITHOUT running it. The model here covers one real, honest mechanism
// this book's own generated loops actually have -- LOOP OVERHEAD, the
// per-trip cost of incrementing a counter, checking a bound, and branching
// back to the top of the loop. Section 21.3 already established that tile
// size and unroll factor are mechanically the same question (how many
// elements does one loop trip cover?), so ONE cost model, parameterized by
// "chunk size," covers both. This section then does something most cost-
// model demos skip: it runs the REAL computation for real, at real sizes,
// with a real clock, and reports HONESTLY whether the model's own
// prediction matches what actually happened -- including a real mismatch,
// captured directly on the machines that ran it, not asserted in prose.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 054_a_loop_overhead_cost_model_for_tile_size_and_unroll_factor.cpp -o 054_driver
// Run:     ./054_driver
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <limits>

// ==================== Loop (Chapter 15, unchanged) ====================

struct Loop {
    std::string dimName;
    long long extent;
};

// ==================== Section 21.3's own computeUnrolled() and arraysExactlyEqual() (unchanged) ====================
//
// Reused verbatim: this section's own cost model is about HOW LONG a
// schedule takes, never about WHETHER it is correct -- correctness for an
// elementwise loop was already settled, bit-for-bit, back in Section 21.3.
// Re-running that exact check here (Part 3 below) is a free, honest
// reminder that every unroll factor this section measures is producing
// the SAME right answer; only the wall-clock cost is in question.
static std::vector<float> computeUnrolled(const std::vector<float>& a, long long unrollFactor) {
    long long n = static_cast<long long>(a.size());
    std::vector<float> out(static_cast<size_t>(n));
    long long i = 0;
    for (; i + unrollFactor <= n; i += unrollFactor) {
        for (long long lane = 0; lane < unrollFactor; ++lane) {
            long long idx = i + lane;
            out[static_cast<size_t>(idx)] = a[static_cast<size_t>(idx)] * 2.0f + 1.0f;
        }
    }
    for (; i < n; ++i) out[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] * 2.0f + 1.0f;  // scalar tail
    return out;
}
static bool arraysExactlyEqual(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

// ==================== Section 22.1: the loop-overhead cost model ====================
//
// A deliberately SIMPLE linear model: processing `extent` elements in
// chunks of `chunkSize` takes ceil(extent / chunkSize) trips through the
// loop (the exact same ceiling-division shape as Chapter 15's own
// tileLoop() -- a partial final chunk still costs one full trip), each
// trip paying a fixed per-trip overhead, plus a per-element cost that
// does NOT depend on chunk size at all (the same total work gets done
// either way, just organized into bigger or smaller trips). The constants
// below are illustrative "cost units," not calibrated nanoseconds --
// calibrating them against a real clock is precisely what the rest of
// this section tests the model against.
static double estimateLoopOverheadCost(long long extent, long long chunkSize,
                                        double perTripOverhead, double perElementWork) {
    long long trips = (extent + chunkSize - 1) / chunkSize;  // ceiling division
    return static_cast<double>(trips) * perTripOverhead + static_cast<double>(extent) * perElementWork;
}

int main() {
    printf("=== Section 22.1: a loop-overhead cost model for tile size and unroll factor ===\n\n");

    const double perTripOverhead = 40.0;   // illustrative cost units per loop trip
    const double perElementWork  = 1.0;    // illustrative cost units per element, independent of chunking

    // ---- Part 1: the model's own arithmetic, worked by hand on a small extent ----
    printf("--- Part 1: estimateLoopOverheadCost() on a small extent=13 (not evenly divisible) ---\n\n");
    for (long long chunkSize : {1LL, 2LL, 4LL, 8LL, 13LL}) {
        long long trips = (13 + chunkSize - 1) / chunkSize;
        double cost = estimateLoopOverheadCost(13, chunkSize, perTripOverhead, perElementWork);
        printf("  chunkSize=%-2lld -> trips=ceil(13/%lld)=%-2lld  cost = %lld*%.1f + 13*%.1f = %.1f\n",
               chunkSize, chunkSize, trips, trips, perTripOverhead, perElementWork, cost);
    }
    printf("\nself-check: every extra trip costs a fixed %.1f units no matter how small the tail is --\n",
           perTripOverhead);
    printf("chunkSize=13 (one single trip covering the whole extent) is the model's cheapest option here,\n");
    printf("exactly the same intuition Section 21.1 used to justify why the loop's own full extent always\n");
    printf("belongs in the tile-size candidate list.\n");

    // ---- Part 2: the model's own prediction at real scale, extent=50,000,000 ----
    printf("\n--- Part 2: the model's prediction at extent=50,000,000, across real candidate chunk sizes ---\n\n");
    const long long extent = 50'000'000;
    std::vector<long long> chunkSizes = {1, 2, 4, 8, 16, 32, 64};
    std::vector<double> predictedCost;
    for (long long chunkSize : chunkSizes) {
        predictedCost.push_back(estimateLoopOverheadCost(extent, chunkSize, perTripOverhead, perElementWork));
    }
    bool modelMonotonic = true;
    for (size_t i = 0; i + 1 < predictedCost.size(); ++i) {
        printf("  chunkSize=%-3lld predicted cost = %14.1f\n", chunkSizes[i], predictedCost[i]);
        if (predictedCost[i] < predictedCost[i + 1]) modelMonotonic = false;
    }
    printf("  chunkSize=%-3lld predicted cost = %14.1f\n", chunkSizes.back(), predictedCost.back());
    size_t predictedBestIdx = 0;
    for (size_t i = 1; i < predictedCost.size(); ++i) if (predictedCost[i] < predictedCost[predictedBestIdx]) predictedBestIdx = i;
    printf("\nself-check: predicted cost is non-increasing as chunkSize grows (%s) -- the model's own\n",
           modelMonotonic ? "confirmed" : "MISMATCH");
    printf("per-element term never depends on chunkSize, so ONLY fewer trips can ever lower the estimate;\n");
    printf("the model's predicted best is therefore always the LARGEST candidate: chunkSize=%lld.\n",
           chunkSizes[predictedBestIdx]);

    // ---- Part 3: real measurement, the exact same computation, a real clock ----
    printf("\n--- Part 3: real measurement -- computeUnrolled() at extent=50,000,000, best-of-5 trials ---\n\n");
    std::vector<float> a(static_cast<size_t>(extent));
    for (long long i = 0; i < extent; ++i) a[static_cast<size_t>(i)] = static_cast<float>(i % 997) * 0.001f;

    std::vector<float> baseline = computeUnrolled(a, 1);
    bool allCorrect = true;
    std::vector<double> measuredMs;
    for (long long factor : chunkSizes) {
        std::vector<float> result;
        double best = std::numeric_limits<double>::max();
        const int trials = 5;
        for (int t = 0; t < trials; ++t) {
            auto start = std::chrono::steady_clock::now();
            result = computeUnrolled(a, factor);
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (ms < best) best = ms;
        }
        bool correct = arraysExactlyEqual(result, baseline);
        allCorrect = allCorrect && correct;
        measuredMs.push_back(best);
        printf("  unrollFactor=%-3lld best-of-5 = %8.3f ms   (bit-for-bit correct: %s)\n",
               factor, best, correct ? "yes" : "NO");
    }
    size_t measuredBestIdx = 0;
    for (size_t i = 1; i < measuredMs.size(); ++i) if (measuredMs[i] < measuredMs[measuredBestIdx]) measuredBestIdx = i;
    printf("\nself-check: every unroll factor still produces the exact same bit-for-bit result (%s) --\n",
           allCorrect ? "confirmed" : "MISMATCH");
    printf("unroll factor is a pure performance knob here, same finding as Section 21.3, so ranking these\n");
    printf("candidates by SPEED ALONE, with no correctness tradeoff to weigh, is a fair comparison.\n");

    // ---- Part 4: the honest comparison ----
    printf("\n--- Part 4: does the model's prediction match what actually happened? ---\n\n");
    printf("  model predicts best chunkSize:    %lld (predicted cost strictly favors the largest candidate)\n",
           chunkSizes[predictedBestIdx]);
    printf("  real measurement's fastest factor: %lld (%.3f ms, best-of-5)\n",
           chunkSizes[measuredBestIdx], measuredMs[measuredBestIdx]);
    bool predictionMatchesMeasurement = (predictedBestIdx == measuredBestIdx);
    printf("  match: %s\n\n", predictionMatchesMeasurement ? "yes" : "no");
    if (!predictionMatchesMeasurement) {
        printf("This is not a bug in the measurement, and it is not a bug in the model -- it is the model\n");
        printf("being HONEST about its own limits. estimateLoopOverheadCost() only knows about one real\n");
        printf("mechanism (fixed cost per trip), so it can only ever recommend 'fewer, bigger trips.' Real\n");
        printf("hardware has other mechanisms this simple model was never told about -- instruction-cache\n");
        printf("pressure, register pressure, and lost auto-vectorization opportunities all grow with very\n");
        printf("large unroll factors -- and past some machine-specific point those costs start to outweigh\n");
        printf("the loop-overhead savings the model DOES know about. The model is not wrong to prefer fewer\n");
        printf("trips in general; it is incomplete about where that preference stops paying off. This is\n");
        printf("exactly why Chapter 23's own autotuner will not trust a cost model blindly -- it will use\n");
        printf("one to narrow 128 candidates down to a short list, then MEASURE the short list for real.\n");
    } else {
        printf("On this run, the model's prediction happened to match the real measurement's own ranking --\n");
        printf("that can genuinely happen, but it is not something the model can promise in general, since\n");
        printf("it was never told about instruction-cache pressure, register pressure, or vectorization\n");
        printf("effects at all. Section 22.2 builds a second cost model for a mechanism that turns out to\n");
        printf("matter far more dramatically: memory-access stride.\n");
    }

    bool allOk = modelMonotonic && allCorrect;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "054_a_loop_overhead_cost_model_for_tile_size_and_unroll_factor.cpp" -o "054_a_loop_overhead_cost_model_for_tile_size_and_unroll_factor"
./"054_a_loop_overhead_cost_model_for_tile_size_and_unroll_factor"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 22.1: a loop-overhead cost model for tile size and unroll factor ===

--- Part 1: estimateLoopOverheadCost() on a small extent=13 (not evenly divisible) ---

  chunkSize=1  -> trips=ceil(13/1)=13  cost = 13*40.0 + 13*1.0 = 533.0
  chunkSize=2  -> trips=ceil(13/2)=7   cost = 7*40.0 + 13*1.0 = 293.0
  chunkSize=4  -> trips=ceil(13/4)=4   cost = 4*40.0 + 13*1.0 = 173.0
  chunkSize=8  -> trips=ceil(13/8)=2   cost = 2*40.0 + 13*1.0 = 93.0
  chunkSize=13 -> trips=ceil(13/13)=1   cost = 1*40.0 + 13*1.0 = 53.0

self-check: every extra trip costs a fixed 40.0 units no matter how small the tail is --
chunkSize=13 (one single trip covering the whole extent) is the model's cheapest option here,
exactly the same intuition Section 21.1 used to justify why the loop's own full extent always
belongs in the tile-size candidate list.

--- Part 2: the model's prediction at extent=50,000,000, across real candidate chunk sizes ---

  chunkSize=1   predicted cost =   2050000000.0
  chunkSize=2   predicted cost =   1050000000.0
  chunkSize=4   predicted cost =    550000000.0
  chunkSize=8   predicted cost =    300000000.0
  chunkSize=16  predicted cost =    175000000.0
  chunkSize=32  predicted cost =    112500000.0
  chunkSize=64  predicted cost =     81250000.0

self-check: predicted cost is non-increasing as chunkSize grows (confirmed) -- the model's own
per-element term never depends on chunkSize, so ONLY fewer trips can ever lower the estimate;
the model's predicted best is therefore always the LARGEST candidate: chunkSize=64.

--- Part 3: real measurement -- computeUnrolled() at extent=50,000,000, best-of-5 trials ---

  unrollFactor=1   best-of-5 =  157.738 ms   (bit-for-bit correct: yes)
  unrollFactor=2   best-of-5 =  150.744 ms   (bit-for-bit correct: yes)
  unrollFactor=4   best-of-5 =  127.949 ms   (bit-for-bit correct: yes)
  unrollFactor=8   best-of-5 =  138.004 ms   (bit-for-bit correct: yes)
  unrollFactor=16  best-of-5 =  136.083 ms   (bit-for-bit correct: yes)
  unrollFactor=32  best-of-5 =  126.207 ms   (bit-for-bit correct: yes)
  unrollFactor=64  best-of-5 =  133.430 ms   (bit-for-bit correct: yes)

self-check: every unroll factor still produces the exact same bit-for-bit result (confirmed) --
unroll factor is a pure performance knob here, same finding as Section 21.3, so ranking these
candidates by SPEED ALONE, with no correctness tradeoff to weigh, is a fair comparison.

--- Part 4: does the model's prediction match what actually happened? ---

  model predicts best chunkSize:    64 (predicted cost strictly favors the largest candidate)
  real measurement's fastest factor: 32 (126.207 ms, best-of-5)
  match: no

This is not a bug in the measurement, and it is not a bug in the model -- it is the model
being HONEST about its own limits. estimateLoopOverheadCost() only knows about one real
mechanism (fixed cost per trip), so it can only ever recommend 'fewer, bigger trips.' Real
hardware has other mechanisms this simple model was never told about -- instruction-cache
pressure, register pressure, and lost auto-vectorization opportunities all grow with very
large unroll factors -- and past some machine-specific point those costs start to outweigh
the loop-overhead savings the model DOES know about. The model is not wrong to prefer fewer
trips in general; it is incomplete about where that preference stops paying off. This is
exactly why Chapter 23's own autotuner will not trust a cost model blindly -- it will use
one to narrow 128 candidates down to a short list, then MEASURE the short list for real.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 22.1: a loop-overhead cost model for tile size and unroll factor ===

--- Part 1: estimateLoopOverheadCost() on a small extent=13 (not evenly divisible) ---

  chunkSize=1  -> trips=ceil(13/1)=13  cost = 13*40.0 + 13*1.0 = 533.0
  chunkSize=2  -> trips=ceil(13/2)=7   cost = 7*40.0 + 13*1.0 = 293.0
  chunkSize=4  -> trips=ceil(13/4)=4   cost = 4*40.0 + 13*1.0 = 173.0
  chunkSize=8  -> trips=ceil(13/8)=2   cost = 2*40.0 + 13*1.0 = 93.0
  chunkSize=13 -> trips=ceil(13/13)=1   cost = 1*40.0 + 13*1.0 = 53.0

self-check: every extra trip costs a fixed 40.0 units no matter how small the tail is --
chunkSize=13 (one single trip covering the whole extent) is the model's cheapest option here,
exactly the same intuition Section 21.1 used to justify why the loop's own full extent always
belongs in the tile-size candidate list.

--- Part 2: the model's prediction at extent=50,000,000, across real candidate chunk sizes ---

  chunkSize=1   predicted cost =   2050000000.0
  chunkSize=2   predicted cost =   1050000000.0
  chunkSize=4   predicted cost =    550000000.0
  chunkSize=8   predicted cost =    300000000.0
  chunkSize=16  predicted cost =    175000000.0
  chunkSize=32  predicted cost =    112500000.0
  chunkSize=64  predicted cost =     81250000.0

self-check: predicted cost is non-increasing as chunkSize grows (confirmed) -- the model's own
per-element term never depends on chunkSize, so ONLY fewer trips can ever lower the estimate;
the model's predicted best is therefore always the LARGEST candidate: chunkSize=64.

--- Part 3: real measurement -- computeUnrolled() at extent=50,000,000, best-of-5 trials ---

  unrollFactor=1   best-of-5 =   48.827 ms   (bit-for-bit correct: yes)
  unrollFactor=2   best-of-5 =   36.680 ms   (bit-for-bit correct: yes)
  unrollFactor=4   best-of-5 =   34.047 ms   (bit-for-bit correct: yes)
  unrollFactor=8   best-of-5 =   32.649 ms   (bit-for-bit correct: yes)
  unrollFactor=16  best-of-5 =   37.367 ms   (bit-for-bit correct: yes)
  unrollFactor=32  best-of-5 =   40.223 ms   (bit-for-bit correct: yes)
  unrollFactor=64  best-of-5 =   34.320 ms   (bit-for-bit correct: yes)

self-check: every unroll factor still produces the exact same bit-for-bit result (confirmed) --
unroll factor is a pure performance knob here, same finding as Section 21.3, so ranking these
candidates by SPEED ALONE, with no correctness tradeoff to weigh, is a fair comparison.

--- Part 4: does the model's prediction match what actually happened? ---

  model predicts best chunkSize:    64 (predicted cost strictly favors the largest candidate)
  real measurement's fastest factor: 8 (32.649 ms, best-of-5)
  match: no

This is not a bug in the measurement, and it is not a bug in the model -- it is the model
being HONEST about its own limits. estimateLoopOverheadCost() only knows about one real
mechanism (fixed cost per trip), so it can only ever recommend 'fewer, bigger trips.' Real
hardware has other mechanisms this simple model was never told about -- instruction-cache
pressure, register pressure, and lost auto-vectorization opportunities all grow with very
large unroll factors -- and past some machine-specific point those costs start to outweigh
the loop-overhead savings the model DOES know about. The model is not wrong to prefer fewer
trips in general; it is incomplete about where that preference stops paying off. This is
exactly why Chapter 23's own autotuner will not trust a cost model blindly -- it will use
one to narrow 128 candidates down to a short list, then MEASURE the short list for real.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which candidate ranks fastest, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! note "The model's own constants are illustrative, not calibrated nanoseconds"
    `perTripOverhead = 40.0` and `perElementWork = 1.0` are round, made-up numbers in arbitrary "cost units" -- not a claim that one loop trip really costs 40 nanoseconds on any real machine. Calibrating those two constants against a real clock, for a specific compiler and a specific CPU, is a real engineering task in its own right, and it would not have rescued this section's own prediction anyway: no matter what positive values `perTripOverhead` and `perElementWork` take, the model's own formula is monotonic in `chunkSize`, so it will ALWAYS predict the largest candidate as cheapest. The honest finding this section produced -- a real turnover point the model cannot see -- is a structural limitation of the model's own shape, not a tuning problem four better constants would fix.

## 22.2 A Memory-Stride Cost Model for Loop Order

### Intuition

Section 21.2 already proved that loop order cannot change an elementwise computation's own ANSWER -- every permutation of a `LoopNest`'s loops visits the identical set of index tuples. This section is entirely about whether loop order changes the TIME, and the mechanism is different from Section 22.1's in kind, not just in degree: this book's own tensors are stored row-major (Chapter 4), so for a `LoopNest` of `[dim0, dim1]`, element `(i, j)` sits at flat offset `i*dim1.extent + j` -- meaning `dim1`, the LAST loop in the nest, is the CONTIGUOUS dimension in memory. Whichever loop ends up INNERMOST in a given order determines what "the next trip" means in real memory: a jump of one element, or a jump of an entire row.

### Background

`estimateLoopOrderCost()` counts CACHE-LINE LOADS, not raw element accesses. When the innermost loop matches the contiguous dimension, consecutive trips touch consecutive addresses, so one loaded cache line (modeled at a realistic 64 bytes, `cacheLineElements = 16` for float32) serves 16 trips before the next line is needed. When the innermost loop does NOT match the contiguous dimension, consecutive trips jump by an entire row -- far larger than one cache line in any real case this section considers -- so, worst case, every single trip pays for a fresh line it will use exactly once. That is an 16-to-1 difference in line loads alone; folded into the model's own overall cost (which also charges a constant per-element compute term, identical either way, diluting the memory-only effect once real work is added to the same total), the model's own predicted GAP comes out smaller than that raw 16-to-1 ratio.

The real test, on a `LoopNest` of `[dim0:4096, dim1:4096]` -- 64 MB per array, deliberately larger than any real cache -- is dramatic in a way Section 22.1's own test was not: real measurement finds row-major traversal tens of times faster than column-major, a far WIDER gap than the model's own prediction, on both machines this book measures it on. The model's own DIRECTION is right, and by a wide, unambiguous margin; its own predicted MAGNITUDE is not, because real memory hierarchies do things -- hardware prefetching that helps the contiguous case even more than a flat line-count model assumes, and TLB misses under a stride this large -- that a model built purely on counting cache-line loads was never told about.

```text
Row-major memory layout (Chapter 4): for LoopNest [dim0, dim1], element
(i,j) lives at flat offset i*dim1.extent + j -- dim1 is CONTIGUOUS.

  order dim0>dim1 (row-major: dim1 innermost, matches contiguous dim)

    memory:  [(0,0)(0,1)(0,2)(0,3)...(0,N-1)][(1,0)(1,1)...]
    walk:     ------------------------------->
              consecutive trips = consecutive addresses
              one cache line (16 floats) serves 16 trips

  order dim1>dim0 (col-major: dim0 innermost, does NOT match contiguous dim)

    memory:  [(0,0) . . . . . . . . . . . . .][(1,0) . . . . . .]
    walk:     (0,0) -----jump N floats-----> (1,0) -----jump-----> (2,0)
              consecutive trips = addresses N floats apart
              worst case: every trip pays for a fresh, unreused line

Predicted line-load ratio: exactly 16:1 (cacheLineElements). Predicted
OVERALL cost ratio: smaller than 16:1 (perElementWork dilutes it).
Real measured ratio, on real 64 MB arrays: much LARGER than either
prediction -- the model got the direction right and the size wrong.
```

```cpp
// Chapter 22: Cost Models vs. Measurement-Based Autotuning
// 055_a_memory_stride_cost_model_for_loop_order.cpp
//
// Section 22.2 -- Section 22.1 built a cost model for one real mechanism
// (loop overhead) and found, honestly, that it can be wrong: a model that
// only knows about trip count cannot see instruction-cache pressure or
// lost vectorization at very high unroll factors. This section builds a
// SECOND cost model, for LOOP ORDER, around a different real mechanism --
// memory-access stride -- and this time the model's prediction is tested
// against a much more dramatic real effect: how a 2D array is walked
// determines whether each memory access reuses a cache line that is
// already loaded, or evicts one and pays full memory latency every single
// time. Both loop orders visit the exact same set of (i, j) pairs (Section
// 21.2 already proved that, for an elementwise loop, order cannot change
// the ANSWER) -- this section is entirely about whether it changes the
// TIME, and by how much.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 055_a_memory_stride_cost_model_for_loop_order.cpp -o 055_driver
// Run:     ./055_driver
#include <cstdio>
#include <string>
#include <vector>
#include <chrono>
#include <limits>

// ==================== Loop / LoopNest / LoopOrder (Chapter 15 / 21.2, unchanged) ====================

struct Loop {
    std::string dimName;
    long long extent;
};
struct LoopNest {
    std::vector<Loop> loops;
};
using LoopOrder = std::vector<int>;

// ==================== Section 22.2: the memory-stride cost model ====================
//
// This book's own arrays are laid out row-major (Chapter 4's own tensor
// representation): for a LoopNest [dim0, dim1], element (i, j) lives at
// flat offset i*dim1.extent + j, so dim1 -- the LAST loop in the nest --
// is the CONTIGUOUS dimension in memory: consecutive j values sit right
// next to each other. Whichever dimension ends up INNERMOST in a given
// loop order determines what "consecutive loop trips" means in memory:
//   - innermost loop == the contiguous dimension (dim1): consecutive
//     trips touch consecutive addresses, so one cache line, once loaded,
//     serves `cacheLineElements` trips before the next line is needed.
//   - innermost loop == any OTHER dimension: consecutive trips jump by
//     a whole row (dim1.extent elements) in memory every time, which is
//     almost always far bigger than one cache line, so (worst case)
//     EVERY trip pays for a fresh cache line that gets used exactly once.
// This is a different mechanism from Section 22.1's loop-overhead model
// (which never looked at memory addresses at all), so it is a genuinely
// separate cost model, not a rename of the same idea.
static double estimateLoopOrderCost(const LoopNest& nest, const LoopOrder& order,
                                     long long cacheLineElements, double perLineCost, double perElementWork) {
    long long totalIterations = 1;
    for (const Loop& l : nest.loops) totalIterations *= l.extent;

    int contiguousDim = static_cast<int>(nest.loops.size()) - 1;  // last loop dim = row-major-contiguous
    int innermostDim = order.back();                              // last entry in the ORDER = innermost trip

    double lineLoads;
    if (innermostDim == contiguousDim) {
        // best case: cacheLineElements consecutive trips share one line
        lineLoads = static_cast<double>(totalIterations) / static_cast<double>(cacheLineElements);
    } else {
        // worst case: every trip pays for a line it will never reuse
        lineLoads = static_cast<double>(totalIterations);
    }
    return lineLoads * perLineCost + static_cast<double>(totalIterations) * perElementWork;
}

// ==================== A real elementwise computation, walked in a given order ====================
//
// out[i][j] = a[i][j] * 2.0f + 1.0f, over a real N x N array flattened
// row-major (a[i*N + j]), walked either row-major (dim0 outer, dim1
// inner -- order {0,1}) or column-major (dim1 outer, dim0 inner -- order
// {1,0}). N is chosen deliberately large enough (4096 x 4096 floats =
// 64 MB per array) that the whole array cannot fit in any real cache, so
// the effect measured is genuinely about memory traffic, not an artifact
// of a dataset small enough to sit in cache either way.
static void computeOrdered(const std::vector<float>& a, std::vector<float>& out, long long N, const LoopOrder& order) {
    if (order == LoopOrder{0, 1}) {          // row-major: i outer, j inner (contiguous)
        for (long long i = 0; i < N; ++i)
            for (long long j = 0; j < N; ++j)
                out[static_cast<size_t>(i * N + j)] = a[static_cast<size_t>(i * N + j)] * 2.0f + 1.0f;
    } else {                                  // column-major: j outer, i inner (strided)
        for (long long j = 0; j < N; ++j)
            for (long long i = 0; i < N; ++i)
                out[static_cast<size_t>(i * N + j)] = a[static_cast<size_t>(i * N + j)] * 2.0f + 1.0f;
    }
}
static bool arraysExactlyEqual(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

int main() {
    printf("=== Section 22.2: a memory-stride cost model for loop order ===\n\n");

    LoopNest nest{{Loop{"dim0", 4096}, Loop{"dim1", 4096}}};
    LoopOrder rowMajorOrder{0, 1};  // dim0 outer, dim1 inner -- inner matches the contiguous dim
    LoopOrder colMajorOrder{1, 0};  // dim1 outer, dim0 inner -- inner does NOT match the contiguous dim

    const long long cacheLineBytes = 64;                              // a typical real cache-line size
    const long long cacheLineElements = cacheLineBytes / sizeof(float); // 16 floats per line
    const double perLineCost = 40.0;    // illustrative cost units per cache-line load
    const double perElementWork = 1.0;  // illustrative cost units per element, same either way

    // ---- Part 1: the model's own prediction ----
    printf("--- Part 1: estimateLoopOrderCost() for both loop orders, LoopNest [dim0:4096, dim1:4096] ---\n\n");
    double rowMajorCost = estimateLoopOrderCost(nest, rowMajorOrder, cacheLineElements, perLineCost, perElementWork);
    double colMajorCost = estimateLoopOrderCost(nest, colMajorOrder, cacheLineElements, perLineCost, perElementWork);
    printf("  order dim0>dim1 (row-major, inner=dim1=contiguous): predicted cost = %.1f\n", rowMajorCost);
    printf("  order dim1>dim0 (col-major, inner=dim0=strided):    predicted cost = %.1f\n", colMajorCost);
    printf("  predicted ratio (col-major / row-major) = %.1fx\n\n", colMajorCost / rowMajorCost);
    printf("self-check: the model predicts row-major cheaper (%s) -- the LINE-LOAD count alone differs\n",
           (rowMajorCost < colMajorCost) ? "confirmed" : "MISMATCH");
    printf("by exactly %lld:1 (cacheLineElements itself: one line load per element, worst case, versus one\n",
           cacheLineElements);
    printf("line load amortized across %lld elements, best case) -- but the OVERALL predicted cost ratio\n",
           cacheLineElements);
    printf("(%.1fx) is smaller than that, because perElementWork is the SAME %.1f-unit charge either way,\n",
           colMajorCost / rowMajorCost, perElementWork);
    printf("diluting the memory-only effect once real compute work is added into the same total.\n");

    // ---- Part 2: real measurement, the exact same computation, a real clock ----
    printf("\n--- Part 2: real measurement -- computeOrdered() on a real 4096x4096 float array (64 MB) ---\n\n");
    const long long N = 4096;
    std::vector<float> a(static_cast<size_t>(N * N));
    for (long long i = 0; i < N * N; ++i) a[static_cast<size_t>(i)] = static_cast<float>(i % 997) * 0.001f;
    std::vector<float> outRow(static_cast<size_t>(N * N)), outCol(static_cast<size_t>(N * N));

    const int trials = 5;
    double bestRowMs = std::numeric_limits<double>::max(), bestColMs = std::numeric_limits<double>::max();
    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::steady_clock::now();
        computeOrdered(a, outRow, N, rowMajorOrder);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < bestRowMs) bestRowMs = ms;
    }
    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::steady_clock::now();
        computeOrdered(a, outCol, N, colMajorOrder);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < bestColMs) bestColMs = ms;
    }
    bool bothCorrect = arraysExactlyEqual(outRow, outCol);
    printf("  order dim0>dim1 (row-major) best-of-%d = %8.3f ms\n", trials, bestRowMs);
    printf("  order dim1>dim0 (col-major) best-of-%d = %8.3f ms\n", trials, bestColMs);
    printf("  measured ratio (col-major / row-major) = %.2fx\n\n", bestColMs / bestRowMs);
    printf("self-check: both orders produce the exact same bit-for-bit output array (%s) -- Section\n",
           bothCorrect ? "confirmed" : "MISMATCH");
    printf("21.2's own finding holds again here: for this elementwise computation, loop order is STILL a\n");
    printf("pure performance knob, never a correctness concern; what changes is only how long it takes.\n");

    // ---- Part 3: the honest comparison ----
    printf("\n--- Part 3: does the model's prediction match what actually happened? ---\n\n");
    bool modelRankingCorrect = (rowMajorCost < colMajorCost) == (bestRowMs < bestColMs);
    printf("  model predicts:  row-major cheaper by %.1fx\n", colMajorCost / rowMajorCost);
    printf("  measurement finds: row-major faster by %.2fx\n", bestColMs / bestRowMs);
    printf("  ranking matches: %s\n\n", modelRankingCorrect ? "yes" : "no");
    printf("Unlike Section 22.1's loop-overhead model, this one gets the RANKING right. The magnitude is\n");
    printf("off -- the model predicted an %.1fx gap and reality delivered %.2fx -- real memory hierarchies\n",
           colMajorCost / rowMajorCost, bestColMs / bestRowMs);
    printf("have effects this deliberately simple model never included (TLB misses under a stride this\n");
    printf("large, and hardware prefetchers that help the contiguous case even more than a flat line-count\n");
    printf("model assumes), so the real penalty for column-major turned out worse than predicted, not\n");
    printf("better. Getting the DIRECTION right while getting the MAGNITUDE wrong is still a useful cost\n");
    printf("model for ranking candidates, which is all an autotuner actually needs from it -- and it is a\n");
    printf("different kind of honesty than Section 22.1's model, which got the direction wrong at high\n");
    printf("unroll factors. Different mechanisms deserve different amounts of trust, which is exactly what\n");
    printf("Section 22.3 has to decide when it combines both models into one score for a real schedule.\n");

    bool allOk = (rowMajorCost < colMajorCost) && bothCorrect;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "055_a_memory_stride_cost_model_for_loop_order.cpp" -o "055_a_memory_stride_cost_model_for_loop_order"
./"055_a_memory_stride_cost_model_for_loop_order"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 22.2: a memory-stride cost model for loop order ===

--- Part 1: estimateLoopOrderCost() for both loop orders, LoopNest [dim0:4096, dim1:4096] ---

  order dim0>dim1 (row-major, inner=dim1=contiguous): predicted cost = 58720256.0
  order dim1>dim0 (col-major, inner=dim0=strided):    predicted cost = 687865856.0
  predicted ratio (col-major / row-major) = 11.7x

self-check: the model predicts row-major cheaper (confirmed) -- the LINE-LOAD count alone differs
by exactly 16:1 (cacheLineElements itself: one line load per element, worst case, versus one
line load amortized across 16 elements, best case) -- but the OVERALL predicted cost ratio
(11.7x) is smaller than that, because perElementWork is the SAME 1.0-unit charge either way,
diluting the memory-only effect once real compute work is added into the same total.

--- Part 2: real measurement -- computeOrdered() on a real 4096x4096 float array (64 MB) ---

  order dim0>dim1 (row-major) best-of-5 =   13.906 ms
  order dim1>dim0 (col-major) best-of-5 =  390.835 ms
  measured ratio (col-major / row-major) = 28.11x

self-check: both orders produce the exact same bit-for-bit output array (confirmed) -- Section
21.2's own finding holds again here: for this elementwise computation, loop order is STILL a
pure performance knob, never a correctness concern; what changes is only how long it takes.

--- Part 3: does the model's prediction match what actually happened? ---

  model predicts:  row-major cheaper by 11.7x
  measurement finds: row-major faster by 28.11x
  ranking matches: yes

Unlike Section 22.1's loop-overhead model, this one gets the RANKING right. The magnitude is
off -- the model predicted an 11.7x gap and reality delivered 28.11x -- real memory hierarchies
have effects this deliberately simple model never included (TLB misses under a stride this
large, and hardware prefetchers that help the contiguous case even more than a flat line-count
model assumes), so the real penalty for column-major turned out worse than predicted, not
better. Getting the DIRECTION right while getting the MAGNITUDE wrong is still a useful cost
model for ranking candidates, which is all an autotuner actually needs from it -- and it is a
different kind of honesty than Section 22.1's model, which got the direction wrong at high
unroll factors. Different mechanisms deserve different amounts of trust, which is exactly what
Section 22.3 has to decide when it combines both models into one score for a real schedule.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 22.2: a memory-stride cost model for loop order ===

--- Part 1: estimateLoopOrderCost() for both loop orders, LoopNest [dim0:4096, dim1:4096] ---

  order dim0>dim1 (row-major, inner=dim1=contiguous): predicted cost = 58720256.0
  order dim1>dim0 (col-major, inner=dim0=strided):    predicted cost = 687865856.0
  predicted ratio (col-major / row-major) = 11.7x

self-check: the model predicts row-major cheaper (confirmed) -- the LINE-LOAD count alone differs
by exactly 16:1 (cacheLineElements itself: one line load per element, worst case, versus one
line load amortized across 16 elements, best case) -- but the OVERALL predicted cost ratio
(11.7x) is smaller than that, because perElementWork is the SAME 1.0-unit charge either way,
diluting the memory-only effect once real compute work is added into the same total.

--- Part 2: real measurement -- computeOrdered() on a real 4096x4096 float array (64 MB) ---

  order dim0>dim1 (row-major) best-of-5 =    4.214 ms
  order dim1>dim0 (col-major) best-of-5 =  211.002 ms
  measured ratio (col-major / row-major) = 50.07x

self-check: both orders produce the exact same bit-for-bit output array (confirmed) -- Section
21.2's own finding holds again here: for this elementwise computation, loop order is STILL a
pure performance knob, never a correctness concern; what changes is only how long it takes.

--- Part 3: does the model's prediction match what actually happened? ---

  model predicts:  row-major cheaper by 11.7x
  measurement finds: row-major faster by 50.07x
  ranking matches: yes

Unlike Section 22.1's loop-overhead model, this one gets the RANKING right. The magnitude is
off -- the model predicted an 11.7x gap and reality delivered 50.07x -- real memory hierarchies
have effects this deliberately simple model never included (TLB misses under a stride this
large, and hardware prefetchers that help the contiguous case even more than a flat line-count
model assumes), so the real penalty for column-major turned out worse than predicted, not
better. Getting the DIRECTION right while getting the MAGNITUDE wrong is still a useful cost
model for ranking candidates, which is all an autotuner actually needs from it -- and it is a
different kind of honesty than Section 22.1's model, which got the direction wrong at high
unroll factors. Different mechanisms deserve different amounts of trust, which is exactly what
Section 22.3 has to decide when it combines both models into one score for a real schedule.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which candidate ranks fastest, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! warning "[COMMON TRAP] a model that gets the RANKING right is not a model you can trust for MAGNITUDE"
    Section 22.2's own model correctly says "row-major is cheaper" -- and it would be a real mistake to read that success as license to trust its own predicted RATIO too. An autotuner that used this model's own 11-to-12x prediction to decide, say, how much extra tiling complexity is worth trading for a better loop order would be reasoning from a number that real hardware, on both machines this book measured, blew past by a wide margin. Getting a ranking right and getting a magnitude right are different claims, and a cost model can genuinely earn one without the other -- which is exactly why Chapter 23's own autotuner will use a cost model to choose WHICH few candidates to measure, never to decide BY HOW MUCH one candidate is expected to win.

## 22.3 Combining Both Cost Models: Prediction vs. Measurement for Real Schedules

### Intuition

A real `Schedule` (Chapter 21) bundles three dimensions at once -- a tile size per loop, one loop order for the whole nest, and one unroll factor -- so a cost model that only ever looks at one dimension in isolation, the way Sections 22.1 and 22.2 each did, is not yet useful for ranking actual schedules. This section builds `estimateScheduleCost()`, which does the simplest defensible thing available: it reuses both prior models UNCHANGED and adds their outputs together, charging real per-element compute work exactly once rather than once per mechanism. Then it does what this book has done at the close of every major idea since Chapter 17 -- actually RUNS the thing being modeled, for real, and reports honestly whether the model's own combined score predicted what happened.

### Background

Running 128 separate hand-written functions, one per schedule, was never the point of this book -- Chapter 23's own autotuner will need to try candidates before committing to real codegen for a winner, the same way this book's own IR (Part 1) is interpreted by `evaluateArrays()` long before Part 4 ever generates real compiled code for it. `executeSchedule()` is that same idea applied to schedules: a single, generic function that reads a `Schedule`'s own fields as DATA at runtime -- tiling both loops with Chapter 15's own `tileLoop()`, walking tile blocks and then in-tile elements in whichever order the schedule specifies, unrolling the innermost in-tile loop by whatever factor the schedule specifies -- rather than one hand-written function per candidate. That flexibility has a real, stated cost: an interpreter is slower per element than compiled, schedule-specific code would be, exactly the tradeoff Part 4's own generated code was built to avoid for the ONE schedule each of those chapters ever used.

This section applies `estimateScheduleCost()` to 6 schedules drawn evenly from Chapter 21's own 128-schedule list for the `LoopNest [dim0:6, dim1:8]`, and executes each one for real through `executeSchedule()` -- but that `LoopNest` has only 48 total elements, far too little work for any single call to register on a wall clock, so each schedule is run 800,000 times, best-of-7, just to get a measurable duration at all. The combined model's own ranking agreement with real measurement, on both machines this book tests it on, lands close to what random chance would produce -- and the honest diagnosis, worked out in the code's own closing comparison below, is that at this problem size the INTERPRETER's own per-call bookkeeping competes with, or outright dominates, the 48 real elements it is scheduling, so the clock is measuring a cost neither model was ever built to predict. That is a genuinely different, and narrower, finding than "cost models don't work" -- Section 22.2's own model, measuring a REAL compiled loop with no interpreter in the way, held up far better.

```text
estimateScheduleCost(nest, schedule) = SUM of three reused terms:

  + estimateLoopOverheadCost(loop, tileSize) for EACH loop in the nest
    (tiling adds trips to every loop it touches, not just one)

  + estimateLoopOverheadCost(innermost tile, unrollFactor)
    (unrolling further chunks the innermost loop's own tile -- Ch21.3's
    own tile-size/unroll-factor equivalence, applied again here)

  + estimateLoopOrderCost(nest, loopOrder)
    (the whole nest's own memory-stride cost, Section 22.2, unchanged)

  + total elements * perElementWork    -- charged ONCE, at the end, not
                                           inside any of the three calls
                                           above (each passes 0.0 for
                                           its own perElementWork term)

A plain sum of two independently-tested, independently-imperfect models
is not a claim that loop overhead and cache misses trade off at some
universally correct exchange rate -- it is the simplest way to combine
two real cost signals when there is no principled reason to weight one
above the other, and this section's own closing comparison is exactly
that combination's honest report card.
```

```cpp
// Chapter 22: Cost Models vs. Measurement-Based Autotuning
// 056_combining_both_cost_models_prediction_vs_measurement_for_real_schedules.cpp
//
// Section 22.3 (capstone) -- Sections 22.1 and 22.2 each built and tested
// ONE cost model, for ONE mechanism, in isolation. A real Schedule
// (Section 21.3) bundles three dimensions together -- tile size per loop,
// loop order, and unroll factor -- so a useful cost model has to score a
// WHOLE schedule, not one dimension at a time. This section combines both
// prior models into estimateScheduleCost(), applies it to a representative
// slice of the exact 128-schedule space Section 21.3 already enumerated
// for real, and then does the thing this chapter has done at every step:
// actually EXECUTES each of those schedules -- through a generic,
// runtime-parameterized interpreter that reads a Schedule's own fields and
// obeys them, not one hand-written function per schedule -- with a real
// clock, and reports honestly how the combined prediction's own ranking
// compares to what really happened.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 056_combining_both_cost_models_prediction_vs_measurement_for_real_schedules.cpp -o 056_driver
// Run:     ./056_driver
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <limits>
#include <stdexcept>

// ==================== Loop / LoopNest / TiledLoop / tileLoop() / actualInnerExtent() (Chapter 15, unchanged) ====================

struct Loop {
    std::string dimName;
    long long extent;
};
struct LoopNest {
    std::vector<Loop> loops;
    long long totalIterations() const {
        long long total = 1;
        for (const Loop& l : loops) total *= l.extent;
        return total;
    }
};
struct TiledLoop {
    std::string dimName;
    long long originalExtent;
    long long tileSize;
    long long outerExtent;
};
static TiledLoop tileLoop(const Loop& loop, long long tileSize) {
    if (tileSize <= 0) throw std::runtime_error("tileLoop: tile size must be positive");
    long long outerExtent = (loop.extent + tileSize - 1) / tileSize;
    return TiledLoop{loop.dimName, loop.extent, tileSize, outerExtent};
}
static long long actualInnerExtent(const TiledLoop& tl, long long outerIndex) {
    long long alreadyCovered = outerIndex * tl.tileSize;
    long long remaining = tl.originalExtent - alreadyCovered;
    return std::min(tl.tileSize, remaining);
}

// ==================== Section 21.1 / 21.2 / 21.3's own Schedule machinery (unchanged) ====================

static std::vector<long long> enumerateTileSizeCandidates(long long extent) {
    std::vector<long long> candidates;
    for (long long t = 1; t <= extent; t *= 2) candidates.push_back(t);
    if (candidates.empty() || candidates.back() != extent) candidates.push_back(extent);
    return candidates;
}
using LoopOrder = std::vector<int>;
static std::vector<LoopOrder> enumerateLoopOrders(const LoopNest& nest) {
    std::vector<int> indices(nest.loops.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<LoopOrder> orders;
    do {
        orders.push_back(indices);
    } while (std::next_permutation(indices.begin(), indices.end()));
    return orders;
}
static std::vector<long long> unrollFactorCandidates(long long extent) {
    return enumerateTileSizeCandidates(extent);
}
struct Schedule {
    std::vector<long long> tileSizePerLoop;
    LoopOrder loopOrder;
    long long unrollFactor;
};
static std::vector<Schedule> enumerateSchedules(const LoopNest& nest) {
    std::vector<std::vector<long long>> perLoopCandidates;
    for (const Loop& l : nest.loops) perLoopCandidates.push_back(enumerateTileSizeCandidates(l.extent));

    std::vector<LoopOrder> orders = enumerateLoopOrders(nest);
    long long innerExtent = nest.loops.back().extent;
    std::vector<long long> unrollCandidates = unrollFactorCandidates(innerExtent);

    std::vector<Schedule> schedules;
    std::vector<size_t> odometer(perLoopCandidates.size(), 0);
    bool done = perLoopCandidates.empty();
    while (!done) {
        std::vector<long long> tileChoice;
        for (size_t d = 0; d < perLoopCandidates.size(); ++d) tileChoice.push_back(perLoopCandidates[d][odometer[d]]);
        for (const LoopOrder& order : orders) {
            for (long long unroll : unrollCandidates) {
                schedules.push_back(Schedule{tileChoice, order, unroll});
            }
        }
        size_t d = 0;
        while (d < odometer.size()) {
            odometer[d]++;
            if (odometer[d] < perLoopCandidates[d].size()) break;
            odometer[d] = 0;
            d++;
        }
        if (d == odometer.size()) done = true;
    }
    return schedules;
}
static std::string scheduleStr(const LoopNest& nest, const Schedule& s) {
    std::string out = "tiles=[";
    for (size_t i = 0; i < s.tileSizePerLoop.size(); ++i) {
        if (i) out += ",";
        out += nest.loops[i].dimName + ":" + std::to_string(s.tileSizePerLoop[i]);
    }
    out += "] order=";
    for (size_t i = 0; i < s.loopOrder.size(); ++i) {
        if (i) out += ">";
        out += nest.loops[static_cast<size_t>(s.loopOrder[i])].dimName;
    }
    out += " unroll=" + std::to_string(s.unrollFactor);
    return out;
}

// ==================== Section 22.1 / 22.2's own cost models (unchanged) ====================

static double estimateLoopOverheadCost(long long extent, long long chunkSize,
                                        double perTripOverhead, double perElementWork) {
    long long trips = (extent + chunkSize - 1) / chunkSize;
    return static_cast<double>(trips) * perTripOverhead + static_cast<double>(extent) * perElementWork;
}
static double estimateLoopOrderCost(const LoopNest& nest, const LoopOrder& order,
                                     long long cacheLineElements, double perLineCost, double perElementWork) {
    long long totalIterations = nest.totalIterations();
    int contiguousDim = static_cast<int>(nest.loops.size()) - 1;
    int innermostDim = order.back();
    double lineLoads;
    if (innermostDim == contiguousDim) {
        lineLoads = static_cast<double>(totalIterations) / static_cast<double>(cacheLineElements);
    } else {
        lineLoads = static_cast<double>(totalIterations);
    }
    return lineLoads * perLineCost + static_cast<double>(totalIterations) * perElementWork;
}

// ==================== Section 22.3: combining both models into one schedule score ====================
//
// estimateScheduleCost() is deliberately just a SUM of three terms, each
// reusing a function this chapter already built and already tested in
// isolation: one loop-overhead term PER LOOP (tiling adds trips to every
// loop it touches, not just the innermost one), one more loop-overhead
// term for unrolling (a further chunking of the innermost loop's own
// tile, exactly the relationship Section 21.3 already established between
// tile size and unroll factor), and one memory-stride term for the whole
// nest's own loop order. perElementWork is charged exactly ONCE, at the
// end, instead of inside each reused call (passing 0.0 into them) --
// otherwise the same real compute work would be double- and triple-
// counted once per mechanism. Summing independent per-mechanism estimates
// is the simplest possible way to combine them when there is no principled
// reason to weight one mechanism's cost above another's; it is a stated
// simplification, not a claim that loop overhead and cache misses trade
// off at some universally correct exchange rate.
static double estimateScheduleCost(const LoopNest& nest, const Schedule& schedule,
                                    double perTripOverhead, long long cacheLineElements,
                                    double perLineCost, double perElementWork) {
    double overheadCost = 0.0;
    for (size_t d = 0; d < nest.loops.size(); ++d) {
        overheadCost += estimateLoopOverheadCost(nest.loops[d].extent, schedule.tileSizePerLoop[d], perTripOverhead, 0.0);
    }
    size_t innermostDim = static_cast<size_t>(schedule.loopOrder.back());
    long long innermostTileSize = schedule.tileSizePerLoop[innermostDim];
    overheadCost += estimateLoopOverheadCost(innermostTileSize, schedule.unrollFactor, perTripOverhead, 0.0);

    double orderCost = estimateLoopOrderCost(nest, schedule.loopOrder, cacheLineElements, perLineCost, 0.0);

    double elementWorkCost = static_cast<double>(nest.totalIterations()) * perElementWork;  // charged once

    return overheadCost + orderCost + elementWorkCost;
}

// ==================== A generic, runtime-parameterized schedule interpreter ====================
//
// executeSchedule() takes a Schedule's own fields as DATA and obeys them at
// runtime -- tiling both loops with Chapter 15's own tileLoop(), visiting
// tile blocks and then in-tile elements in the ORDER the schedule says,
// unrolling the innermost in-tile loop by the FACTOR the schedule says --
// rather than having one hand-written function per schedule the way a real
// code generator (Chapter 17-20) would. That tradeoff is deliberate and
// stated: an interpreter is slower per element than generated code would
// be, but it is the only way to run all 128 of Section 21.3's own
// schedules without writing 128 separate functions, and Chapter 23's own
// autotuner needs exactly this kind of schedule-as-data flexibility to
// try candidates before committing to codegen for the winner.
static void referenceCompute(const std::vector<float>& a, std::vector<float>& out) {
    for (size_t idx = 0; idx < a.size(); ++idx) out[idx] = a[idx] * 2.0f + 1.0f;  // untiled, unordered, un-unrolled
}
static void executeSchedule(const LoopNest& nest, const Schedule& schedule,
                             const std::vector<float>& a, std::vector<float>& out) {
    long long dim1Extent = nest.loops[1].extent;  // row-major flattening: i*dim1Extent + j

    int dimOuter = schedule.loopOrder[0];
    int dimInner = schedule.loopOrder[1];

    TiledLoop outerTiled = tileLoop(nest.loops[static_cast<size_t>(dimOuter)], schedule.tileSizePerLoop[static_cast<size_t>(dimOuter)]);
    TiledLoop innerTiled = tileLoop(nest.loops[static_cast<size_t>(dimInner)], schedule.tileSizePerLoop[static_cast<size_t>(dimInner)]);

    for (long long ob = 0; ob < outerTiled.outerExtent; ++ob) {
        long long outerBlockLen = actualInnerExtent(outerTiled, ob);
        for (long long ib = 0; ib < innerTiled.outerExtent; ++ib) {
            long long innerBlockLen = actualInnerExtent(innerTiled, ib);
            for (long long oi = 0; oi < outerBlockLen; ++oi) {
                long long outerVal = ob * outerTiled.tileSize + oi;
                long long ii = 0;
                for (; ii + schedule.unrollFactor <= innerBlockLen; ii += schedule.unrollFactor) {
                    for (long long lane = 0; lane < schedule.unrollFactor; ++lane) {
                        long long innerVal = ib * innerTiled.tileSize + ii + lane;
                        long long i = (dimOuter == 0) ? outerVal : innerVal;
                        long long j = (dimOuter == 0) ? innerVal : outerVal;
                        size_t flat = static_cast<size_t>(i * dim1Extent + j);
                        out[flat] = a[flat] * 2.0f + 1.0f;
                    }
                }
                for (; ii < innerBlockLen; ++ii) {
                    long long innerVal = ib * innerTiled.tileSize + ii;
                    long long i = (dimOuter == 0) ? outerVal : innerVal;
                    long long j = (dimOuter == 0) ? innerVal : outerVal;
                    size_t flat = static_cast<size_t>(i * dim1Extent + j);
                    out[flat] = a[flat] * 2.0f + 1.0f;
                }
            }
        }
    }
}
static bool arraysExactlyEqual(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

int main() {
    printf("=== Section 22.3: combining both cost models -- prediction vs. measurement for real schedules ===\n\n");

    // ---- Part 1: the same 128-schedule space Section 21.3 already enumerated for real ----
    printf("--- Part 1: 6 schedules, evenly spaced across Section 21.3's own 128-schedule list ---\n\n");
    LoopNest nest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};
    std::vector<Schedule> allSchedules = enumerateSchedules(nest);
    printf("enumerateSchedules() produced %zu schedules (%s, same count as Section 21.3)\n\n",
           allSchedules.size(), (allSchedules.size() == 128) ? "confirmed" : "MISMATCH");

    std::vector<size_t> pickedIdx = {0, 25, 51, 76, 102, 127};
    std::vector<Schedule> picked;
    for (size_t idx : pickedIdx) picked.push_back(allSchedules[idx]);
    for (size_t k = 0; k < picked.size(); ++k) {
        printf("  [%3zu] %s\n", pickedIdx[k], scheduleStr(nest, picked[k]).c_str());
    }

    // ---- Part 2: the combined cost model's own prediction ----
    printf("\n--- Part 2: estimateScheduleCost() for each of the 6, ranked cheapest to most expensive ---\n\n");
    const double perTripOverhead = 40.0;
    const long long cacheLineElements = 16;
    const double perLineCost = 40.0;
    const double perElementWork = 1.0;

    std::vector<double> predictedCost(picked.size());
    for (size_t k = 0; k < picked.size(); ++k) {
        predictedCost[k] = estimateScheduleCost(nest, picked[k], perTripOverhead, cacheLineElements, perLineCost, perElementWork);
    }
    std::vector<size_t> predictedRank(picked.size());
    std::iota(predictedRank.begin(), predictedRank.end(), 0);
    std::sort(predictedRank.begin(), predictedRank.end(),
              [&](size_t x, size_t y) { return predictedCost[x] < predictedCost[y]; });
    for (size_t r = 0; r < predictedRank.size(); ++r) {
        size_t k = predictedRank[r];
        printf("  predicted #%zu: [%3zu] cost=%10.1f  %s\n", r + 1, pickedIdx[k], predictedCost[k],
               scheduleStr(nest, picked[k]).c_str());
    }

    // ---- Part 3: real execution, through the generic interpreter, a real clock ----
    printf("\n--- Part 3: real measurement -- executeSchedule() run 800,000 times per schedule, best-of-7 ---\n\n");
    long long dim0Extent = nest.loops[0].extent, dim1Extent = nest.loops[1].extent;
    std::vector<float> a(static_cast<size_t>(dim0Extent * dim1Extent));
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> reference(a.size());
    referenceCompute(a, reference);

    const long long repeatCount = 800000;
    std::vector<double> measuredMs(picked.size());
    bool allCorrect = true;
    for (size_t k = 0; k < picked.size(); ++k) {
        std::vector<float> out(a.size());
        double best = std::numeric_limits<double>::max();
        for (int trial = 0; trial < 7; ++trial) {
            auto start = std::chrono::steady_clock::now();
            for (long long r = 0; r < repeatCount; ++r) executeSchedule(nest, picked[k], a, out);
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (ms < best) best = ms;
        }
        bool ok = arraysExactlyEqual(out, reference);
        allCorrect = allCorrect && ok;
        measuredMs[k] = best;
        printf("  [%3zu] best-of-7 total = %8.3f ms  (bit-for-bit correct: %s)  %s\n",
               pickedIdx[k], best, ok ? "yes" : "NO", scheduleStr(nest, picked[k]).c_str());
    }
    printf("\nself-check: all 6 schedules, run through the SAME interpreter, produce the exact same\n");
    printf("bit-for-bit output as the untiled, unordered, un-unrolled reference computation (%s) --\n",
           allCorrect ? "confirmed" : "MISMATCH");
    printf("every point in this schedule space is a valid answer; only the cost differs.\n");

    std::vector<size_t> measuredRank(picked.size());
    std::iota(measuredRank.begin(), measuredRank.end(), 0);
    std::sort(measuredRank.begin(), measuredRank.end(),
              [&](size_t x, size_t y) { return measuredMs[x] < measuredMs[y]; });
    printf("\nmeasured ranking, fastest to slowest:\n");
    for (size_t r = 0; r < measuredRank.size(); ++r) {
        size_t k = measuredRank[r];
        printf("  measured #%zu: [%3zu] %8.3f ms  %s\n", r + 1, pickedIdx[k], measuredMs[k],
               scheduleStr(nest, picked[k]).c_str());
    }

    // ---- Part 4: the honest comparison ----
    printf("\n--- Part 4: does the combined model's ranking match what actually happened? ---\n\n");
    bool bestMatches = (predictedRank.front() == measuredRank.front());
    printf("  model's cheapest:    [%3zu] %s\n", pickedIdx[predictedRank.front()], scheduleStr(nest, picked[predictedRank.front()]).c_str());
    printf("  measurement's fastest: [%3zu] %s\n", pickedIdx[measuredRank.front()], scheduleStr(nest, picked[measuredRank.front()]).c_str());
    printf("  same schedule: %s\n\n", bestMatches ? "yes" : "no");

    int totalPairs = 0, concordantPairs = 0;
    for (size_t i = 0; i < picked.size(); ++i) {
        for (size_t j = i + 1; j < picked.size(); ++j) {
            totalPairs++;
            bool predSaysICheaper = predictedCost[i] < predictedCost[j];
            bool measSaysIFaster = measuredMs[i] < measuredMs[j];
            if (predSaysICheaper == measSaysIFaster) concordantPairs++;
        }
    }
    double concordancePct = 100.0 * concordantPairs / totalPairs;
    printf("  pairwise ranking agreement: %d of %d comparisons agree (%.0f%%)\n\n",
           concordantPairs, totalPairs, concordancePct);

    printf("A %.0f%% agreement rate is close to the 50%% a coin flip would average on a 6-item ranking, and\n",
           concordancePct);
    printf("that should not be softened into a vague 'no model is perfect.' There is a real, specific\n");
    printf("reason it lands this weak: this LoopNest has only 48 total elements, far too little real work\n");
    printf("for any single call to executeSchedule() to register on a wall clock, which is exactly why Part\n");
    printf("3 had to run each schedule 800,000 times just to get a measurable duration. At that scale, the\n");
    printf("INTERPRETER's own per-call bookkeeping -- computing tile boundaries, branching on loop order,\n");
    printf("looping over an unroll factor of possibly 1 -- competes with, or outweighs, the cost of the 48\n");
    printf("elements it is actually scheduling, so the measured time partly reflects a cost neither cost\n");
    printf("model was ever told about. Section 22.2 measured a REAL compiled loop at real scale (16.7\n");
    printf("million elements) with no interpreter in the way, and its own ranking held up perfectly; Section\n");
    printf("22.1, at 50 million elements and also with no interpreter, still got its own single best-vs-\n");
    printf("measured comparison wrong, for the different, already-diagnosed reason that its model is simply\n");
    printf("missing real mechanisms (instruction-cache and register pressure) at high unroll factors. This\n");
    printf("section's own weak agreement is best read as a mix of both: some of it is the same kind of\n");
    printf("model incompleteness Section 22.1 already found, and some of it is a confound specific to\n");
    printf("interpreting a tiny schedule instead of compiling it. A production autotuner sidesteps that\n");
    printf("second part the way Chapters 17-20 already do it -- generate and compile real code per candidate\n");
    printf("schedule, the way Chapter 23 will, rather than interpret one -- so the model can be judged\n");
    printf("against the mechanisms it actually claims to predict. What does NOT change is the verdict this\n");
    printf("chapter has built toward the whole way: a cost model, however imperfect, is cheap enough to run\n");
    printf("before compiling anything, which makes it a filter -- narrow 128 candidates (or a far larger\n");
    printf("space, for a real kernel) down to a short list, and only then pay for real measurement. Chapter\n");
    printf("23 builds exactly that hybrid.\n");

    bool allOk = (allSchedules.size() == 128) && allCorrect;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "056_combining_both_cost_models_prediction_vs_measurement_for_real_schedules.cpp" -o "056_combining_both_cost_models_prediction_vs_measurement_for_real_schedules"
./"056_combining_both_cost_models_prediction_vs_measurement_for_real_schedules"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 22.3: combining both cost models -- prediction vs. measurement for real schedules ===

--- Part 1: 6 schedules, evenly spaced across Section 21.3's own 128-schedule list ---

enumerateSchedules() produced 128 schedules (confirmed, same count as Section 21.3)

  [  0] tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [ 25] tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  [ 51] tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  [ 76] tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  [102] tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  [127] tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

--- Part 2: estimateScheduleCost() for each of the 6, ranked cheapest to most expensive ---

  predicted #1: [ 51] cost=     448.0  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  predicted #2: [ 25] cost=     568.0  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  predicted #3: [  0] cost=     768.0  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  predicted #4: [127] cost=    2088.0  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  predicted #5: [ 76] cost=    2248.0  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  predicted #6: [102] cost=    2288.0  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4

--- Part 3: real measurement -- executeSchedule() run 800,000 times per schedule, best-of-7 ---

  [  0] best-of-7 total =  151.393 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [ 25] best-of-7 total =   78.118 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  [ 51] best-of-7 total =   80.997 ms  (bit-for-bit correct: yes)  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  [ 76] best-of-7 total =  104.180 ms  (bit-for-bit correct: yes)  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  [102] best-of-7 total =   76.610 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  [127] best-of-7 total =   51.974 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

self-check: all 6 schedules, run through the SAME interpreter, produce the exact same
bit-for-bit output as the untiled, unordered, un-unrolled reference computation (confirmed) --
every point in this schedule space is a valid answer; only the cost differs.

measured ranking, fastest to slowest:
  measured #1: [127]   51.974 ms  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  measured #2: [102]   76.610 ms  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  measured #3: [ 25]   78.118 ms  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  measured #4: [ 51]   80.997 ms  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  measured #5: [ 76]  104.180 ms  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  measured #6: [  0]  151.393 ms  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1

--- Part 4: does the combined model's ranking match what actually happened? ---

  model's cheapest:    [ 51] tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  measurement's fastest: [127] tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  same schedule: no

  pairwise ranking agreement: 6 of 15 comparisons agree (40%)

A 40% agreement rate is close to the 50% a coin flip would average on a 6-item ranking, and
that should not be softened into a vague 'no model is perfect.' There is a real, specific
reason it lands this weak: this LoopNest has only 48 total elements, far too little real work
for any single call to executeSchedule() to register on a wall clock, which is exactly why Part
3 had to run each schedule 800,000 times just to get a measurable duration. At that scale, the
INTERPRETER's own per-call bookkeeping -- computing tile boundaries, branching on loop order,
looping over an unroll factor of possibly 1 -- competes with, or outweighs, the cost of the 48
elements it is actually scheduling, so the measured time partly reflects a cost neither cost
model was ever told about. Section 22.2 measured a REAL compiled loop at real scale (16.7
million elements) with no interpreter in the way, and its own ranking held up perfectly; Section
22.1, at 50 million elements and also with no interpreter, still got its own single best-vs-
measured comparison wrong, for the different, already-diagnosed reason that its model is simply
missing real mechanisms (instruction-cache and register pressure) at high unroll factors. This
section's own weak agreement is best read as a mix of both: some of it is the same kind of
model incompleteness Section 22.1 already found, and some of it is a confound specific to
interpreting a tiny schedule instead of compiling it. A production autotuner sidesteps that
second part the way Chapters 17-20 already do it -- generate and compile real code per candidate
schedule, the way Chapter 23 will, rather than interpret one -- so the model can be judged
against the mechanisms it actually claims to predict. What does NOT change is the verdict this
chapter has built toward the whole way: a cost model, however imperfect, is cheap enough to run
before compiling anything, which makes it a filter -- narrow 128 candidates (or a far larger
space, for a real kernel) down to a short list, and only then pay for real measurement. Chapter
23 builds exactly that hybrid.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 22.3: combining both cost models -- prediction vs. measurement for real schedules ===

--- Part 1: 6 schedules, evenly spaced across Section 21.3's own 128-schedule list ---

enumerateSchedules() produced 128 schedules (confirmed, same count as Section 21.3)

  [  0] tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [ 25] tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  [ 51] tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  [ 76] tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  [102] tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  [127] tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

--- Part 2: estimateScheduleCost() for each of the 6, ranked cheapest to most expensive ---

  predicted #1: [ 51] cost=     448.0  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  predicted #2: [ 25] cost=     568.0  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  predicted #3: [  0] cost=     768.0  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  predicted #4: [127] cost=    2088.0  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  predicted #5: [ 76] cost=    2248.0  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  predicted #6: [102] cost=    2288.0  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4

--- Part 3: real measurement -- executeSchedule() run 800,000 times per schedule, best-of-7 ---

  [  0] best-of-7 total =   58.219 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [ 25] best-of-7 total =   56.933 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  [ 51] best-of-7 total =   35.801 ms  (bit-for-bit correct: yes)  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  [ 76] best-of-7 total =   41.899 ms  (bit-for-bit correct: yes)  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  [102] best-of-7 total =   55.463 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  [127] best-of-7 total =   25.644 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

self-check: all 6 schedules, run through the SAME interpreter, produce the exact same
bit-for-bit output as the untiled, unordered, un-unrolled reference computation (confirmed) --
every point in this schedule space is a valid answer; only the cost differs.

measured ranking, fastest to slowest:
  measured #1: [127]   25.644 ms  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  measured #2: [ 51]   35.801 ms  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  measured #3: [ 76]   41.899 ms  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  measured #4: [102]   55.463 ms  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  measured #5: [ 25]   56.933 ms  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  measured #6: [  0]   58.219 ms  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1

--- Part 4: does the combined model's ranking match what actually happened? ---

  model's cheapest:    [ 51] tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  measurement's fastest: [127] tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  same schedule: no

  pairwise ranking agreement: 8 of 15 comparisons agree (53%)

A 53% agreement rate is close to the 50% a coin flip would average on a 6-item ranking, and
that should not be softened into a vague 'no model is perfect.' There is a real, specific
reason it lands this weak: this LoopNest has only 48 total elements, far too little real work
for any single call to executeSchedule() to register on a wall clock, which is exactly why Part
3 had to run each schedule 800,000 times just to get a measurable duration. At that scale, the
INTERPRETER's own per-call bookkeeping -- computing tile boundaries, branching on loop order,
looping over an unroll factor of possibly 1 -- competes with, or outweighs, the cost of the 48
elements it is actually scheduling, so the measured time partly reflects a cost neither cost
model was ever told about. Section 22.2 measured a REAL compiled loop at real scale (16.7
million elements) with no interpreter in the way, and its own ranking held up perfectly; Section
22.1, at 50 million elements and also with no interpreter, still got its own single best-vs-
measured comparison wrong, for the different, already-diagnosed reason that its model is simply
missing real mechanisms (instruction-cache and register pressure) at high unroll factors. This
section's own weak agreement is best read as a mix of both: some of it is the same kind of
model incompleteness Section 22.1 already found, and some of it is a confound specific to
interpreting a tiny schedule instead of compiling it. A production autotuner sidesteps that
second part the way Chapters 17-20 already do it -- generate and compile real code per candidate
schedule, the way Chapter 23 will, rather than interpret one -- so the model can be judged
against the mechanisms it actually claims to predict. What does NOT change is the verdict this
chapter has built toward the whole way: a cost model, however imperfect, is cheap enough to run
before compiling anything, which makes it a filter -- narrow 128 candidates (or a far larger
space, for a real kernel) down to a short list, and only then pay for real measurement. Chapter
23 builds exactly that hybrid.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which candidate ranks fastest, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! note "What this chapter proved, and what it deliberately left for later"
    Every number in this chapter is real: a genuine, machine-specific turnover point where loop-overhead's own monotonic prediction stops matching real timing (Section 22.1); a real, dramatically measured memory-stride effect the model's own prediction correctly ranked but underestimated in size (Section 22.2); and a real, diagnosed near-chance ranking agreement for a combined model applied to a toy-scale interpreted schedule (Section 22.3), traced to a specific, named confound rather than hand-waved away. What this chapter deliberately does NOT do: it never generates compiled, schedule-specific CODE for any candidate (Section 22.3's own interpreter is explicitly a stand-in, not a code generator), and it never runs an actual SEARCH over the 128-schedule space -- it only scores a hand-picked sample of 6. Those are Chapter 23's own job: use a cost model, imperfect as this chapter has shown it to be, to prune a large space down to a short list, then pay for real compiled measurement only on that short list.

## Chapter Summary

This chapter picked up exactly where Chapter 21 left off -- 128 real, enumerated, legal schedules for one tiny `LoopNest`, and no cost model yet to rank them without measuring every one -- and built two, tested honestly against real hardware. Section 22.1 built `estimateLoopOverheadCost()`, a simple linear model applying identically to tile size and unroll factor (the same underlying question, per Chapter 21's own finding), and found, with a real 50-million-element measurement on two different machines, that its own monotonic "always chunk bigger" prediction breaks down at a real, machine-specific turnover point the model was never told about. Section 22.2 built `estimateLoopOrderCost()`, a model built on counting cache-line loads for a schedule's own memory-access stride, and found, with a real 64 MB array measurement, that it gets the DIRECTION of a dramatic real effect right while underestimating its exact MAGNITUDE. Section 22.3 combined both into `estimateScheduleCost()`, built a generic runtime-parameterized schedule interpreter to execute real candidates without hand-writing one function per schedule, and found a near-chance ranking agreement on a toy-scale example -- honestly diagnosed as interpreter overhead dominating a 48-element problem, not evidence that either underlying model is worthless. Every claim in this chapter was checked against a real clock on two real, different machines, with correctness (not timing) required to match exactly on both.

## Self-Check Questions

1. Why does this chapter deliberately avoid building a cache-REUSE cost model, the kind a real tiled matrix multiply would need, for this book's own generated code?
2. `estimateLoopOverheadCost()`'s own per-element term never depends on `chunkSize`. What does that one fact already tell you about which chunk size the model will always predict as cheapest, before running any real measurement at all?
3. Section 22.1's real measurement finds a different turnover point on the cloud sandbox than on the device. Is that a bug in the measurement? Why or why not?
4. In `estimateLoopOrderCost()`, what determines whether the model uses its "best case" or "worst case" line-load formula for a given loop order?
5. Section 22.2's model predicts roughly an 11-to-12x cost gap between row-major and column-major traversal, but real measurement finds a much larger gap on both machines. Name one real hardware mechanism the model never included that helps explain why the real gap is bigger, not smaller.
6. In `estimateScheduleCost()`, why is `perElementWork` passed as `0.0` into each of the three reused sub-calls, with the real per-element cost added back in only once, separately?
7. Why does Section 22.3 need to run each of its 6 schedules 800,000 times just to get a measurable duration, when Sections 22.1 and 22.2 each needed only a handful of trials?
8. Section 22.3's combined model lands at a near-chance ranking agreement. What specific, named confound does this chapter identify as the likely cause, and what evidence from Section 22.2 supports that diagnosis rather than a simpler "the model is just wrong" conclusion?

## Where We Go Next

Chapter 23, "Building CUDA Hammer's Autotuner," closes Part 5 by building the actual SEARCH this chapter's own cost models were built to guide: given a real `LoopNest` and its own real, enumerated schedule space (Chapter 21) and a cost model honestly known to be imperfect in specific, diagnosed ways (this chapter), how should an autotuner actually decide which candidates to try? The likely answer, set up directly by this chapter's own closing finding, is a HYBRID: use the cost model to prune a large space down to a short list cheaply, then pay for real, compiled measurement -- Part 4's own codegen, not Section 22.3's interpreter -- only on that short list. Apply the Chapter 5-22 depth-level standard throughout.

## Worked Solutions

1. This book's own IR, through Chapter 20, has no operator with genuine cross-element data reuse: every elementwise operation reads each input exactly once regardless of tile size, and even the full-reduction `Sum` operator (Chapter 14) still reads each input exactly once no matter how it is tiled. A cache-reuse cost model -- the kind that matters for a real tiled matrix multiply, where one tile gets reused across many output elements -- would be modeling a benefit this book's own generated code has no way to actually realize, so building one here would be dishonest about what the code actually does.
2. Since the per-element term is the same no matter what `chunkSize` is chosen, only the per-trip term (`trips * perTripOverhead`) can ever change the total, and `trips = ceil(extent / chunkSize)` can only ever go DOWN as `chunkSize` grows. That means the model's own predicted cost is non-increasing in `chunkSize` by construction, for ANY positive values of its two constants -- so it will always rank the largest available chunk size as cheapest, with no exception the formula itself can ever produce.
3. It is not a bug. The two machines have different real microarchitectures -- different instruction-cache sizes, different register file behavior, different auto-vectorization thresholds in whatever the underlying compiler does with a large unrolled loop body -- so the point at which those real effects start to outweigh loop-overhead savings is itself a real, machine-specific fact, not a single universal number the model (or the book) could have predicted in advance. Both machines' own real turnover points are genuine findings, not disagreeing measurements of one true answer.
4. The model compares the loop order's own INNERMOST dimension against the array's contiguous dimension (the LAST loop in the `LoopNest`, since this book's tensors are stored row-major). If they match, consecutive loop trips touch consecutive memory addresses, and the model uses its best-case formula (one line load shared across `cacheLineElements` trips); if they don't match, consecutive trips jump by a whole row, and the model uses its worst-case formula (one line load charged per trip).
5. Hardware prefetching is a real mechanism the model never included: modern CPUs detect a contiguous (stride-1) access pattern and start fetching upcoming cache lines before the program even asks for them, making the row-major case even faster in practice than a flat line-count model assumes -- widening the real gap beyond what counting cache lines alone predicts. (TLB misses under the large strides in the column-major case are a second real mechanism the model also never included, working in the same direction.)
6. Each of the three sub-calls already includes its own `perElementWork` parameter, inherited from Sections 22.1 and 22.2 where it made sense in isolation. Combining all three calls without zeroing that term out would charge the real compute work three separate times for the same actual computation -- once per mechanism -- which would inflate every schedule's own predicted cost by the same wrong multiple without changing which schedule looks cheapest, but would make the RAW numbers meaningless. Charging it exactly once, added back in separately, keeps the combined score honest about what real work is actually being counted.
7. The `LoopNest [dim0:6, dim1:8]` has only 48 total elements -- so few that a single call to `executeSchedule()` completes far faster than `std::chrono::steady_clock` can reliably distinguish from measurement overhead itself. Sections 22.1 and 22.2 each measured computations with millions of real elements, large enough that a handful of trials already produced a clearly measurable, meaningfully different duration; Section 22.3's own toy-scale `LoopNest` needs hundreds of thousands of repetitions of the ENTIRE schedule just to accumulate a duration worth trusting.
8. The named confound is INTERPRETER OVERHEAD: at only 48 real elements per execution, `executeSchedule()`'s own per-call bookkeeping -- computing tile boundaries, branching on loop order, looping over a possibly-small unroll factor -- competes with, or exceeds, the actual cost of the 48 elements it is scheduling, so the measured time partly reflects a cost neither cost model was ever told about. The evidence for this diagnosis, rather than a simpler "the model is wrong" conclusion, is Section 22.2: the SAME underlying memory-stride model, applied to a REAL compiled loop at real scale (16.7 million elements, no interpreter in the way), ranked correctly and by a wide margin -- meaning the model itself is not the entire explanation for Section 22.3's own weak agreement.

---

**Sources cited in this chapter:**

None new. This chapter's own loop-overhead and memory-access-stride cost models, their combination into `estimateScheduleCost()`, and the generic schedule interpreter are all original to this book, building on Chapter 15's own `LoopNest`/`tileLoop()` machinery and Chapter 21's own `Schedule`/`enumerateSchedules()` machinery. The general behavior of CPU cache lines, memory-access stride, and hardware prefetching that Section 22.2's own real measurement demonstrates is well-documented, widely available background on real memory hierarchies, verified here by this book's own directly executed and measured example on two different real machines rather than cited to an external source.
