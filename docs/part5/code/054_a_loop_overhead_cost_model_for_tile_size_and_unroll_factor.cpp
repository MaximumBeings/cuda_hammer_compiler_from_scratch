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
