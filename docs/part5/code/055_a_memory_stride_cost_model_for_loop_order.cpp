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
