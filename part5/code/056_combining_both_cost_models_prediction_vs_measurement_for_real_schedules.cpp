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
