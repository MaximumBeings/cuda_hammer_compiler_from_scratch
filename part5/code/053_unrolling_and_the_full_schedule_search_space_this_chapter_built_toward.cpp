// Chapter 21: The Search Space: Tile Sizes, Loop Orders, and Unrolling
// 053_unrolling_and_the_full_schedule_search_space_this_chapter_built_toward.cpp
//
// Section 21.3 (capstone) -- this chapter builds and validates the SEARCH
// SPACE itself: which schedules are even legal, and how large that space
// really is. It deliberately does NOT yet generate scheduled code (Chapter
// 22 needs a cost model first, to avoid measuring every single candidate)
// or run an actual autotuning search (Chapter 23) -- every worked example
// in this chapter still compiles and runs for real, computing genuine
// numbers about the search space itself, never simulated in a comment.
// This section adds the third and final schedule dimension -- UNROLL
// FACTOR, the software analogue of Chapter 19's own SIMD width: instead of
// one hardware instruction covering W lanes, an unrolled loop processes W
// scalar iterations per trip, advancing by W each time, with a scalar tail
// for whatever's left over -- the exact same "main loop + tail" shape
// Section 19.1 already established for real vector hardware, this time as
// a free software choice instead of a fixed hardware width. It then
// combines all three dimensions -- tile size (21.1), loop order (21.2),
// unroll factor (this section) -- into one Schedule, and enumerates the
// full CARTESIAN PRODUCT for a real LoopNest, closing the chapter on the
// combinatorial-explosion problem Chapter 22 and 23 both exist to solve.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 053_unrolling_and_the_full_schedule_search_space_this_chapter_built_toward.cpp -o 053_driver
// Run:     ./053_driver
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

// ==================== Loop / LoopNest (Chapter 15, unchanged) ====================

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

// ==================== Section 21.1's own tile-size candidates (unchanged) ====================

static std::vector<long long> enumerateTileSizeCandidates(long long extent) {
    std::vector<long long> candidates;
    for (long long t = 1; t <= extent; t *= 2) candidates.push_back(t);
    if (candidates.empty() || candidates.back() != extent) candidates.push_back(extent);
    return candidates;
}

// ==================== Section 21.2's own loop-order enumeration (unchanged) ====================

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

// ==================== Section 21.3: unrollFactorCandidates() and a real unrolled loop ====================
//
// The candidate list is the exact same doubling-plus-extent shape Section
// 21.1 already used for tile sizes -- unroll factor and tile size are, in
// fact, the same underlying idea (how many elements does one loop trip
// cover) applied to two different concerns (memory blocking vs. reducing
// per-iteration loop overhead), so it would be a real inconsistency for
// their own candidate-generation logic to differ.
static std::vector<long long> unrollFactorCandidates(long long extent) {
    return enumerateTileSizeCandidates(extent);
}

// computeUnrolled() actually EXECUTES a real elementwise computation --
// out[i] = a[i] * 2.0f + 1.0f -- using an unroll factor exactly the way a
// compiled unrolled loop would: a main loop processing `unrollFactor`
// elements per trip (advancing i by unrollFactor each time), followed by a
// scalar tail for whatever remains when extent isn't evenly divisible.
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
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;  // bit-for-bit, on purpose
    return true;
}

// ==================== The capstone: Schedule / enumerateSchedules() ====================
//
// A Schedule ties together one choice from EACH of this chapter's own three
// dimensions: a tile size PER LOOP (21.1 already showed the candidate list
// differs per loop, since it depends on that loop's own extent), ONE loop
// order for the whole nest (21.2), and ONE unroll factor applied to the
// innermost loop after tiling and ordering are both fixed (21.3). This is
// deliberately the simplest possible schedule representation that still
// has all three real dimensions in it -- a production autotuner's own
// Schedule would likely have more (per-loop unroll, vectorization width,
// memory-staging choices), but every one of those is the SAME cartesian-
// product idea this section already makes concrete.
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
    // Cartesian product over EVERY loop's own tile-size candidates, via a
    // simple odometer (base-varies-per-digit counter) -- the most direct,
    // most literal way to enumerate a cartesian product, no library needed.
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
        // advance the odometer
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

int main() {
    printf("=== Section 21.3: unrolling, and the full schedule search space this chapter built toward ===\n\n");

    // ---- Part 1: unrolling a real elementwise computation, two different factors ----
    printf("--- Unrolling a real computation: out[i] = a[i]*2+1, extent=13 (not evenly divisible) ---\n\n");
    std::vector<float> a;
    for (int i = 0; i < 13; ++i) a.push_back(static_cast<float>(i) * 0.5f - 3.0f);
    std::vector<float> baseline = computeUnrolled(a, 1);  // unroll factor 1 == the plain, un-unrolled loop

    bool allExact = true;
    for (long long factor : {2LL, 4LL, 8LL}) {
        std::vector<float> unrolled = computeUnrolled(a, factor);
        bool ok = arraysExactlyEqual(unrolled, baseline);
        allExact = allExact && ok;
        long long fullTrips = 13 / factor, tailElements = 13 % factor;
        printf("  unrollFactor=%-2lld -> %lld full trip(s) of %lld + a %lld-element scalar tail, bit-for-bit\n",
               factor, fullTrips, factor, tailElements);
        printf("                 identical to the un-unrolled result (%s)\n", ok ? "confirmed" : "MISMATCH");
    }
    printf("\nself-check: unlike Section 21.2's own reduction-order finding, unrolling an ELEMENTWISE loop\n");
    printf("never changes so much as one bit of the result -- every iteration is independent, so\n");
    printf("processing them %s at a time instead of one at a time is a PURE performance knob (%s)\n",
           "W", allExact ? "confirmed" : "MISMATCH");

    // ---- Part 2: the capstone -- one Schedule, all three dimensions, enumerated for real ----
    printf("\n--- The capstone: tile size x loop order x unroll factor, ONE real LoopNest ---\n\n");
    LoopNest nest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};
    std::vector<Schedule> schedules = enumerateSchedules(nest);

    size_t tileCountDim0 = enumerateTileSizeCandidates(nest.loops[0].extent).size();
    size_t tileCountDim1 = enumerateTileSizeCandidates(nest.loops[1].extent).size();
    size_t orderCount = enumerateLoopOrders(nest).size();
    size_t unrollCount = unrollFactorCandidates(nest.loops.back().extent).size();

    printf("LoopNest [dim0:6, dim1:8] (48 total iterations):\n");
    printf("  dim0 tile-size candidates: %zu\n", tileCountDim0);
    printf("  dim1 tile-size candidates: %zu\n", tileCountDim1);
    printf("  loop orders:               %zu\n", orderCount);
    printf("  unroll-factor candidates:  %zu (applied to the innermost loop, dim1, extent 8)\n\n", unrollCount);

    size_t expectedCount = tileCountDim0 * tileCountDim1 * orderCount * unrollCount;
    printf("cartesian product: %zu x %zu x %zu x %zu = %zu total schedules\n", tileCountDim0, tileCountDim1,
           orderCount, unrollCount, expectedCount);
    printf("enumerateSchedules() actually produced: %zu schedules (%s)\n\n", schedules.size(),
           (schedules.size() == expectedCount) ? "confirmed" : "MISMATCH");

    printf("a few real schedules from that list:\n");
    for (size_t i : {static_cast<size_t>(0), schedules.size() / 2, schedules.size() - 1}) {
        printf("  [%zu] %s\n", i, scheduleStr(nest, schedules[i]).c_str());
    }

    printf("\nself-check: %zu schedules for a TINY 2-loop, 48-iteration nest -- a real fused kernel in\n",
           schedules.size());
    printf("this book has more loops (multi-dim shapes, Chapter 6), larger extents, and this book's own\n");
    printf("candidate lists are already a REDUCED heuristic (Section 21.1), not the naive full search --\n");
    printf("the space only grows from here. Brute-force MEASURING every one is not how Chapter 23's own\n");
    printf("autotuner will work; Chapter 22 builds a cost model specifically to rank candidates without\n");
    printf("compiling and running every single one of them.\n");

    bool allOk = allExact && (schedules.size() == expectedCount);
    return allOk ? 0 : 1;
}
