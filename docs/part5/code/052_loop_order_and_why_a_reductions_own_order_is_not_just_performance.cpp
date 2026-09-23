// Chapter 21: The Search Space: Tile Sizes, Loop Orders, and Unrolling
// 052_loop_order_and_why_a_reductions_own_order_is_not_just_performance.cpp
//
// Section 21.2 -- Section 21.1 built the search space for ONE loop's own
// tile size. A real LoopNest usually has more than one loop, and a second,
// independent schedule dimension asks a different question entirely: given
// several loops, in what ORDER should they nest? This section builds
// enumerateLoopOrders() (every permutation of a LoopNest's own loops) and
// then proves something that sounds obvious but genuinely is not always
// true: for an ELEMENTWISE loop nest, EVERY order computes the exact same
// answer, because each iteration is independent of every other. For a
// REDUCTION, that is no longer quite true -- not because any order is
// WRONG, but because float32 addition is not associative, and this section
// proves that concretely, with real numbers, rather than asserting it.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 052_loop_order_and_why_a_reductions_own_order_is_not_just_performance.cpp -o 052_driver
// Run:     ./052_driver
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>

// ==================== Loop / LoopNest (from Chapter 15, unchanged) ====================

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

// ==================== Section 21.2: LoopOrder / enumerateLoopOrders() ====================
//
// A LoopOrder is a permutation of a LoopNest's own loop INDICES, listed
// outermost-first -- {0, 1} means "loop 0 is outer, loop 1 is inner";
// {1, 0} means the reverse. enumerateLoopOrders() returns every one of the
// N! permutations for an N-loop nest, using std::next_permutation the same
// way Section 21.1 used a plain doubling loop: a small, standard-library
// building block, not a hand-rolled combinatorics routine.
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
static std::string loopOrderStr(const LoopNest& nest, const LoopOrder& order) {
    std::string s;
    for (size_t i = 0; i < order.size(); ++i) {
        if (i) s += " -> ";
        s += nest.loops[static_cast<size_t>(order[i])].dimName;
    }
    return s;
}

// ==================== Proof by enumeration, extended from Chapter 15 to ORDER instead of tile boundaries ====================
//
// enumerateVisitedPoints() walks a LoopNest in a GIVEN order and records the
// full coordinate tuple (dim0 index, dim1 index, ...) for every iteration,
// always in the loop nest's own NATURAL dimension order regardless of which
// order the loops were actually nested in -- so two different loop orders
// can be compared directly by set equality, the same discipline Chapter
// 15's own enumerateUntiled()/enumerateTiled() pair already established.
static void enumerateVisitedPointsRec(const LoopNest& nest, const LoopOrder& order, size_t depth,
                                       std::vector<long long>& current, std::set<std::vector<long long>>& out) {
    if (depth == order.size()) {
        out.insert(current);
        return;
    }
    int dim = order[depth];
    for (long long i = 0; i < nest.loops[static_cast<size_t>(dim)].extent; ++i) {
        current[static_cast<size_t>(dim)] = i;
        enumerateVisitedPointsRec(nest, order, depth + 1, current, out);
    }
}
static std::set<std::vector<long long>> enumerateVisitedPoints(const LoopNest& nest, const LoopOrder& order) {
    std::set<std::vector<long long>> out;
    std::vector<long long> current(nest.loops.size(), 0);
    enumerateVisitedPointsRec(nest, order, 0, current, out);
    return out;
}

int main() {
    printf("=== Section 21.2: enumerateLoopOrders() -- and what order does and doesn't change ===\n\n");

    // ---- Part 1: an ELEMENTWISE loop nest -- Chapter 15's own diamond graph shape ----
    LoopNest nest{{Loop{"dim0", 3}, Loop{"dim1", 4}}};
    std::vector<LoopOrder> orders = enumerateLoopOrders(nest);
    printf("LoopNest %s (elementwise, %lld total iterations): %zu loop orders\n\n",
           "[dim0:3, dim1:4]", nest.totalIterations(), orders.size());

    std::set<std::vector<long long>> baseline = enumerateVisitedPoints(nest, orders[0]);
    bool allMatch = true;
    for (const LoopOrder& order : orders) {
        std::set<std::vector<long long>> points = enumerateVisitedPoints(nest, order);
        bool ok = (points == baseline);
        allMatch = allMatch && ok;
        printf("  order %-16s -> visits %zu points, matches baseline order (%s)\n",
               loopOrderStr(nest, order).c_str(), points.size(), ok ? "confirmed" : "MISMATCH");
    }
    printf("\nself-check: all %zu loop orders visit the exact same %lld (dim0,dim1) index pairs -- for\n",
           orders.size(), nest.totalIterations());
    printf("an ELEMENTWISE nest, loop order is a pure performance knob (cache locality), never a\n");
    printf("correctness concern, because every iteration writes a distinct output independent of every\n");
    printf("other iteration (%s)\n\n", allMatch ? "confirmed" : "MISMATCH");

    // ---- Part 2: a REDUCTION -- why summation order is a genuinely different kind of knob ----
    printf("--- Why a reduction's own summation ORDER is not just a performance knob ---\n\n");
    printf("float32 has a 24-bit mantissa: every integer up to 2^24 = 16,777,216 is exactly\n");
    printf("representable, but past that point the gap between representable values (the ULP)\n");
    printf("exceeds 1.0 -- so adding 1.0f to a value already at 2^24 can do NOTHING AT ALL.\n\n");

    std::vector<float> data;
    data.push_back(16777216.0f);
    for (int i = 0; i < 16; ++i) data.push_back(1.0f);

    float forwardSum = 0.0f;
    for (float v : data) forwardSum += v;  // big value FIRST, sixteen +1.0f increments added on top
    float reverseSum = 0.0f;
    for (auto it = data.rbegin(); it != data.rend(); ++it) reverseSum += *it;  // ones summed FIRST, big value added once
    double trueSum = 16777216.0 + 16.0;

    printf("same 17 real float32 values, two different (both perfectly valid) summation orders:\n\n");
    printf("  forward order (big value first, then sixteen +1.0f):  %.6f\n", forwardSum);
    printf("  reverse order (sixteen 1.0f summed first, big value added once): %.6f\n", reverseSum);
    printf("  true mathematical sum:                                 %.6f\n\n", trueSum);

    bool forwardLostThem = (std::fabs(static_cast<double>(forwardSum) - 16777216.0) < 1e-6);
    bool reverseExact = (std::fabs(static_cast<double>(reverseSum) - trueSum) < 1e-6);
    printf("self-check: forward order loses ALL sixteen +1.0f increments (result == the big value alone,\n");
    printf("unchanged) while reverse order recovers the exact mathematical answer (%s) -- neither\n",
           (forwardLostThem && reverseExact) ? "confirmed" : "MISMATCH");
    printf("program has a bug; float32 addition is genuinely not associative, and this is exactly why\n");
    printf("every array comparison since Chapter 17 (evaluateArrays() vs. generated code) has used a\n");
    printf("TOLERANCE (1e-3f, 1e-2f) instead of exact equality -- not a hedge, a direct consequence of\n");
    printf("this real, measurable fact. A reduction's own summation order -- partial sums, a tree\n");
    printf("reduction, Chapter 19's own lane-parallel vector accumulator -- is a real schedule choice a\n");
    printf("later autotuner may want to search over, but it is the ONE schedule dimension in this\n");
    printf("chapter that can change the actual floating-point ANSWER, not just how fast it arrives.\n");

    bool allOk = allMatch && forwardLostThem && reverseExact;
    return allOk ? 0 : 1;
}
