# 21. The Search Space: Tile Sizes, Loop Orders, and Unrolling

**What you will understand:** `enumerateTileSizeCandidates()`, a heuristic candidate list -- powers of two plus the loop's own full extent -- that trades search COMPLETENESS for TRACTABILITY, the same heuristic real autotuning systems actually use; `LoopOrder` and `enumerateLoopOrders()`, every permutation of a `LoopNest`'s own loops, and a real proof, by direct enumeration rather than assertion, that an ELEMENTWISE loop nest computes the exact same answer under every one of them; a concrete, measured fact -- float32 addition is not associative -- demonstrated with 17 real numbers where two equally valid summation orders differ by exactly 16.0, and why that makes a REDUCTION's own summation order the one schedule dimension in this chapter that can change the floating-point ANSWER, not just its speed; `unrollFactorCandidates()` and a real unrolled loop, the software analogue of Chapter 19's own SIMD width -- the exact same "main loop plus scalar tail" shape, this time a free software choice instead of a fixed hardware width; and `Schedule`, which ties tile size, loop order, and unroll factor into one CARTESIAN-PRODUCT search space, enumerated for real over a concrete `LoopNest`, closing the chapter on the combinatorial-explosion problem Chapters 22 and 23 both exist to solve.

**What you need to know first:** Chapter 15's own `Loop`/`LoopNest`/`TiledLoop`/`tileLoop()`/`actualInnerExtent()` (reused verbatim throughout this chapter); Chapter 6's own `broadcastShapes()`/`inferShapes()` (the reason a real `LoopNest` can have more than one loop in the first place); the array-comparison tolerance (`1e-3f`, `1e-2f`) every chapter since 17 has used when checking generated code against `evaluateArrays()` (explained concretely, for the first time, in Section 21.2); and Chapter 19's own vector-width "main loop plus scalar tail" shape (reused and generalized in Section 21.3).

---

Every backend through Part 4 -- Chapter 17's scalar loops, Chapter 18's CUDA kernels, Chapter 19's vectorized code, Chapter 20's JIT-loaded functions -- compiled exactly ONE fixed schedule per fused kernel: one tile size (or none at all), one loop order, no unrolling decisions ever revisited. That was never a limitation those chapters needed to apologize for -- Parts 1 through 4 were about a genuinely prior question, whether this book's own generated code computes the CORRECT answer at all, and every one of those chapters answered it for real, compiled and run on real hardware. Part 5 asks a different question: given that a fused kernel's own arithmetic is already proven correct, there are usually SEVERAL different, equally correct ways to execute it -- different tile sizes, different loop nesting orders, different unroll factors -- and some of them run measurably faster than others on real hardware. This chapter builds and validates the SEARCH SPACE itself: which schedules are even legal, and how large that space really is. It deliberately does not yet generate scheduled code (Chapter 22 needs a cost model first, precisely so a later chapter doesn't have to compile and measure every single candidate to find a good one) or run an actual autotuning search (Chapter 23). Every worked example in this chapter still compiles and runs for real, computing genuine numbers about the search space itself -- candidate counts, index-set equality proofs, real float32 sums -- never simulated in a comment.

The chapter's own three schedule dimensions interact in three different ways, and each section is built around exactly one of them. TILE SIZE (Section 21.1) is a per-loop decision: a `LoopNest` with two loops has two independent tile-size choices, one per loop, reusing Chapter 15's own `tileLoop()`/`actualInnerExtent()` completely unchanged -- this section's only new idea is which tile sizes are even worth trying. LOOP ORDER (Section 21.2) is a single decision for the WHOLE nest at once: given N loops, there are N! ways to nest them, and this section proves a real, useful fact about when that choice matters for CORRECTNESS (an elementwise nest: never) versus when it can shift the actual floating-point ANSWER (a reduction: genuinely, and this section measures it). UNROLL FACTOR (Section 21.3) is once again a per-loop decision, most naturally applied to whichever loop ends up innermost after tiling and ordering are both fixed -- and the section that adds it also closes the chapter by combining all three into one `Schedule` and enumerating the real cartesian product for a concrete `LoopNest`.

```text
Three schedule dimensions, three different shapes of decision:

  TILE SIZE (21.1)     -- one choice PER LOOP, reusing Chapter 15's
                           tileLoop()/actualInnerExtent() unchanged;
                           this chapter's only new idea is WHICH sizes
                           are worth trying (powers of two + the full
                           extent, not every integer 1..extent)

  LOOP ORDER (21.2)     -- one choice for the WHOLE nest: which of the
                           N! permutations of N loops to nest in;
                           proven a pure performance knob for an
                           elementwise nest (every order visits the
                           same index set), but NOT purely cosmetic
                           for a reduction (float32 addition is not
                           associative -- measured, not assumed)

  UNROLL FACTOR (21.3)  -- one more per-loop choice, the software
                           analogue of Chapter 19's own SIMD width:
                           process W scalar iterations per loop trip
                           instead of one hardware instruction over W
                           lanes -- same "main loop + scalar tail"
                           shape, now a free choice instead of a fixed
                           hardware constant

Combined into ONE Schedule (21.3's own capstone), even a TINY 2-loop,
48-iteration LoopNest already has 128 legal combinations -- and this
book's own candidate lists are already a REDUCED heuristic, not the
naive full search. The space only grows from here: this chapter builds
and validates it; Chapter 22 builds a cost model to rank candidates
without measuring every one; Chapter 23 builds the actual search.
```

## 21.1 One Loop's Own Search Space: Tile Size Candidates

### Intuition

Chapter 15's own `tileLoop()` could split a loop at any tile size a caller happened to pass in -- 4, 7, 100, any positive integer -- and it never asked whether that particular number was worth trying in the first place. That was the right scope for Chapter 15: it needed to prove tiling ITSELF correct, at the boundary, for whatever tile size a caller chose. This section asks the question Chapter 15 left open: for a loop of a given extent, what is the actual LIST of tile sizes an autotuner should even consider? The naive answer -- every integer from 1 to the extent -- is a real, valid search space, but it grows directly with the loop's own iteration count, and most real loops have far more than a handful of iterations. `enumerateTileSizeCandidates()` replaces that naive list with a small, standard heuristic.

### Background

The candidate list is powers of two -- 1, 2, 4, 8, 16, and so on -- up to the loop's own extent, plus the extent itself if it isn't already a power of two (meaning "one single tile," no splitting at all, a schedule choice Chapter 15 never needed a name for since it only ever tiled with a fixed caller-chosen size). This is not an arbitrary shortcut: powers of two align naturally with cache-line sizes, with the SIMD vector widths Chapter 19 already made concrete (AVX2's own 8 lanes, NEON's own 4, both powers of two), and with the thread-block sizes real GPU programming conventionally uses -- the same real reasons a production autotuner restricts its own tile-size search the same way, rather than trying every integer between 1 and the extent.

The section's own correctness proof extends Chapter 15's own discipline rather than replacing it: Chapter 15 proved `tileLoop()` correct for three hand-picked tile sizes, one at a time. This section proves something slightly stronger -- that EVERY candidate `enumerateTileSizeCandidates()` proposes is itself a correct tiling -- by running Chapter 15's own set-equality check (the tiled index set must exactly equal the untiled index set: no index missing, none visited twice, nothing read past the end) across the WHOLE candidate list in one loop, for two different loop extents: 13 (deliberately not a power of two, forcing the "append the extent itself" branch to fire, since 8 is the largest power of two below it) and 16 (exactly a power of two, where the candidate list needs no separate append at all). Both extents produce 5 candidates each, all individually confirmed correct, against a naive exhaustive search that would have needed 13 candidates for the first loop alone.

```text
enumerateTileSizeCandidates(extent): powers of two, capped, plus the
extent itself if it isn't already one of them

  extent=13:  1, 2, 4, 8   (doubling stops -- 16 > 13)
              + 13 appended (not already in the list)
              = 5 candidates: {1, 2, 4, 8, 13}

  extent=16:  1, 2, 4, 8, 16   (doubling lands exactly on 16)
              (16 already in the list -- nothing appended)
              = 5 candidates: {1, 2, 4, 8, 16}

Every one of those candidates is proven correct by Chapter 15's own
discipline, extended from "one hand-picked value" to "the WHOLE list":

  for each candidate tileSize:
    tiled = tileLoop(loop, tileSize)
    enumerateTiled(tiled) == enumerateUntiled(loop)  ?  MUST be true

  extent=13, naive exhaustive search: 13 candidates (every integer)
  extent=13, this section's own heuristic:  5 candidates -- a real,
             checkable reduction, not a guess -- and all 5 individually
             confirmed correct above, not merely assumed
```


```cpp
// Chapter 21: The Search Space: Tile Sizes, Loop Orders, and Unrolling
// 051_tile_size_candidates_enumerating_one_loops_own_search_space.cpp
//
// Section 21.1 -- every backend through Part 4 (Chapters 17-20) compiled
// exactly ONE fixed schedule per fused kernel: Chapter 15's own tileLoop()
// could split a loop at any tile size a caller happened to pass in, but
// nothing in this book has ever asked WHICH tile sizes are worth trying, or
// how many there even are to choose from. This section builds the first
// piece of a real search space: enumerateTileSizeCandidates(), a function
// that turns "tile size" from a single number a caller picks by hand into
// a genuine LIST of candidates a later chapter's own autotuner (Chapter 23)
// will eventually search over.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 051_tile_size_candidates_enumerating_one_loops_own_search_space.cpp -o 051_driver
// Run:     ./051_driver
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>

// ==================== Loop / LoopNest / TiledLoop / tileLoop() / actualInnerExtent() (from Chapter 15, unchanged) ====================

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

// ==================== Section 21.1: enumerateTileSizeCandidates() ====================
//
// The naive search space for ONE loop's own tile size is every integer from
// 1 to that loop's own extent -- a real, valid choice at every single value,
// but O(extent) candidates for a loop that might have thousands of
// iterations. This function trades COMPLETENESS for TRACTABILITY, using the
// same heuristic real autotuning systems actually use: powers of two, plus
// the loop's own full extent (meaning "one single tile" -- no splitting at
// all, a schedule choice this book's own Chapter 15 never had a name for).
// Powers of two are not an arbitrary choice -- they align naturally with
// cache-line sizes, SIMD vector widths (Chapter 19's own AVX2 width of 8 and
// NEON width of 4 are both powers of two), and thread-block sizes real GPU
// programming conventionally uses, all real reasons a production autotuner
// restricts its own search the same way rather than trying every integer.
static std::vector<long long> enumerateTileSizeCandidates(long long extent) {
    if (extent <= 0) throw std::runtime_error("enumerateTileSizeCandidates: extent must be positive");
    std::vector<long long> candidates;
    for (long long t = 1; t <= extent; t *= 2) candidates.push_back(t);
    if (candidates.empty() || candidates.back() != extent) candidates.push_back(extent);
    return candidates;
}

// ==================== Correctness proof, extended from Chapter 15: every candidate, not just one ====================
//
// Chapter 15 proved tileLoop() correct for THREE hand-picked tile sizes,
// one at a time. This section proves something slightly stronger: EVERY
// candidate enumerateTileSizeCandidates() proposes is itself a correct
// tiling, by running Chapter 15's own set-equality proof across the WHOLE
// candidate list in one loop, not spot-checking a single value.
static std::set<long long> enumerateUntiled(const Loop& loop) {
    std::set<long long> visited;
    for (long long i = 0; i < loop.extent; ++i) visited.insert(i);
    return visited;
}
static std::set<long long> enumerateTiled(const TiledLoop& tl) {
    std::set<long long> visited;
    for (long long outer = 0; outer < tl.outerExtent; ++outer) {
        long long innerExtent = actualInnerExtent(tl, outer);
        for (long long inner = 0; inner < innerExtent; ++inner) visited.insert(outer * tl.tileSize + inner);
    }
    return visited;
}

static bool checkAllCandidates(const Loop& loop) {
    std::vector<long long> candidates = enumerateTileSizeCandidates(loop.extent);
    std::set<long long> untiled = enumerateUntiled(loop);
    printf("Loop %s: extent=%lld -> %zu tile-size candidates: ", loop.dimName.c_str(), loop.extent, candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) printf("%s%lld", i ? ", " : "", candidates[i]);
    printf("\n");
    bool allOk = true;
    for (long long c : candidates) {
        TiledLoop tl = tileLoop(loop, c);
        std::set<long long> tiled = enumerateTiled(tl);
        bool ok = (tiled == untiled);
        allOk = allOk && ok;
        printf("  tileSize=%-3lld -> outerExtent=%-3lld  index set matches untiled (%s)\n", c, tl.outerExtent,
               ok ? "confirmed" : "MISMATCH");
    }
    return allOk;
}

int main() {
    printf("=== Section 21.1: enumerateTileSizeCandidates() -- one loop's own search space ===\n\n");

    // Case 1: extent=13, deliberately not a power of two -- forces the
    // "append the extent itself" branch to fire (8 is the largest power of
    // two <= 13, so 13 would otherwise never appear as its own candidate).
    bool ok1 = checkAllCandidates(Loop{"i", 13});

    printf("\n");

    // Case 2: extent=16, EXACTLY a power of two -- the candidate list ends
    // up being {1,2,4,8,16} with no separate "append the extent" needed,
    // since 16 already appears as a power of two in its own right.
    bool ok2 = checkAllCandidates(Loop{"j", 16});

    printf("\n=== What this buys, and what it costs ===\n\n");
    long long extent = 13;
    std::vector<long long> candidates = enumerateTileSizeCandidates(extent);
    printf("naive exhaustive search (every tile size 1..extent): %lld candidates\n", extent);
    printf("this section's own heuristic (powers of two + the extent itself): %zu candidates\n", candidates.size());
    printf("self-check: %zu candidates is a real, checkable reduction from %lld -- and every single one of\n",
           candidates.size(), extent);
    printf("those %zu candidates was just individually proven correct above, not merely assumed (%s)\n",
           candidates.size(), (ok1 && ok2) ? "confirmed" : "MISMATCH");

    bool allOk = ok1 && ok2;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "051_tile_size_candidates_enumerating_one_loops_own_search_space.cpp" -o "051_tile_size_candidates_enumerating_one_loops_own_search_space"
./"051_tile_size_candidates_enumerating_one_loops_own_search_space"
```

**Output (cloud sandbox, x86-64):**

```text
=== Section 21.1: enumerateTileSizeCandidates() -- one loop's own search space ===

Loop i: extent=13 -> 5 tile-size candidates: 1, 2, 4, 8, 13
  tileSize=1   -> outerExtent=13   index set matches untiled (confirmed)
  tileSize=2   -> outerExtent=7    index set matches untiled (confirmed)
  tileSize=4   -> outerExtent=4    index set matches untiled (confirmed)
  tileSize=8   -> outerExtent=2    index set matches untiled (confirmed)
  tileSize=13  -> outerExtent=1    index set matches untiled (confirmed)

Loop j: extent=16 -> 5 tile-size candidates: 1, 2, 4, 8, 16
  tileSize=1   -> outerExtent=16   index set matches untiled (confirmed)
  tileSize=2   -> outerExtent=8    index set matches untiled (confirmed)
  tileSize=4   -> outerExtent=4    index set matches untiled (confirmed)
  tileSize=8   -> outerExtent=2    index set matches untiled (confirmed)
  tileSize=16  -> outerExtent=1    index set matches untiled (confirmed)

=== What this buys, and what it costs ===

naive exhaustive search (every tile size 1..extent): 13 candidates
this section's own heuristic (powers of two + the extent itself): 5 candidates
self-check: 5 candidates is a real, checkable reduction from 13 -- and every single one of
those 5 candidates was just individually proven correct above, not merely assumed (confirmed)
```

**Output (device, aarch64 Linux VM):**

```text
=== Section 21.1: enumerateTileSizeCandidates() -- one loop's own search space ===

Loop i: extent=13 -> 5 tile-size candidates: 1, 2, 4, 8, 13
  tileSize=1   -> outerExtent=13   index set matches untiled (confirmed)
  tileSize=2   -> outerExtent=7    index set matches untiled (confirmed)
  tileSize=4   -> outerExtent=4    index set matches untiled (confirmed)
  tileSize=8   -> outerExtent=2    index set matches untiled (confirmed)
  tileSize=13  -> outerExtent=1    index set matches untiled (confirmed)

Loop j: extent=16 -> 5 tile-size candidates: 1, 2, 4, 8, 16
  tileSize=1   -> outerExtent=16   index set matches untiled (confirmed)
  tileSize=2   -> outerExtent=8    index set matches untiled (confirmed)
  tileSize=4   -> outerExtent=4    index set matches untiled (confirmed)
  tileSize=8   -> outerExtent=2    index set matches untiled (confirmed)
  tileSize=16  -> outerExtent=1    index set matches untiled (confirmed)

=== What this buys, and what it costs ===

naive exhaustive search (every tile size 1..extent): 13 candidates
this section's own heuristic (powers of two + the extent itself): 5 candidates
self-check: 5 candidates is a real, checkable reduction from 13 -- and every single one of
those 5 candidates was just individually proven correct above, not merely assumed (confirmed)
```


!!! note "Why the extent itself is always a candidate"
    A tile size equal to the loop's own full extent means `tileLoop()` produces exactly one tile, covering the whole loop in a single pass -- the schedule this book's own Chapters 17 through 20 have implicitly used every single time, since none of them ever called `tileLoop()` at all. Leaving that choice out of the candidate list would silently exclude the one schedule this book has already proven correct and fast enough to write an entire four-chapter Part around; `enumerateTileSizeCandidates()` always appends it explicitly rather than relying on it to already be a power of two.

## 21.2 Loop Order: What It Changes, and What It Doesn't

### Intuition

Section 21.1 built the search space for ONE loop's own tile size. A real `LoopNest` usually has more than one loop -- Chapter 6's own `broadcastShapes()` can produce a multi-dimensional output shape, and Chapter 17's own `buildLoopNest()` turns every dimension of that shape into its own loop -- and a second, genuinely independent schedule dimension asks a different question: given several loops, in what ORDER should they be nested? This section builds `enumerateLoopOrders()`, every permutation of a `LoopNest`'s own loops, and then proves something that sounds obvious but is not always true: for an ELEMENTWISE loop nest, every order computes the exact same answer, because each iteration is independent of every other. For a REDUCTION, that turns out not to be quite true -- not because any order is WRONG, but because float32 addition is not associative, and this section proves that concretely, with real numbers, rather than asserting it.

### Background

A `LoopOrder` is a permutation of a `LoopNest`'s own loop INDICES, listed outermost-first: `{0, 1}` means loop 0 nests outside loop 1; `{1, 0}` means the reverse. `enumerateLoopOrders()` returns every one of the N! permutations for an N-loop nest using `std::next_permutation`, the same small standard-library building block Section 21.1 already reached for instead of hand-rolling a combinatorics routine. `enumerateVisitedPoints()` extends Chapter 15's own proof-by-enumeration discipline from tile BOUNDARIES to loop ORDER: it walks a `LoopNest` in a GIVEN order and records the full coordinate tuple for every iteration, always in the nest's own natural dimension order regardless of which order the loops were actually nested in -- so two different loop orders can be compared directly by set equality. On Chapter 15's own diamond-shaped loop nest (`dim0:3, dim1:4`, the same shape Section 15.2's own Case 4 already tiled), both possible loop orders visit the exact same 12 `(dim0, dim1)` pairs: for an elementwise computation, every iteration writes a distinct output location and reads only from inputs at that SAME location, so nothing about the final result depends on the SEQUENCE in which independent iterations run -- loop order there is a pure performance knob (which access pattern is more cache-friendly), never a correctness concern.

A reduction's own single accumulator makes this genuinely different, and the section proves why with a real, measured fact rather than a hand-wave: float32 has a 24-bit mantissa, so every integer up to 2^24 = 16,777,216 is exactly representable -- but past that point, the gap between representable values (the ULP) exceeds 1.0, and adding 1.0f to a value already at 2^24 can do NOTHING AT ALL. Seventeen real float32 values -- 16,777,216.0 followed by sixteen 1.0 values -- summed in FORWARD order (the big value first, then the sixteen increments added on top of it one at a time) land back on exactly 16,777,216.0: every single increment is silently absorbed, because each individual `+1.0f` lands on a value whose own ULP is already 2.0. The exact same 17 values summed in REVERSE order (the sixteen 1.0 values added together first -- itself exact, since 16.0 is tiny and safely representable -- and only THEN added to the big value once) land on the true mathematical answer, 16,777,232.0. Neither program has a bug; both orders are perfectly valid summations of the same numbers, and they differ by exactly 16.0 -- the whole missing contribution. This is exactly why every array comparison since Chapter 17 -- `evaluateArrays()` checked against generated code, in every backend through Chapter 20 -- has used a TOLERANCE (`1e-3f`, `1e-2f`) instead of exact equality: not a hedge, but a direct, now-measured consequence of float32 addition genuinely not being associative. A reduction's own summation order -- partial sums, a tree reduction, Chapter 19's own lane-parallel vector accumulator -- remains a real schedule choice a later autotuner may want to search over, but it is the one schedule dimension in this whole chapter that can change the actual floating-point ANSWER, not merely how fast it arrives.

```text
enumerateLoopOrders(): every permutation, outermost-first

  LoopNest [dim0:3, dim1:4]  ->  2 loop orders (2! = 2):
    {0,1}: dim0 -> dim1        {1,0}: dim1 -> dim0

  Both visit the SAME 12 (dim0,dim1) pairs -- proven by recording the
  full coordinate tuple for every iteration and comparing sets, the
  same discipline Chapter 15 used for tile boundaries:

    elementwise: out[i,j] depends ONLY on inputs at (i,j) -- no
    iteration reads or writes anything another iteration touches ->
    loop order is a pure PERFORMANCE knob, never correctness

Why a REDUCTION's own order is different -- float32's 24-bit mantissa:

  17 real values: 16777216.0, then sixteen 1.0's

  FORWARD (big value first):
    16777216.0 + 1.0 + 1.0 + ... (16 times)
    each +1.0 lands on a value whose own ULP is already 2.0 -- EVERY
    increment is silently absorbed, result = 16777216.0 (unchanged!)

  REVERSE (small values first):
    1.0 + 1.0 + ... (16 times, exact) = 16.0, THEN + 16777216.0 once
    result = 16777232.0 (the true mathematical answer)

  diff = 16.0 -- the WHOLE missing contribution, not a rounding
  smudge -- and the reason every evaluateArrays()-vs-generated-code
  check since Chapter 17 has used a TOLERANCE, not exact equality
```

```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "052_loop_order_and_why_a_reductions_own_order_is_not_just_performance.cpp" -o "052_loop_order_and_why_a_reductions_own_order_is_not_just_performance"
./"052_loop_order_and_why_a_reductions_own_order_is_not_just_performance"
```

**Output (cloud sandbox, x86-64):**

```text
=== Section 21.2: enumerateLoopOrders() -- and what order does and doesn't change ===

LoopNest [dim0:3, dim1:4] (elementwise, 12 total iterations): 2 loop orders

  order dim0 -> dim1     -> visits 12 points, matches baseline order (confirmed)
  order dim1 -> dim0     -> visits 12 points, matches baseline order (confirmed)

self-check: all 2 loop orders visit the exact same 12 (dim0,dim1) index pairs -- for
an ELEMENTWISE nest, loop order is a pure performance knob (cache locality), never a
correctness concern, because every iteration writes a distinct output independent of every
other iteration (confirmed)

--- Why a reduction's own summation ORDER is not just a performance knob ---

float32 has a 24-bit mantissa: every integer up to 2^24 = 16,777,216 is exactly
representable, but past that point the gap between representable values (the ULP)
exceeds 1.0 -- so adding 1.0f to a value already at 2^24 can do NOTHING AT ALL.

same 17 real float32 values, two different (both perfectly valid) summation orders:

  forward order (big value first, then sixteen +1.0f):  16777216.000000
  reverse order (sixteen 1.0f summed first, big value added once): 16777232.000000
  true mathematical sum:                                 16777232.000000

self-check: forward order loses ALL sixteen +1.0f increments (result == the big value alone,
unchanged) while reverse order recovers the exact mathematical answer (confirmed) -- neither
program has a bug; float32 addition is genuinely not associative, and this is exactly why
every array comparison since Chapter 17 (evaluateArrays() vs. generated code) has used a
TOLERANCE (1e-3f, 1e-2f) instead of exact equality -- not a hedge, a direct consequence of
this real, measurable fact. A reduction's own summation order -- partial sums, a tree
reduction, Chapter 19's own lane-parallel vector accumulator -- is a real schedule choice a
later autotuner may want to search over, but it is the ONE schedule dimension in this
chapter that can change the actual floating-point ANSWER, not just how fast it arrives.
```

**Output (device, aarch64 Linux VM):**

```text
=== Section 21.2: enumerateLoopOrders() -- and what order does and doesn't change ===

LoopNest [dim0:3, dim1:4] (elementwise, 12 total iterations): 2 loop orders

  order dim0 -> dim1     -> visits 12 points, matches baseline order (confirmed)
  order dim1 -> dim0     -> visits 12 points, matches baseline order (confirmed)

self-check: all 2 loop orders visit the exact same 12 (dim0,dim1) index pairs -- for
an ELEMENTWISE nest, loop order is a pure performance knob (cache locality), never a
correctness concern, because every iteration writes a distinct output independent of every
other iteration (confirmed)

--- Why a reduction's own summation ORDER is not just a performance knob ---

float32 has a 24-bit mantissa: every integer up to 2^24 = 16,777,216 is exactly
representable, but past that point the gap between representable values (the ULP)
exceeds 1.0 -- so adding 1.0f to a value already at 2^24 can do NOTHING AT ALL.

same 17 real float32 values, two different (both perfectly valid) summation orders:

  forward order (big value first, then sixteen +1.0f):  16777216.000000
  reverse order (sixteen 1.0f summed first, big value added once): 16777232.000000
  true mathematical sum:                                 16777232.000000

self-check: forward order loses ALL sixteen +1.0f increments (result == the big value alone,
unchanged) while reverse order recovers the exact mathematical answer (confirmed) -- neither
program has a bug; float32 addition is genuinely not associative, and this is exactly why
every array comparison since Chapter 17 (evaluateArrays() vs. generated code) has used a
TOLERANCE (1e-3f, 1e-2f) instead of exact equality -- not a hedge, a direct consequence of
this real, measurable fact. A reduction's own summation order -- partial sums, a tree
reduction, Chapter 19's own lane-parallel vector accumulator -- is a real schedule choice a
later autotuner may want to search over, but it is the ONE schedule dimension in this
chapter that can change the actual floating-point ANSWER, not just how fast it arrives.
```


!!! warning "[COMMON TRAP] treating loop order as purely cosmetic"
    It is tempting to assume loop order only ever affects cache locality -- true for every elementwise kernel this book has generated since Chapter 13, and true for most of the code in this chapter's own capstone. But the moment a loop nest includes a REDUCTION accumulator, "just try every order, they all give the same answer" stops being true in the strict, bit-for-bit sense: this section measured a real 16.0 difference between two valid summation orders of the same 17 numbers. A later cost model (Chapter 22) or autotuner (Chapter 23) that treats loop order as free to reorder for reductions needs to account for this -- not because either order is wrong, but because they are not IDENTICAL, and this book's own tolerance-based comparisons exist because of exactly this fact.

## 21.3 Unrolling, and the Full Schedule Search Space

### Intuition

Sections 21.1 and 21.2 each built one schedule dimension in isolation: tile size per loop, and one loop order for the whole nest. This section adds the third and final dimension -- UNROLL FACTOR -- and then combines all three into one `Schedule`, enumerated for real over a concrete `LoopNest`, closing the chapter on exactly the problem Chapters 22 and 23 exist to solve. Unrolling is the software analogue of Chapter 19's own SIMD width: instead of one hardware instruction covering W lanes simultaneously, an unrolled loop processes W scalar iterations per trip, advancing the loop index by W each time, with a scalar tail for whatever remains when the extent doesn't divide evenly -- the exact same "main loop plus scalar tail" shape Section 19.1 already established for real vector hardware, this time as a free software choice instead of a fixed hardware constant.

### Background

`unrollFactorCandidates()` reuses Section 21.1's own doubling-plus-extent candidate list unchanged, and that reuse is deliberate, not lazy: tile size and unroll factor are, underneath, the same underlying question -- how many elements does one loop trip cover -- asked for two different reasons (memory blocking versus reducing per-iteration loop overhead), so it would be a real inconsistency for their own candidate-generation logic to differ. `computeUnrolled()` makes the technique concrete on a real elementwise computation (`out[i] = a[i]*2+1`, extent 13, deliberately not evenly divisible by any of the unroll factors tried): a main loop processes `unrollFactor` elements per trip, a scalar tail handles whatever's left. Three different factors -- 2, 4, 8 -- each produce a DIFFERENT number of full trips and a DIFFERENT tail length (6 trips of 2 plus a 1-element tail; 3 trips of 4 plus a 1-element tail; 1 trip of 8 plus a 5-element tail), and every one of them is checked BIT-FOR-BIT identical to the un-unrolled (`unrollFactor=1`) baseline -- a direct, informative contrast with Section 21.2's own reduction-order finding: unrolling an ELEMENTWISE loop never changes so much as one bit of the result, because every iteration is independent, so processing several of them per trip instead of one at a time is a pure performance knob, exactly like loop order was for the elementwise case and exactly UNLIKE loop order was for a reduction.

`Schedule` ties together one choice from each of this chapter's own three dimensions: a tile size PER LOOP (Section 21.1 already showed the candidate list differs per loop, since it depends on that loop's own extent), one loop order for the whole nest (Section 21.2), and one unroll factor applied to the innermost loop once tiling and ordering are both fixed (this section). `enumerateSchedules()` builds the full CARTESIAN PRODUCT with a plain odometer -- a counter whose own digits each have a different base, one per loop's own tile-size candidate count -- the most direct, literal way to enumerate a cartesian product, needing no combinatorics library. Run on a concrete two-loop `LoopNest` (`dim0:6, dim1:8`, 48 total iterations): `dim0` has 4 tile-size candidates, `dim1` has 4, there are 2 loop orders, and 4 unroll-factor candidates apply to the innermost loop (`dim1`, extent 8) -- a cartesian product of 4 x 4 x 2 x 4 = 128 total schedules, confirmed to match `enumerateSchedules()`'s own actual output exactly, not merely predicted. For a LOOP NEST THIS SMALL. Real fused kernels in this book have more loops (any multi-dimensional shape, back to Chapter 6), larger extents, and this chapter's own candidate lists are already a REDUCED heuristic, not the naive full search -- the space only grows from here. Brute-force MEASURING every one of them is not how Chapter 23's own autotuner will actually work; Chapter 22 builds a cost model specifically so candidates can be ranked without compiling and running every single one.

```text
unrollFactorCandidates(): the SAME candidate shape as 21.1's own
tile-size list -- tile size and unroll factor are the same underlying
question (how many elements per loop trip), asked for two different
reasons

Main loop + scalar tail, out[i]=a[i]*2+1, extent=13 (not evenly
divisible by 2, 4, or 8):

  unrollFactor=2: 6 full trips of 2  + 1-element tail  (6*2+1=13)
  unrollFactor=4: 3 full trips of 4  + 1-element tail  (3*4+1=13)
  unrollFactor=8: 1 full trip  of 8  + 5-element tail  (1*8+5=13)

  every one BIT-FOR-BIT identical to the un-unrolled (factor=1) result
  -- elementwise unrolling, like elementwise loop order, is a PURE
  performance knob, never a correctness concern

The capstone: Schedule = {tileSizePerLoop, loopOrder, unrollFactor},
enumerated as a full cartesian product over ONE real LoopNest:

  LoopNest [dim0:6, dim1:8]  (48 total iterations)

    dim0 tile-size candidates: {1,2,4,6}         -> 4 choices
    dim1 tile-size candidates: {1,2,4,8}         -> 4 choices
    loop orders: dim0->dim1, dim1->dim0          -> 2 choices
    unroll-factor candidates (innermost=dim1):
      {1,2,4,8}                                  -> 4 choices

    4 x 4 x 2 x 4 = 128 total legal schedules -- for a TINY 2-loop,
    48-iteration kernel. This chapter builds and validates that space;
    Chapter 22 builds a cost model to rank it without measuring every
    candidate; Chapter 23 builds the actual search.
```

```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "053_unrolling_and_the_full_schedule_search_space_this_chapter_built_toward.cpp" -o "053_unrolling_and_the_full_schedule_search_space_this_chapter_built_toward"
./"053_unrolling_and_the_full_schedule_search_space_this_chapter_built_toward"
```

**Output (cloud sandbox, x86-64):**

```text
=== Section 21.3: unrolling, and the full schedule search space this chapter built toward ===

--- Unrolling a real computation: out[i] = a[i]*2+1, extent=13 (not evenly divisible) ---

  unrollFactor=2  -> 6 full trip(s) of 2 + a 1-element scalar tail, bit-for-bit
                 identical to the un-unrolled result (confirmed)
  unrollFactor=4  -> 3 full trip(s) of 4 + a 1-element scalar tail, bit-for-bit
                 identical to the un-unrolled result (confirmed)
  unrollFactor=8  -> 1 full trip(s) of 8 + a 5-element scalar tail, bit-for-bit
                 identical to the un-unrolled result (confirmed)

self-check: unlike Section 21.2's own reduction-order finding, unrolling an ELEMENTWISE loop
never changes so much as one bit of the result -- every iteration is independent, so
processing them W at a time instead of one at a time is a PURE performance knob (confirmed)

--- The capstone: tile size x loop order x unroll factor, ONE real LoopNest ---

LoopNest [dim0:6, dim1:8] (48 total iterations):
  dim0 tile-size candidates: 4
  dim1 tile-size candidates: 4
  loop orders:               2
  unroll-factor candidates:  4 (applied to the innermost loop, dim1, extent 8)

cartesian product: 4 x 4 x 2 x 4 = 128 total schedules
enumerateSchedules() actually produced: 128 schedules (confirmed)

a few real schedules from that list:
  [0] tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [64] tiles=[dim0:1,dim1:4] order=dim0>dim1 unroll=1
  [127] tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

self-check: 128 schedules for a TINY 2-loop, 48-iteration nest -- a real fused kernel in
this book has more loops (multi-dim shapes, Chapter 6), larger extents, and this book's own
candidate lists are already a REDUCED heuristic (Section 21.1), not the naive full search --
the space only grows from here. Brute-force MEASURING every one is not how Chapter 23's own
autotuner will work; Chapter 22 builds a cost model specifically to rank candidates without
compiling and running every single one of them.
```

**Output (device, aarch64 Linux VM):**

```text
=== Section 21.3: unrolling, and the full schedule search space this chapter built toward ===

--- Unrolling a real computation: out[i] = a[i]*2+1, extent=13 (not evenly divisible) ---

  unrollFactor=2  -> 6 full trip(s) of 2 + a 1-element scalar tail, bit-for-bit
                 identical to the un-unrolled result (confirmed)
  unrollFactor=4  -> 3 full trip(s) of 4 + a 1-element scalar tail, bit-for-bit
                 identical to the un-unrolled result (confirmed)
  unrollFactor=8  -> 1 full trip(s) of 8 + a 5-element scalar tail, bit-for-bit
                 identical to the un-unrolled result (confirmed)

self-check: unlike Section 21.2's own reduction-order finding, unrolling an ELEMENTWISE loop
never changes so much as one bit of the result -- every iteration is independent, so
processing them W at a time instead of one at a time is a PURE performance knob (confirmed)

--- The capstone: tile size x loop order x unroll factor, ONE real LoopNest ---

LoopNest [dim0:6, dim1:8] (48 total iterations):
  dim0 tile-size candidates: 4
  dim1 tile-size candidates: 4
  loop orders:               2
  unroll-factor candidates:  4 (applied to the innermost loop, dim1, extent 8)

cartesian product: 4 x 4 x 2 x 4 = 128 total schedules
enumerateSchedules() actually produced: 128 schedules (confirmed)

a few real schedules from that list:
  [0] tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [64] tiles=[dim0:1,dim1:4] order=dim0>dim1 unroll=1
  [127] tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

self-check: 128 schedules for a TINY 2-loop, 48-iteration nest -- a real fused kernel in
this book has more loops (multi-dim shapes, Chapter 6), larger extents, and this book's own
candidate lists are already a REDUCED heuristic (Section 21.1), not the naive full search --
the space only grows from here. Brute-force MEASURING every one is not how Chapter 23's own
autotuner will work; Chapter 22 builds a cost model specifically to rank candidates without
compiling and running every single one of them.
```


!!! note "What this chapter proved, and what it deliberately left for later"
    Every number in this chapter is real: 5 tile-size candidates individually proven correct for two different extents, 2 loop orders proven to visit identical index sets for an elementwise nest, a genuine 16.0 float32 discrepancy between two valid reduction orders, 3 unroll factors proven bit-for-bit identical to an un-unrolled baseline, and 128 real schedules enumerated for one concrete `LoopNest`. What this chapter deliberately does NOT do, named rather than hidden: it never generates scheduled CODE for any of these candidates (no tiled, reordered, or unrolled version of Chapter 17-20's own codegen exists yet), and it never RUNS or TIMES any of them against real hardware. Those are Chapter 22's job (a cost model, so candidates can be ranked without measuring every one) and Chapter 23's job (an actual search over the space this chapter just proved is real and already, even at this toy scale, too large to brute-force one measurement at a time.

## Chapter Summary

This chapter opened Part 5, Autotuning, by building and validating the SEARCH SPACE a later autotuner will need to search over -- deliberately stopping short of generating scheduled code or measuring anything on real hardware, both explicitly left for Chapters 22 and 23. Section 21.1 built `enumerateTileSizeCandidates()`, replacing Chapter 15's own single caller-chosen tile size with a real candidate LIST -- powers of two plus the loop's own full extent, the same heuristic real autotuning systems use -- and extended Chapter 15's own set-equality proof from one hand-picked value to the WHOLE candidate list, for two different loop extents. Section 21.2 built `enumerateLoopOrders()` and proved, by direct enumeration rather than assertion, that an elementwise loop nest computes the identical answer under every one of its N! possible orders -- then proved, with a real, measured 16.0 float32 discrepancy between two valid summation orders of the same 17 numbers, that a REDUCTION's own order is not purely cosmetic, and that this is exactly why every array comparison since Chapter 17 has used a tolerance instead of exact equality. Section 21.3 added unroll factor -- the software analogue of Chapter 19's own SIMD width, proven bit-for-bit safe for elementwise computation across three different factors -- and closed the chapter by combining all three dimensions into one `Schedule`, enumerating a real cartesian product of 128 legal schedules for a tiny two-loop, 48-iteration `LoopNest`, and stating plainly that this number only grows from here.

## Self-Check Questions

1. Why does `enumerateTileSizeCandidates()` use powers of two plus the loop's own extent, instead of every integer from 1 to the extent?
2. Section 21.1 proves EVERY candidate correct, not just one. What specific check does it run for each candidate, and which earlier chapter's own discipline does that check come from?
3. For an elementwise `LoopNest`, why does loop order never affect the final result, no matter which of the N! permutations is chosen?
4. What specific property of float32 explains why summing 16777216.0 and sixteen 1.0 values in forward order loses all sixteen increments?
5. What is the actual numeric difference between the forward-order and reverse-order sums in Section 21.2's own worked example, and what does that difference explain about this book's own array-comparison tolerances?
6. Why does `unrollFactorCandidates()` reuse Section 21.1's own tile-size candidate logic unchanged, rather than defining a separate heuristic?
7. Section 21.3 checks that every unrolled result is BIT-FOR-BIT identical to the un-unrolled baseline. Contrast this with Section 21.2's own reduction-order finding -- why are the two sections' own correctness guarantees different in kind?
8. For the capstone `LoopNest` (`dim0:6, dim1:8`), what are the four factors that multiply together to give 128 total schedules, and why does the book explicitly NOT try to measure all 128 of them in this chapter?

## Where We Go Next

Chapter 22, "Cost Models vs. Measurement-Based Autotuning," picks up exactly where this chapter's own closing number leaves off: 128 legal schedules for a toy two-loop kernel, a number this chapter explicitly declined to brute-force by compiling and running every single one. Chapter 22 builds a COST MODEL -- a way to estimate how good a schedule is likely to be without actually running it -- and honestly compares its own predictions against real measurement, the same "correct, not fastest, and honest about the difference" discipline this book has practiced since Chapter 18's own atomicAdd() discussion. Apply the Chapter 5-21 depth-level standard throughout.

## Worked Solutions

1. The naive search space -- every integer from 1 to the loop's own extent -- grows directly with the loop's own iteration count, and most real loops have far more than a handful of iterations. Powers of two are a real, standard heuristic: they align with cache-line sizes, with the SIMD vector widths Chapter 19 already made concrete (AVX2's 8 lanes, NEON's 4, both powers of two), and with conventional GPU thread-block sizes -- the same reasons a production autotuner restricts its own search the same way. The loop's own full extent is always appended too, since it represents "one single tile, no splitting," the schedule every backend through Chapter 20 has implicitly used without ever calling `tileLoop()` at all.
2. For each candidate tile size, the section calls Chapter 15's own `tileLoop()` to build a `TiledLoop`, then checks that the SET of indices the tiled form actually visits (via `enumerateTiled()`) exactly equals the set the original untiled loop visits (via `enumerateUntiled()`) -- no index missing, none visited twice, nothing read past the end. This is Chapter 15's own proof-by-enumeration discipline (first used to prove `tileLoop()` correct for three hand-picked tile sizes), extended here to run automatically across the whole candidate list rather than one value at a time.
3. An elementwise computation's own output at any index depends ONLY on inputs at that SAME index -- no iteration reads or writes anything another iteration touches. Since the loops are independent of each other, changing the ORDER in which they run changes only the SEQUENCE of (otherwise identical) operations, never which operations happen or what values they compute -- proven directly in this section by recording every iteration's own full coordinate tuple under two different orders and confirming the two resulting sets are identical.
4. float32 has a 24-bit mantissa, so every integer up to 2^24 = 16,777,216 is exactly representable -- but past that value, the gap between two adjacent representable float32 values (the ULP) becomes 2.0, larger than the 1.0 being added. Adding `1.0f` to a value whose own ULP already exceeds 1.0 rounds right back to the original value, unchanged -- so all sixteen `+1.0f` increments, added one at a time on top of the already-large accumulator, are each individually absorbed with no effect.
5. The forward-order sum is 16,777,216.0 (unchanged from the starting value) and the reverse-order sum is 16,777,232.0 (the true mathematical answer) -- a difference of exactly 16.0, the entire missing contribution from all sixteen increments, not a small rounding smudge. This is exactly why every array comparison since Chapter 17 (checking generated code against `evaluateArrays()`, across every backend through Chapter 20) has used a TOLERANCE (`1e-3f`, `1e-2f`) instead of requiring exact equality: real, valid summation orders of the same underlying numbers can produce genuinely different float32 results, so exact equality would be too strict a bar even for correct code.
6. Tile size and unroll factor are, underneath, the same underlying question: how many elements does one loop trip cover? Tile size asks it for memory-blocking reasons (how much data one pass should touch); unroll factor asks it for loop-overhead reasons (how many iterations one trip should cover before looping back) -- but the actual SET of reasonable candidate values to try is identical either way, so defining two separate heuristics would be a real, unmotivated inconsistency rather than two genuinely different ideas.
7. Section 21.3's own unrolling check compares an unrolled elementwise computation against its own un-unrolled baseline and requires BIT-FOR-BIT equality, which it gets, because every iteration of an elementwise loop is independent -- processing several of them per trip instead of one changes nothing about what gets computed. Section 21.2's own reduction-order finding is different in kind: it is not a bug or a broken guarantee that two summation orders differ, because float32 addition is genuinely not associative -- there is no bit-for-bit guarantee to prove there in the first place, only a bounded, honestly measured difference between two both-correct answers.
8. The four factors are: 4 tile-size candidates for `dim0` (extent 6: `{1,2,4,6}`), 4 tile-size candidates for `dim1` (extent 8: `{1,2,4,8}`), 2 loop orders (2! for a 2-loop nest), and 4 unroll-factor candidates applied to the innermost loop (`dim1`, extent 8, same candidate list as its own tile sizes). The book explicitly does not measure all 128 because this is already a TINY, toy-scale kernel -- a real fused kernel has more loops, larger extents, and this chapter's own candidate lists are already a reduced heuristic, not the naive full search -- so the number only grows from here, and Chapter 22 exists specifically to avoid needing to compile and run every single candidate to find a good one.

---

**Sources cited in this chapter:**

None new. This chapter's own tile-size and unroll-factor candidate heuristics, loop-order enumeration, and cartesian-product schedule search space are all original to this book, building on Chapter 15's own `LoopNest`/`tileLoop()` machinery and Chapter 19's own vector-width "main loop plus scalar tail" shape; the float32 non-associativity fact demonstrated in Section 21.2 is a well-known property of IEEE 754 single-precision arithmetic, verified here by this book's own directly executed and measured example rather than cited to an external source.