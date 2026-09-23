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
