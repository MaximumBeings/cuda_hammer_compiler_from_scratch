// Chapter 15: Loop Fusion and Tiling
// 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp
//
// Section 15.2 -- a LoopNest (Section 15.1) tells CUDA Hammer how many
// iterations a fused node's own generated loop needs. It says nothing yet
// about how much of a tensor's own data one PASS through that loop should
// touch at a time -- the real, practical question behind cache blocking,
// shared-memory staging, and thread-block sizing, all of which Part 4's own
// codegen chapters and Part 5's own autotuner will eventually have to
// answer for real. This section builds the piece those later chapters will
// need: TILING, splitting one loop of extent N into an OUTER loop (which
// tile) and an INNER loop (which element within that tile), given a chosen
// tile size.
//
// The one genuine subtlety, and the one this section spends most of its
// own code proving rather than asserting: tiling is only a clean, even
// split when the tile size happens to divide the original extent evenly.
// It usually doesn't. The LAST tile is then partial -- shorter than every
// tile before it -- and a tiling scheme that doesn't account for that
// either visits too FEW indices (silently dropping real tensor elements)
// or too MANY (reading or writing past the end of real memory). Getting
// this right is not a diagram-level nicety; it is the exact kind of
// off-by-one a real compiler cannot afford to get wrong.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp -o 034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary
// Run:     ./034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>

// ==================== Loop / LoopNest (from Section 15.1, unchanged) ====================

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
static std::string loopNestStr(const LoopNest& nest) {
    std::string out = "[";
    for (size_t i = 0; i < nest.loops.size(); ++i) {
        if (i) out += ", ";
        out += nest.loops[i].dimName + ":" + std::to_string(nest.loops[i].extent);
    }
    out += "]";
    return out;
}

// ==================== Section 15.2: TiledLoop / tileLoop() ====================
//
// A TiledLoop keeps FOUR numbers, not two, precisely because "the inner
// loop's own extent" is not one fixed number once the boundary tile is
// accounted for -- it is tileSize for every tile except (possibly) the
// last one. Storing originalExtent alongside tileSize lets
// actualInnerExtent() compute the true extent for ANY outer index,
// including that last, possibly-shorter tile, without ever needing a
// special-cased "if this is the last tile" branch at the call site.
struct TiledLoop {
    std::string dimName;
    long long originalExtent;
    long long tileSize;
    long long outerExtent;
};

static TiledLoop tileLoop(const Loop& loop, long long tileSize) {
    if (tileSize <= 0) throw std::runtime_error("tileLoop: tile size must be positive");
    // Ceiling division -- the number of tiles needed to cover originalExtent
    // elements at tileSize elements per tile, rounding UP: a partial final
    // tile still needs a full extra outer iteration to reach it at all.
    long long outerExtent = (loop.extent + tileSize - 1) / tileSize;
    return TiledLoop{loop.dimName, loop.extent, tileSize, outerExtent};
}

// The one function this whole section exists to get right: how many
// elements does the INNER loop actually cover for a given outer index?
// Every tile except the last covers exactly tileSize elements. The last
// tile covers whatever remains -- originalExtent minus everything the
// tiles before it already covered -- which is tileSize only when
// tileSize evenly divides originalExtent, and strictly less otherwise.
static long long actualInnerExtent(const TiledLoop& tl, long long outerIndex) {
    long long alreadyCovered = outerIndex * tl.tileSize;
    long long remaining = tl.originalExtent - alreadyCovered;
    return std::min(tl.tileSize, remaining);
}

// ==================== Correctness proof: enumerate, don't assert ====================
//
// The book's own standing discipline (every chapter since Part 2) is to
// PROVE an equivalence by actually executing both sides and comparing real
// results, not by inspecting the formulas and declaring them equivalent.
// Here, that means: the SET of indices a tiled, nested pair of loops
// actually visits must be exactly the set the single untiled loop visits --
// no index missing, no index visited twice, nothing visited past the end.
static std::set<long long> enumerateUntiled(const Loop& loop) {
    std::set<long long> visited;
    for (long long i = 0; i < loop.extent; ++i) visited.insert(i);
    return visited;
}
static std::set<long long> enumerateTiled(const TiledLoop& tl) {
    std::set<long long> visited;
    for (long long outer = 0; outer < tl.outerExtent; ++outer) {
        long long innerExtent = actualInnerExtent(tl, outer);
        for (long long inner = 0; inner < innerExtent; ++inner) {
            visited.insert(outer * tl.tileSize + inner);
        }
    }
    return visited;
}

int main() {
    printf("=== Section 15.2: tileLoop() -- splitting one loop into two, correctly at the boundary ===\n\n");

    // ---- Case 1: tile size divides the extent evenly ----
    // Hand-derivation: extent=12, tileSize=4 -> outerExtent = ceil(12/4) = 3,
    // three FULL tiles of 4 elements each (4, 4, 4), no boundary tile at all.
    {
        Loop loop{"i", 12};
        TiledLoop tiled = tileLoop(loop, 4);
        printf("Case 1 (even split): extent=%lld, tileSize=%lld -> outerExtent=%lld\n",
               loop.extent, tiled.tileSize, tiled.outerExtent);
        bool outerOk = (tiled.outerExtent == 3);
        long long e0 = actualInnerExtent(tiled, 0), e1 = actualInnerExtent(tiled, 1), e2 = actualInnerExtent(tiled, 2);
        printf("  tile sizes actually visited: %lld, %lld, %lld\n", e0, e1, e2);
        bool tileSizesOk = (e0 == 4 && e1 == 4 && e2 == 4);
        std::set<long long> untiled = enumerateUntiled(loop);
        std::set<long long> tiledSet = enumerateTiled(tiled);
        bool setsEqual = (untiled == tiledSet);
        printf("  self-check: outerExtent=3, every tile full (4,4,4), tiled/untiled index sets equal (%s)\n",
               (outerOk && tileSizesOk && setsEqual) ? "confirmed" : "MISMATCH");
        if (!(outerOk && tileSizesOk && setsEqual)) return 1;
    }

    // ---- Case 2: tile size does NOT divide the extent evenly ----
    // Hand-derivation: extent=13, tileSize=4 -> outerExtent = ceil(13/4) = 4.
    // Three full tiles (4, 4, 4) cover indices 0..11 -- 12 elements -- and
    // the FOURTH tile covers only what's left: 13 - 12 = 1 element. A
    // tiling scheme that used a fixed inner extent of 4 for every outer
    // index would read index 12, 13, 14 -- three indices past the real
    // data. actualInnerExtent() is exactly the function that prevents that.
    {
        Loop loop{"i", 13};
        TiledLoop tiled = tileLoop(loop, 4);
        printf("\nCase 2 (uneven split): extent=%lld, tileSize=%lld -> outerExtent=%lld\n",
               loop.extent, tiled.tileSize, tiled.outerExtent);
        bool outerOk = (tiled.outerExtent == 4);
        long long e0 = actualInnerExtent(tiled, 0), e1 = actualInnerExtent(tiled, 1);
        long long e2 = actualInnerExtent(tiled, 2), e3 = actualInnerExtent(tiled, 3);
        printf("  tile sizes actually visited: %lld, %lld, %lld, %lld  (last tile is PARTIAL)\n", e0, e1, e2, e3);
        bool tileSizesOk = (e0 == 4 && e1 == 4 && e2 == 4 && e3 == 1);
        std::set<long long> untiled = enumerateUntiled(loop);
        std::set<long long> tiledSet = enumerateTiled(tiled);
        bool setsEqual = (untiled == tiledSet);
        printf("  self-check: outerExtent=4, last tile is 1 element (not 4), tiled/untiled index sets\n");
        printf("  equal -- no index missing, none visited past the end (%s)\n",
               (outerOk && tileSizesOk && setsEqual) ? "confirmed" : "MISMATCH");
        if (!(outerOk && tileSizesOk && setsEqual)) return 1;
    }

    // ---- Case 3: tile size LARGER than the whole extent ----
    // Hand-derivation: extent=4, tileSize=8 -> outerExtent = ceil(4/8) = 1,
    // a single tile whose own actual extent is min(8, 4) = 4, not 8: the
    // whole loop fits inside one (partially-empty) tile.
    {
        Loop loop{"i", 4};
        TiledLoop tiled = tileLoop(loop, 8);
        printf("\nCase 3 (tile larger than extent): extent=%lld, tileSize=%lld -> outerExtent=%lld\n",
               loop.extent, tiled.tileSize, tiled.outerExtent);
        bool outerOk = (tiled.outerExtent == 1);
        long long e0 = actualInnerExtent(tiled, 0);
        printf("  tile size actually visited: %lld (not %lld)\n", e0, tiled.tileSize);
        bool tileSizeOk = (e0 == 4);
        std::set<long long> untiled = enumerateUntiled(loop);
        std::set<long long> tiledSet = enumerateTiled(tiled);
        bool setsEqual = (untiled == tiledSet);
        printf("  self-check: one tile, its own actual extent is 4 (the whole loop), sets equal (%s)\n",
               (outerOk && tileSizeOk && setsEqual) ? "confirmed" : "MISMATCH");
        if (!(outerOk && tileSizeOk && setsEqual)) return 1;
    }

    // ---- Case 4: applying tiling to a REAL, multi-dimensional LoopNest ----
    // Section 15.1's own diamond-graph loop nest was [dim0:3, dim1:4],
    // 12 total iterations. Tile dim1 (extent 4) by tileSize=2 -- an EVENLY
    // dividing choice on purpose, so this case can focus on what tiling
    // does to a loop NEST's own shape rather than repeating the boundary-
    // tile arithmetic Cases 1-3 already covered. Tiling turns a 2-loop nest
    // into a 3-loop nest (dim0, dim1_outer, dim1_inner) while visiting the
    // exact same set of (dim0, dim1) POINTS -- proven the same way, by
    // enumerating both index sets directly rather than trusting the
    // arithmetic.
    printf("\n=== Case 4: tiling one loop inside a real 2-D LoopNest ===\n\n");
    LoopNest original{{Loop{"dim0", 3}, Loop{"dim1", 4}}};
    Loop dim1 = original.loops[1];
    TiledLoop tiledDim1 = tileLoop(dim1, 2);  // 4 / 2 = 2, evenly -- outerExtent=2, tileSize=2

    LoopNest tiledNest;
    tiledNest.loops.push_back(original.loops[0]);                                    // dim0:3, untouched
    tiledNest.loops.push_back(Loop{"dim1_outer", tiledDim1.outerExtent});             // dim1_outer:2
    tiledNest.loops.push_back(Loop{"dim1_inner", tiledDim1.tileSize});                // dim1_inner:2

    printf("original loop nest: %s, %lld total iterations\n", loopNestStr(original).c_str(), original.totalIterations());
    printf("tiled loop nest:     %s, %lld total iterations\n", loopNestStr(tiledNest).c_str(), tiledNest.totalIterations());

    bool sameTotal = (original.totalIterations() == tiledNest.totalIterations());
    bool oneMoreLoop = (tiledNest.loops.size() == original.loops.size() + 1);
    printf("self-check: tiling turned a 2-loop nest into a 3-loop nest, same 12 total iterations (%s)\n",
           (sameTotal && oneMoreLoop) ? "confirmed" : "MISMATCH");

    std::set<std::pair<long long, long long>> untiledPoints;
    for (long long i = 0; i < original.loops[0].extent; ++i)
        for (long long j = 0; j < original.loops[1].extent; ++j)
            untiledPoints.insert({i, j});

    std::set<std::pair<long long, long long>> tiledPoints;
    for (long long i = 0; i < tiledNest.loops[0].extent; ++i)
        for (long long jo = 0; jo < tiledNest.loops[1].extent; ++jo)
            for (long long ji = 0; ji < tiledNest.loops[2].extent; ++ji)
                tiledPoints.insert({i, jo * tiledDim1.tileSize + ji});

    bool pointsEqual = (untiledPoints == tiledPoints);
    printf("self-check: the (dim0, dim1) index pairs visited by the tiled 3-loop form exactly match\n");
    printf("the pairs visited by the original 2-loop form -- 12 points, none missing, none repeated (%s)\n",
           pointsEqual ? "confirmed" : "MISMATCH");

    bool allOk = sameTotal && oneMoreLoop && pointsEqual;
    return allOk ? 0 : 1;
}
