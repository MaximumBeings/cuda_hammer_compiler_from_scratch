// Appendix E: Profiling and Benchmarking
// 103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp
//
// Section E.1 -- Chapter 17's own File 039 built evaluateArrays() and ran it
// on the diamond graph from Chapter 6: a:[3,4], b:[4], t1=a+b, t2=t1*a,
// t3=relu(t1), out=t2+t3, comparing the ORIGINAL graph (t1 materialized as
// its own array, because it has two consumers) against the FUSED graph
// (t2, t3, out collapsed into one FusedElementwise step) -- and found they
// agree bit-for-bit at every one of 12 elements. Chapter 12 then argued, in
// prose, that fusion's real benefit is avoiding round-trips to memory for
// intermediate tensors -- "memory-bound vs. compute-bound" -- but every
// verification in this book through Chapter 22 has been either correctness
// (bit-for-bit agreement, no clock involved) or wall-clock timing (Chapter
// 19's best-of-N discipline, Chapter 22's own honest turnover-point finding
// that a wall clock alone could not explain). Neither approach can show
// what actually happens at the memory system: how many loads and stores
// each version issues, and how many of those miss cache. This file builds
// the SAME diamond graph shape as File 039 at real scale -- N=20,000,000
// elements instead of 12 -- run two ways: MATERIALIZED (every intermediate
// tensor gets its own real heap-allocated std::vector<float>, written in
// full and read back, exactly like the unfused half of File 039's own
// evaluateArrays()) and FUSED (one pass, one register-resident running
// value per output element, exactly like the fused half). Both are checked
// bit-for-bit against each other, continuing this book's own correctness
// discipline; Sections E.1 and E.2 then profile this SAME binary with real
// Valgrind tools instead of asserting which one Chapter 12's prose favors.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 -g 103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp -o 103_driver
// (Section E.1 explains, with real nm evidence, why this file needs
// __attribute__((noinline)) on both functions and NOT a drop to -O0: both
// materialized() and fused() are static and each called exactly once, so
// plain -O1 already inlines both of them into main -- confirmed by `nm`
// reporting no materialized/fused symbols at -O1 or -O2 without the
// attribute -- which would erase the per-function boundary Section E.2's
// Callgrind profiling depends on before a single profiler ever runs. Real
// -O2 code (auto-vectorized, register-allocated) is what Chapter 12's own
// fusion argument is actually about, so keeping -O2 and forcing the two
// functions to stay real, separately-callable symbols with the attribute
// is the honest fix, not trading away the optimization level this book
// verifies everywhere else.)
// Run:     ./103_driver
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

static const long long N = 20'000'000;

// ==================== MATERIALIZED: same four steps as File 039's UNFUSED graph, each its own real buffer ====================
__attribute__((noinline))
static std::vector<float> materialized(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> t1(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) t1[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] + b[static_cast<size_t>(i)];

    std::vector<float> t2(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) t2[static_cast<size_t>(i)] = t1[static_cast<size_t>(i)] * a[static_cast<size_t>(i)];

    std::vector<float> t3(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) t3[static_cast<size_t>(i)] = std::max(0.0f, t1[static_cast<size_t>(i)]);

    std::vector<float> out(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) out[static_cast<size_t>(i)] = t2[static_cast<size_t>(i)] + t3[static_cast<size_t>(i)];

    return out;
}

// ==================== FUSED: same graph, one pass, t1 held in a register, never written to memory ====================
__attribute__((noinline))
static std::vector<float> fused(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) {
        float av = a[static_cast<size_t>(i)];
        float bv = b[static_cast<size_t>(i)];
        float t1 = av + bv;
        float t2 = t1 * av;
        float t3 = std::max(0.0f, t1);
        out[static_cast<size_t>(i)] = t2 + t3;
    }
    return out;
}

int main() {
    printf("=== Appendix E.1/E.2: the same diamond graph, materialized vs. fused, N=%lld ===\n\n", N);

    std::vector<float> a(static_cast<size_t>(N)), b(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) {
        a[static_cast<size_t>(i)] = static_cast<float>((i % 1013) - 500) * 0.01f;   // genuinely non-uniform, signed
        b[static_cast<size_t>(i)] = static_cast<float>((i % 727) - 300) * 0.01f;
    }

    std::vector<float> outMaterialized = materialized(a, b);
    std::vector<float> outFused = fused(a, b);

    bool bitForBitEqual = (outMaterialized.size() == outFused.size());
    if (bitForBitEqual) {
        for (long long i = 0; i < N; ++i) {
            if (outMaterialized[static_cast<size_t>(i)] != outFused[static_cast<size_t>(i)]) { bitForBitEqual = false; break; }
        }
    }
    printf("materialized.out[0]  = %g\n", outMaterialized[0]);
    printf("fused.out[0]         = %g\n", outFused[0]);
    printf("materialized.out[%lld] = %g\n", N - 1, outMaterialized[static_cast<size_t>(N - 1)]);
    printf("fused.out[%lld]        = %g\n", N - 1, outFused[static_cast<size_t>(N - 1)]);
    printf("\nself-check: materialized and fused agree bit-for-bit at all %lld elements (%s) --\n",
           N, bitForBitEqual ? "confirmed" : "MISMATCH");
    printf("same finding as File 039's own Part B, now at real scale instead of 12 elements.\n");

    return bitForBitEqual ? 0 : 1;
}
