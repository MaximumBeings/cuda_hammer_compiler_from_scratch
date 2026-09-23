// Chapter 12: Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels
// 024_arithmetic_intensity_of_an_unfused_vs_fused_chain.cpp
//
// Section 12.1 -- arithmetic intensity, the RATIO (not either total by
// itself) that actually predicts whether a kernel's runtime is limited
// by how fast it can compute or by how fast it can move data.
//
// Chapter 1 counted TOTAL bytes moved by an eager/unfused chain of
// length K against a single fused kernel's own total and stopped there
// (eager(N) = N * 2 * array_bytes vs. fused = 2 * array_bytes). This
// chapter divides each of those same totals by the chain's own total
// FLOP count, producing arithmetic intensity: FLOPs per byte moved.
// Chapter 1's own Worked Solution #7 named exactly this ratio as the
// missing piece needed to say anything quantitative about where a chain
// actually lands, rather than just how much traffic it moves -- this is
// that quantitative test.
//
// Every op in the chain below is treated as costing exactly 1 FLOP per
// output element -- a deliberate simplification (a real ReLU is a
// compare-and-select, not a floating add or multiply) made for the same
// reason Chapter 1 didn't need to distinguish op types for its own
// memory-traffic count: the point here is the SHAPE of the ratio
// (constant vs. growing with chain length), not any one op's exact cost.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 024_arithmetic_intensity_of_an_unfused_vs_fused_chain.cpp -o 024_arithmetic_intensity_of_an_unfused_vs_fused_chain
// Run:     ./024_arithmetic_intensity_of_an_unfused_vs_fused_chain
#include <cstdio>
#include <cmath>
#include <vector>

// One float32 element -- matching Chapter 1's own 4-byte-per-element
// assumption, so this chapter's numbers stay directly comparable to it.
static constexpr double kBytesPerElement = 4.0;

// A chain of K unary elementwise ops over N elements each
// (y1 = op(x), y2 = op(y1), ..., yK = op(y_{K-1})):
//
//   UNFUSED (K separate kernels): kernel i reads its own N-element input
//   and writes its own N-element output from/to memory -- 2*N elements
//   moved, K separate times, once per kernel launch.
//     totalFLOPs = K * N                              (the work itself)
//     totalBytes = K * 2 * N * bytesPerElement         (K round trips to memory)
//
//   FUSED (one kernel): only the chain's very first input is ever read
//   from memory and only its very last output is ever written back --
//   every intermediate y_i lives in a register for the one instruction
//   that produces AND immediately consumes it, never touching memory.
//     totalFLOPs = K * N                              (identical work)
//     totalBytes = 2 * N * bytesPerElement             (ONE round trip, regardless of K)
static double unfusedArithmeticIntensity(long long K, long long N) {
    double totalFlops = static_cast<double>(K) * static_cast<double>(N);
    double totalBytes = static_cast<double>(K) * 2.0 * static_cast<double>(N) * kBytesPerElement;
    return totalFlops / totalBytes;
}

static double fusedArithmeticIntensity(long long K, long long N) {
    double totalFlops = static_cast<double>(K) * static_cast<double>(N);
    double totalBytes = 2.0 * static_cast<double>(N) * kBytesPerElement;
    return totalFlops / totalBytes;
}

int main() {
    printf("=== Section 12.1: arithmetic intensity of an unfused vs. a fused chain ===\n\n");

    const long long N = 1'000'000;  // element count -- cancels out of the ratio entirely, shown for several K
    std::vector<long long> testChainLengths = {1, 2, 4, 8, 16, 32, 64, 100, 101, 128, 1000, 1'000'000};

    printf("  %-10s %-24s %-24s\n", "K", "AI_unfused (FLOPs/byte)", "AI_fused (FLOPs/byte)");
    double firstUnfusedAI = unfusedArithmeticIntensity(testChainLengths[0], N);
    bool unfusedAIConstant = true;
    bool fusedAIStrictlyIncreasing = true;
    double prevFusedAI = -1.0;
    for (long long K : testChainLengths) {
        double aiUnfused = unfusedArithmeticIntensity(K, N);
        double aiFused = fusedArithmeticIntensity(K, N);
        printf("  %-10lld %-24.6f %-24.6f\n", K, aiUnfused, aiFused);
        if (std::fabs(aiUnfused - firstUnfusedAI) > 1e-9) unfusedAIConstant = false;
        if (aiFused <= prevFusedAI) fusedAIStrictlyIncreasing = false;
        prevFusedAI = aiFused;
    }

    printf("\nself-check: AI_unfused is EXACTLY %.6f FLOPs/byte for every K tested, independent of\n", firstUnfusedAI);
    printf("chain length and of N (%s)\n", unfusedAIConstant ? "confirmed" : "MISMATCH");
    printf("self-check: AI_fused is strictly increasing as K grows (%s)\n",
           fusedAIStrictlyIncreasing ? "confirmed" : "MISMATCH");
    printf("\nBoth totals grow with K -- but their RATIO behaves completely differently: unfused\n");
    printf("traffic grows exactly as fast as unfused work does, so the ratio never moves; fused\n");
    printf("traffic stays flat while fused work keeps growing, so the ratio climbs without bound.\n");

    bool allOk = unfusedAIConstant && fusedAIStrictlyIncreasing;
    return allOk ? 0 : 1;
}
