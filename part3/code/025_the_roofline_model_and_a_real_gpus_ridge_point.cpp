// Chapter 12: Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels
// 025_the_roofline_model_and_a_real_gpus_ridge_point.cpp
//
// Section 12.2 -- the roofline model's ridge point, computed here from
// one real GPU's own published peak numbers, that turns Section 12.1's
// arithmetic-intensity ratio into a concrete memory-bound-or-compute-
// bound classification -- the roofline model itself is real, published
// work: Samuel Williams, Andrew Waterman, and David Patterson,
// "Roofline: An Insightful Visual Performance Model for Multicore
// Architectures," Communications of the ACM, April 2009.
//
// Reuses Section 12.1's unfusedArithmeticIntensity()/fusedArithmeticIntensity()
// completely unchanged -- this section's only new idea is a THRESHOLD to
// compare that ratio against.
//
// Real external numbers used below, cited to NVIDIA's own published A100
// datasheet (40GB PCIe SKU): FP32 peak throughput 19.5 TFLOPS, memory
// bandwidth 1,555 GB/s. The ridge point -- the arithmetic intensity at
// which a kernel exactly saturates BOTH the compute engine and the
// memory bus at once -- is peak FLOPs/s divided by peak bytes/s, computed
// here from those two cited numbers rather than looked up as a separate
// published figure.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 025_the_roofline_model_and_a_real_gpus_ridge_point.cpp -o 025_the_roofline_model_and_a_real_gpus_ridge_point
// Run:     ./025_the_roofline_model_and_a_real_gpus_ridge_point
#include <cstdio>
#include <cmath>
#include <vector>

// ==================== Section 12.1's arithmetic intensity (unchanged) ====================

static constexpr double kBytesPerElement = 4.0;

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

// ==================== Section 12.2: a real GPU's ridge point (new) ====================
//
// Cited directly to NVIDIA's own A100 datasheet (40GB PCIe SKU):
//   FP32 peak throughput: 19.5 TFLOPS  = 19.5e12 FLOPs/second
//   Memory bandwidth:     1,555 GB/s   = 1.555e12 bytes/second
static constexpr double kA100Fp32PeakFlopsPerSec = 19.5e12;
static constexpr double kA100BandwidthBytesPerSec = 1555.0e9;

static double ridgePointFlopsPerByte() {
    return kA100Fp32PeakFlopsPerSec / kA100BandwidthBytesPerSec;
}

static const char* classify(double arithmeticIntensity, double ridgePoint) {
    return (arithmeticIntensity >= ridgePoint) ? "compute-bound" : "memory-bound";
}

int main() {
    printf("=== Section 12.2: classifying Section 12.1's chains against a real GPU's ridge point ===\n\n");

    double ridge = ridgePointFlopsPerByte();
    printf("NVIDIA A100 (40GB PCIe) ridge point = %.4f TFLOPS / %.4f GB/s = %.4f FLOPs/byte\n",
           kA100Fp32PeakFlopsPerSec / 1e12, kA100BandwidthBytesPerSec / 1e9, ridge);

    const long long N = 1'000'000;
    std::vector<long long> testChainLengths = {1, 2, 4, 8, 16, 32, 64, 100, 101, 128, 1000, 1'000'000};

    printf("\n  %-10s %-16s %-16s %-16s %-16s\n", "K", "AI_unfused", "class", "AI_fused", "class");
    bool unfusedAlwaysMemoryBound = true;
    bool someFusedIsComputeBound = false;
    double firstUnfusedAI = unfusedArithmeticIntensity(testChainLengths[0], N);
    for (long long K : testChainLengths) {
        double aiUnfused = unfusedArithmeticIntensity(K, N);
        double aiFused = fusedArithmeticIntensity(K, N);
        const char* cu = classify(aiUnfused, ridge);
        const char* cf = classify(aiFused, ridge);
        printf("  %-10lld %-16.6f %-16s %-16.6f %-16s\n", K, aiUnfused, cu, aiFused, cf);
        if (aiUnfused >= ridge) unfusedAlwaysMemoryBound = false;
        if (aiFused >= ridge) someFusedIsComputeBound = true;
    }

    printf("\nself-check: the unfused chain is memory-bound at EVERY tested K, including K=1,000,000\n");
    printf("(its AI never grows) (%s)\n", unfusedAlwaysMemoryBound ? "confirmed" : "MISMATCH");
    printf("self-check: at least one tested K makes the FUSED chain compute-bound (%s)\n",
           someFusedIsComputeBound ? "confirmed" : "MISMATCH");

    // Smallest integer K at which the fused chain crosses the ridge point:
    // AI_fused(K) = K / (2 * bytesPerElement) >= ridge  =>  K >= ridge * 2 * bytesPerElement
    long long crossoverK = static_cast<long long>(std::ceil(ridge * 2.0 * kBytesPerElement));
    bool crossoverKIsComputeBound = fusedArithmeticIntensity(crossoverK, N) >= ridge;
    bool oneLessIsNotComputeBound = fusedArithmeticIntensity(crossoverK - 1, N) < ridge;
    printf("\nSmallest fused chain length that crosses this GPU's ridge point: K = %lld\n", crossoverK);
    printf("self-check: fusedAI(K=%lld) is compute-bound (%s), fusedAI(K=%lld) is NOT (%s)\n",
           crossoverK, crossoverKIsComputeBound ? "confirmed" : "MISMATCH",
           crossoverK - 1, oneLessIsNotComputeBound ? "confirmed" : "MISMATCH");
    printf("No UNFUSED chain of any length ever crosses this ridge point -- its arithmetic\n");
    printf("intensity is fixed at %.6f FLOPs/byte forever, by construction.\n", firstUnfusedAI);

    bool allOk = unfusedAlwaysMemoryBound && someFusedIsComputeBound &&
                 crossoverKIsComputeBound && oneLessIsNotComputeBound;
    return allOk ? 0 : 1;
}
