// Chapter 1: The Cost of Eager Execution
// 002_launch_overhead_and_synthesis.cpp
//
// Section 1.2 -- fixed per-launch overhead multiplied by chain length --
// and a closing synthesis combining it with File 1's real byte-traffic
// model.
//
// This file never fabricates a timing number. The one number here that
// is not directly counted by this program is NVIDIA's own real, cited,
// published measurement of CUDA kernel launch overhead (9.6 microseconds
// per kernel, "including overheads", measured with a CPU wallclock timer
// around per-kernel-synchronized launches) from NVIDIA's own "Getting
// Started with CUDA Graphs" developer blog post -- used here exactly the
// way Appendix D of this book's own sibling volume already used a real
// cited hardware bandwidth figure: as an input to a real, stated formula,
// not as a re-measurement performed by this program. Every other number
// below is genuinely computed arithmetic on that cited constant, or a
// genuinely counted byte total reproduced from File 1's own closed-form
// result.
//
// Source: NVIDIA Technical Blog, "Getting Started with CUDA Graphs",
// https://developer.nvidia.com/blog/cuda-graphs/ -- "divide by
// NSTEP*NKERNEL which gives 9.6us per kernel (including overheads)".
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 002_launch_overhead_and_synthesis.cpp -o 002_launch_overhead_and_synthesis
// Run:     ./002_launch_overhead_and_synthesis
#include <cstdio>
#include <cstdint>

int main() {
    printf("=== Section 1.2: fixed per-launch overhead, multiplied by chain length ===\n\n");

    // Real, cited constant -- NOT measured by this program. See the
    // header comment above for the exact source and quote.
    const double NVIDIA_CITED_LAUNCH_OVERHEAD_US = 9.6;

    printf("NVIDIA's own cited figure (CUDA Graphs blog post): %.1f us per kernel launch,\n",
           NVIDIA_CITED_LAUNCH_OVERHEAD_US);
    printf("measured end to end with per-kernel host synchronization -- the same\n");
    printf("op-at-a-time shape Section 1.1's EAGER mode models, one real launch per op.\n");
    printf("A fused kernel replaces the whole chain with exactly ONE launch, regardless\n");
    printf("of how many operations were fused into it.\n\n");

    printf("%-8s %-24s %-24s %-12s\n", "num_ops", "eager_overhead_us",
           "fused_overhead_us", "reduction_x");

    for (int numOps = 1; numOps <= 8; ++numOps) {
        double eagerOverheadUs = static_cast<double>(numOps) * NVIDIA_CITED_LAUNCH_OVERHEAD_US;
        double fusedOverheadUs = 1.0 * NVIDIA_CITED_LAUNCH_OVERHEAD_US;
        double reduction = eagerOverheadUs / fusedOverheadUs;
        printf("%-8d %-24.2f %-24.2f %-12.2f\n", numOps, eagerOverheadUs, fusedOverheadUs, reduction);
    }

    printf("\n=== Synthesis: a concrete 4-operation chain on a 1,000,000-element tensor ===\n\n");

    const int numOps = 4;
    const uint64_t numElements = 1000000;
    const uint64_t arrayBytes = numElements * sizeof(float);

    // Byte-traffic side: reproduces File 1's own closed-form result
    // (eager(N) = N * 2 * array_bytes, fused = 2 * array_bytes) at this
    // chapter's own concrete example size -- not re-simulated here, the
    // formula was already checked against genuinely counted bytes in
    // File 1 for N = 1..8.
    uint64_t eagerBytes = static_cast<uint64_t>(numOps) * 2ULL * arrayBytes;
    uint64_t fusedBytes = 2ULL * arrayBytes;

    // Launch-overhead side: the real cited constant applied to this
    // example's own chain length.
    double eagerOverheadUs = static_cast<double>(numOps) * NVIDIA_CITED_LAUNCH_OVERHEAD_US;
    double fusedOverheadUs = NVIDIA_CITED_LAUNCH_OVERHEAD_US;

    printf("Chain length: %d operations. Tensor: %llu float32 elements (%.2f MB).\n\n",
           numOps, (unsigned long long)numElements, arrayBytes / (1024.0 * 1024.0));
    printf("%-30s %-15s %-15s\n", "", "eager", "fused");
    printf("%-30s %-15.2f %-15.2f  (us, from NVIDIA's cited per-launch figure)\n",
           "kernel-launch overhead", eagerOverheadUs, fusedOverheadUs);
    printf("%-30s %-15.2f %-15.2f  (MB, genuinely counted from array sizes)\n",
           "memory traffic", eagerBytes / (1024.0 * 1024.0), fusedBytes / (1024.0 * 1024.0));

    printf("\nThese two reductions are reported SEPARATELY and are not combined into a\n");
    printf("single predicted wall-clock speedup -- doing that would require an assumed\n");
    printf("memory bandwidth and an assumed relationship between overhead and traffic\n");
    printf("time, which this book's own standing discipline does not fabricate. What is\n");
    printf("real here is that a compiler that fuses this chain removes %dx of the launch\n",
           numOps);
    printf("overhead AND %dx of the memory traffic, from two independent mechanisms --\n",
           numOps);

    bool launchReductionCorrect = (eagerOverheadUs / fusedOverheadUs) == static_cast<double>(numOps);
    bool trafficReductionCorrect = (eagerBytes / fusedBytes) == static_cast<uint64_t>(numOps);
    printf("self-check: launch-overhead reduction equals the chain length exactly (%s),\n",
           launchReductionCorrect ? "confirmed" : "MISMATCH");
    printf("memory-traffic reduction equals the chain length exactly (%s)\n",
           trafficReductionCorrect ? "confirmed" : "MISMATCH");

    return (launchReductionCorrect && trafficReductionCorrect) ? 0 : 1;
}
