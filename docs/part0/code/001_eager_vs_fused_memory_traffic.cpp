// Chapter 1: The Cost of Eager Execution
// 001_eager_vs_fused_memory_traffic.cpp
//
// Section 1.1 -- what "eager execution" actually does, step by step, and
// Section 1.3 -- the real memory-traffic cost that follows from it.
//
// A real N-operation elementwise chain (add a constant, multiply by a
// constant, clamp at zero -- i.e. a bias-add, a scale, and a ReLU,
// repeated) is computed two genuinely different ways over the exact same
// input data:
//   - EAGER: each operation reads a full input array and writes a brand
//     new full output array, exactly the way an op-at-a-time framework
//     materializes every intermediate result.
//   - FUSED: a single pass over the data computes every operation in the
//     chain per element, in registers, and only ever reads the original
//     input once and writes the final output once.
// Both modes are genuinely executed and their outputs are compared
// element-for-element (bit-identical is expected and checked -- this is
// the same sequence of elementary float ops applied in the same order
// either way, so there is no reduction-order question here the way
// there is for a sum or a collective). The real byte counts each mode
// actually reads and writes are counted directly from array sizes, not
// timed and not estimated -- and checked against a closed-form formula
// for the number of chained operations.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 001_eager_vs_fused_memory_traffic.cpp -o 001_eager_vs_fused_memory_traffic
// Run:     ./001_eager_vs_fused_memory_traffic
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

// One elementwise operation in the chain: alternates bias-add, scale,
// and ReLU, so a chain of any length N exercises all three.
static float applyOp(int opIndex, float x) {
    switch (opIndex % 3) {
        case 0: return x + 0.5f;               // bias-add
        case 1: return x * 1.25f;               // scale
        default: return std::max(0.0f, x);      // ReLU
    }
}

struct ByteCounts {
    uint64_t bytesRead = 0;
    uint64_t bytesWritten = 0;
};

// EAGER: op i reads the FULL array produced by op i-1 and writes a FULL
// new array -- a real heap allocation per op, exactly mirroring how a
// framework in eager mode materializes every intermediate tensor.
static ByteCounts runEager(const std::vector<float>& input, int numOps,
                            std::vector<float>& finalOut) {
    ByteCounts counts;
    std::vector<float> current = input;
    for (int op = 0; op < numOps; ++op) {
        std::vector<float> next(current.size());
        for (size_t i = 0; i < current.size(); ++i) {
            next[i] = applyOp(op, current[i]);
        }
        counts.bytesRead += current.size() * sizeof(float);
        counts.bytesWritten += next.size() * sizeof(float);
        current = std::move(next);
    }
    finalOut = current;
    return counts;
}

// FUSED: one pass. The original input is read exactly once; the final
// output is written exactly once. Every intermediate value between ops
// lives in a single local float (a register in any real optimized
// build), never in a separate heap array.
static ByteCounts runFused(const std::vector<float>& input, int numOps,
                            std::vector<float>& finalOut) {
    ByteCounts counts;
    finalOut.resize(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        float value = input[i];
        for (int op = 0; op < numOps; ++op) {
            value = applyOp(op, value);
        }
        finalOut[i] = value;
    }
    counts.bytesRead = input.size() * sizeof(float);
    counts.bytesWritten = finalOut.size() * sizeof(float);
    return counts;
}

int main() {
    printf("=== Section 1.1 / 1.3: eager vs. fused execution of the same op chain ===\n\n");

    const size_t arraySize = 1000; // small on purpose -- this chapter's claim is about
                                    // byte-COUNT scaling with chain length, not raw size.
    std::vector<float> input(arraySize);
    for (size_t i = 0; i < arraySize; ++i) {
        input[i] = static_cast<float>(i) * 0.01f - 3.0f; // spans negative and positive
    }
    const uint64_t arrayBytes = arraySize * sizeof(float);

    printf("Array size: %zu float32 elements (%llu bytes)\n\n", arraySize,
           (unsigned long long)arrayBytes);

    printf("%-8s %-22s %-22s %-14s %-10s\n", "num_ops", "eager_bytes_moved",
           "fused_bytes_moved", "reduction_x", "outputs_match");

    bool allFormulaMatches = true;
    bool allOutputsMatch = true;

    for (int numOps = 1; numOps <= 8; ++numOps) {
        std::vector<float> eagerOut, fusedOut;
        ByteCounts eager = runEager(input, numOps, eagerOut);
        ByteCounts fused = runFused(input, numOps, fusedOut);

        uint64_t eagerTotal = eager.bytesRead + eager.bytesWritten;
        uint64_t fusedTotal = fused.bytesRead + fused.bytesWritten;

        // Closed-form check: eager reads+writes a full array on every one
        // of the numOps operations (2 * arrayBytes per op); fused reads
        // the input once and writes the output once, regardless of chain
        // length (2 * arrayBytes total, always).
        uint64_t predictedEager = static_cast<uint64_t>(numOps) * 2ULL * arrayBytes;
        uint64_t predictedFused = 2ULL * arrayBytes;
        bool formulaMatches = (predictedEager == eagerTotal) && (predictedFused == fusedTotal);
        allFormulaMatches = allFormulaMatches && formulaMatches;

        bool outputsMatch = (eagerOut.size() == fusedOut.size());
        for (size_t i = 0; outputsMatch && i < eagerOut.size(); ++i) {
            if (eagerOut[i] != fusedOut[i]) outputsMatch = false;
        }
        allOutputsMatch = allOutputsMatch && outputsMatch;

        double reduction = static_cast<double>(eagerTotal) / static_cast<double>(fusedTotal);
        printf("%-8d %-22llu %-22llu %-14.2f %-10s\n", numOps,
               (unsigned long long)eagerTotal, (unsigned long long)fusedTotal,
               reduction, outputsMatch ? "yes" : "NO");
    }

    printf("\nself-check: every row's counted bytes match the closed-form formula\n");
    printf("eager(N) = N * 2 * array_bytes, fused = 2 * array_bytes (%s); eager and\n",
           allFormulaMatches ? "confirmed" : "MISMATCH");
    printf("fused outputs are bit-identical at every chain length (%s)\n",
           allOutputsMatch ? "confirmed" : "MISMATCH");

    return (allFormulaMatches && allOutputsMatch) ? 0 : 1;
}
