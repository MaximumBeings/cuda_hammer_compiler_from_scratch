// Appendix E: Profiling and Benchmarking
// 104_ch22s_own_unroll_factor_experiment_under_a_real_instruction_counter.cpp
//
// Section E.3 (capstone) -- Chapter 22's own File 054 built
// estimateLoopOverheadCost(), a linear model that always recommends the
// LARGEST unroll factor, then measured computeUnrolled() for real across
// unroll factors 1,2,4,8,16,32,64 at extent=50,000,000 and found, honestly,
// that the model's own prediction did NOT always match the real fastest
// factor -- and named the reason in prose, without measuring it: "real
// hardware has other mechanisms this simple model was never told about --
// instruction-cache pressure, register pressure, and lost auto-
// vectorization opportunities." Sections E.1 and E.2 showed that Cachegrind
// and Callgrind can turn a claim like that into a real, counted number
// instead of a plausible-sounding sentence. This file reuses File 054's own
// computeUnrolled() UNCHANGED and asks Cachegrind directly: does real
// INSTRUCTION COUNT grow with unroll factor the way "instruction-cache
// pressure" would predict, and do real D1/I1 miss counts move at all? The
// extent here is reduced from File 054's own 50,000,000 to 10,000,000 --
// stated honestly, not silently: Cachegrind's own instrumentation overhead
// (Sections E.1/E.2 measured roughly 90-115x real time on a 20,000,000-
// element run) makes a full 50,000,000-element, 7-factor sweep impractical
// to run as part of building this book, while 10,000,000 four-byte floats
// (40MB) still comfortably exceeds this machine's own measured 33MB LL
// cache (Section E.1's own cachegrind warning line), so the mechanism under
// test is still real at this size.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 -g 104_ch22s_own_unroll_factor_experiment_under_a_real_instruction_counter.cpp -o 104_driver <factor>
// (Same real reason as Section E.1's File 103: computeUnrolled() keeps
// -O2 -- the same optimization level File 054's own wall-clock numbers
// were measured at -- and __attribute__((noinline)) keeps it a real,
// separately profilable symbol instead of trusting that six call sites
// happen to be enough to stop GCC from inlining it anyway.)
// Run:     ./104_driver <factor>      (factor in {1,2,4,8,16,32,64}; prints result + a self-check)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <limits>

static const long long EXTENT = 10'000'000;

// ==================== computeUnrolled() -- reused UNCHANGED from Chapter 22's File 054 ====================
__attribute__((noinline))
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
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    long long factor = (argc > 1) ? std::atoll(argv[1]) : 1;
    printf("=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=%lld, extent=%lld ===\n\n", factor, EXTENT);

    std::vector<float> a(static_cast<size_t>(EXTENT));
    for (long long i = 0; i < EXTENT; ++i) a[static_cast<size_t>(i)] = static_cast<float>(i % 997) * 0.001f;

    std::vector<float> baseline = computeUnrolled(a, 1);
    std::vector<float> result;
    double bestMs = std::numeric_limits<double>::max();
    const int trials = 5;
    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::steady_clock::now();
        result = computeUnrolled(a, factor);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < bestMs) bestMs = ms;
    }
    bool correct = arraysExactlyEqual(result, baseline);
    printf("result[0]=%g  result[%lld]=%g\n", result[0], EXTENT - 1, result[static_cast<size_t>(EXTENT - 1)]);
    printf("best-of-%d wall-clock: %.3f ms\n", trials, bestMs);
    printf("self-check: bit-for-bit equal to unrollFactor=1 baseline (%s)\n", correct ? "confirmed" : "MISMATCH");

    return correct ? 0 : 1;
}
