// Appendix A: Installation and Setup -- Building CUDA Hammer's Toolchain
// 094_detecting_your_real_vector_isa_before_chapter_19.cpp
//
// Section A.2 -- Chapter 19's own real vectorized-CPU code generator
// (File 045 onward) picks its own target ISA like this, at compile
// time, in its own main():
//
//   #if defined(__x86_64__)
//       Isa hostIsa = Isa::Avx2;
//   #elif defined(__aarch64__)
//       Isa hostIsa = Isa::Neon;
//   #endif
//
// That is a real, working assumption on both of this book's own two
// authoring machines -- but it IS an assumption: `__x86_64__` means
// "this CPU's instruction set is x86-64," not "this specific CPU's
// flags include avx2 and fma" (older x86-64 CPUs predate AVX2 by
// several years). This file makes that assumption explicit and
// CHECKS it directly, the same way this book's own TOC.md already
// verified it once for its own two machines ("AVX2/AVX512F/FMA
// present in /proc/cpuinfo," "genuine NEON (asimd) confirmed") --
// reading the real, live `/proc/cpuinfo` on THIS machine, not
// re-asserting what worked on the authoring machines.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 094_detecting_your_real_vector_isa_before_chapter_19.cpp -o 094_driver
// Run:     ./094_driver

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Reads the real, live /proc/cpuinfo "flags" line (Linux only -- both
// of this book's own machines are Linux, per TOC.md's own toolchain
// notes) and returns the real set of advertised CPU feature flags.
static std::vector<std::string> readRealCpuFlags() {
    std::ifstream f("/proc/cpuinfo");
    std::vector<std::string> flags;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("flags", 0) == 0 || line.rfind("Features", 0) == 0) {
            std::istringstream iss(line);
            std::string token;
            while (iss >> token) flags.push_back(token);
            break;  // first CPU's flags line is enough -- real machines here are single-ISA
        }
    }
    return flags;
}

static bool hasFlag(const std::vector<std::string>& flags, const std::string& want) {
    for (const auto& f : flags) if (f == want) return true;
    return false;
}

int main() {
    printf("=== Appendix A.2: real vector-ISA detection, ahead of Chapter 19 ===\n\n");

    // Compile-time architecture macro -- exactly what Chapter 19's own
    // generateFullVectorizedProgram() driver branches on.
#if defined(__x86_64__)
    printf("compile-time architecture macro: __x86_64__ defined\n");
    printf("-> Chapter 19's own generator will pick Isa::Avx2 for this machine.\n\n");
    std::vector<std::string> flags = readRealCpuFlags();
    bool hasAvx2 = hasFlag(flags, "avx2");
    bool hasFma  = hasFlag(flags, "fma");
    printf("real /proc/cpuinfo check: avx2 flag %s, fma flag %s\n",
           hasAvx2 ? "PRESENT" : "ABSENT", hasFma ? "PRESENT" : "ABSENT");
    if (hasAvx2 && hasFma) {
        printf("self-check: this CPU's own real advertised flags CONFIRM the compile-time\n");
        printf("assumption -- Chapter 19's AVX2/FMA path will genuinely compile and run here\n");
        printf("(confirmed)\n");
    } else {
        printf("self-check: MISMATCH -- this CPU is x86_64 but its own real flags do not\n");
        printf("advertise both avx2 and fma. Chapter 19's -mavx2 -mfma build would fail to\n");
        printf("run correctly here (illegal-instruction risk) even though it would compile.\n");
        printf("Use Chapter 19's own scalar fallback path, or an older -msse4.2-only build.\n");
    }
#elif defined(__aarch64__)
    printf("compile-time architecture macro: __aarch64__ defined\n");
    printf("-> Chapter 19's own generator will pick Isa::Neon for this machine.\n\n");
    std::vector<std::string> flags = readRealCpuFlags();
    bool hasAsimd = hasFlag(flags, "asimd");
    printf("real /proc/cpuinfo check: asimd flag %s\n", hasAsimd ? "PRESENT" : "ABSENT");
    if (hasAsimd) {
        printf("self-check: this CPU's own real advertised flags CONFIRM the compile-time\n");
        printf("assumption -- Chapter 19's NEON path will genuinely compile and run here\n");
        printf("(confirmed). NEON is architecturally mandatory on aarch64, so this should\n");
        printf("never actually fail on a real aarch64 machine -- checked directly anyway,\n");
        printf("not assumed.\n");
    } else {
        printf("self-check: MISMATCH -- this reports as aarch64 but asimd is absent from its\n");
        printf("own real flags. This should not happen on real aarch64 hardware; re-check\n");
        printf("/proc/cpuinfo directly before trusting Chapter 19's own build.\n");
    }
#else
    printf("compile-time architecture macro: neither __x86_64__ nor __aarch64__ is defined.\n");
    printf("Chapter 19's own generator only targets these two real ISAs (its own #error\n");
    printf("says so directly) -- this machine cannot follow Chapter 19's codegen sections\n");
    printf("as written; the interpreter-only path (Chapters 4-17) still applies.\n");
#endif

    printf("\nreal result: read directly from this machine's own /proc/cpuinfo, not assumed\n");
    printf("from the compile-time architecture macro alone.\n");
    return 0;
}
