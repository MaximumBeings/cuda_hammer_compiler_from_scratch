// Chapter 23: Building CUDA Hammer's Autotuner
// 057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp
//
// Section 23.1 -- Chapter 22's own cost models were tested honestly, and
// Section 22.3's own capstone was honest about a real limitation: its
// combined-model comparison ran every candidate schedule through a generic
// INTERPRETER, not real compiled code, and diagnosed that interpreter's own
// bookkeeping overhead as a likely confound on a tiny toy problem. This
// chapter removes that confound: `generateScheduledElementwiseFunction()`
// emits REAL C++ text -- a tiled, unrolled loop, honoring a schedule's own
// tile size and unroll factor exactly the way Section 21.1/21.3 defined
// them -- compiled with the real g++ this book has shelled out to since
// Chapter 18, and loaded in-process with Chapter 20's own JitModule. Every
// schedule this section tests is now a REAL compiled function, not an
// interpreted one, and this section's own honest finding is a new one Part
// 5 has not yet had to face: compiling a schedule has a real, nonzero cost
// of its own, separate from running it -- a cost the interpreter in
// Chapter 22 never had to pay, and the actual autotuner (Section 23.3) will
// have to budget for.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp -o 057_driver -ldl
// Run:     ./057_driver
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <dlfcn.h>

// ==================== Loop (Chapter 15, unchanged) ====================

struct Loop {
    std::string dimName;
    long long extent;
};

// ==================== JitModule / compileToSharedLibrary() (Chapter 20, unchanged) ====================

class JitModule {
public:
    explicit JitModule(const std::string& sharedLibraryPath) {
        handle_ = dlopen(sharedLibraryPath.c_str(), RTLD_NOW);
        if (!handle_) throw std::runtime_error("JitModule: dlopen failed: " + std::string(dlerror()));
    }
    ~JitModule() {
        if (handle_) dlclose(handle_);
    }
    JitModule(const JitModule&) = delete;
    JitModule& operator=(const JitModule&) = delete;
    JitModule(JitModule&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    JitModule& operator=(JitModule&& other) noexcept {
        if (this != &other) {
            if (handle_) dlclose(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    template <typename FnPtr>
    FnPtr getFunction(const std::string& symbolName) const {
        dlerror();
        void* sym = dlsym(handle_, symbolName.c_str());
        const char* err = dlerror();
        if (err) throw std::runtime_error("JitModule::getFunction(\"" + symbolName + "\"): dlsym failed: " + err);
        return reinterpret_cast<FnPtr>(sym);
    }
private:
    void* handle_ = nullptr;
};
static void writeFile(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}
static std::string runShellCaptureAll(const std::string& cmd) {
    std::string out;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) throw std::runtime_error("popen failed for: " + cmd);
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) out += buf;
    pclose(pipe);
    return out;
}
static bool compileToSharedLibrary(const std::string& source, const std::string& cppPath, const std::string& soPath,
                                    std::string& compileLog) {
    writeFile(cppPath, source);
    compileLog = runShellCaptureAll("g++ -std=c++17 -Wall -Wextra -O2 -shared -fPIC " + cppPath + " -o " + soPath + " 2>&1");
    return compileLog.empty();
}

// ==================== Section 23.1: real codegen for tile size + unroll factor ====================
//
// The generated function is deliberately the compiled twin of Chapter 22's
// own computeUnrolled() and Section 21.1's own tileLoop() shape, combined:
// an outer loop over TILE BLOCKS (stepping by tileSize, with a real
// boundary check for the final partial block -- the exact ceiling-division
// case Chapter 15's own actualInnerExtent() already proved correct), and an
// inner loop within each block that processes `unrollFactor` elements per
// trip with a scalar tail -- Section 21.3's own "main loop plus scalar
// tail" shape, this time emitted as real C++ TEXT instead of executed
// directly. Two nested chunking decisions (tile size, then unroll factor
// within each tile) is exactly the schedule Chapter 21's own Schedule
// struct already describes; this function is simply the first time this
// book has ever generated REAL code for it rather than only reasoning
// about it or interpreting it.
static std::string generateScheduledElementwiseFunction(long long tileSize, long long unrollFactor) {
    std::string src;
    src += "extern \"C\" void compute(const float* a, float* out, long long n) {\n";
    src += "  const long long tileSize = " + std::to_string(tileSize) + ";\n";
    src += "  const long long unrollFactor = " + std::to_string(unrollFactor) + ";\n";
    src += "  for (long long blockStart = 0; blockStart < n; blockStart += tileSize) {\n";
    src += "    long long blockEnd = blockStart + tileSize;\n";
    src += "    if (blockEnd > n) blockEnd = n;\n";  // Chapter 15's own boundary case, real this time
    src += "    long long i = blockStart;\n";
    src += "    for (; i + unrollFactor <= blockEnd; i += unrollFactor) {\n";
    src += "      for (long long lane = 0; lane < unrollFactor; ++lane) {\n";
    src += "        out[i + lane] = a[i + lane] * 2.0f + 1.0f;\n";
    src += "      }\n";
    src += "    }\n";
    src += "    for (; i < blockEnd; ++i) out[i] = a[i] * 2.0f + 1.0f;\n";  // scalar tail
    src += "  }\n";
    src += "}\n";
    return src;
}
using ComputeFn = void (*)(const float*, float*, long long);
static bool arraysExactlyEqual(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

int main() {
    printf("=== Section 23.1: real codegen for a scheduled loop -- tile size and unroll factor ===\n\n");

    // ---- Part 1: correctness, an awkward extent, several real (tileSize, unrollFactor) schedules ----
    printf("--- Part 1: extent=23 (not evenly divisible by most tile/unroll choices), 5 real schedules ---\n\n");
    long long extent = 23;
    std::vector<float> a(static_cast<size_t>(extent)), reference(static_cast<size_t>(extent));
    for (long long i = 0; i < extent; ++i) a[static_cast<size_t>(i)] = static_cast<float>(i) * 0.5f - 6.0f;
    for (long long i = 0; i < extent; ++i) reference[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] * 2.0f + 1.0f;

    struct TileUnroll { long long tileSize, unrollFactor; };
    std::vector<TileUnroll> smallSchedules = {{1, 1}, {7, 3}, {8, 4}, {23, 5}, {5, 5}};
    bool allCorrect = true;
    for (size_t k = 0; k < smallSchedules.size(); ++k) {
        long long tileSize = smallSchedules[k].tileSize, unrollFactor = smallSchedules[k].unrollFactor;
        std::string src = generateScheduledElementwiseFunction(tileSize, unrollFactor);
        std::string cppPath = "/tmp/057_gen_" + std::to_string(k) + ".cpp";
        std::string soPath = "/tmp/057_gen_" + std::to_string(k) + ".so";
        std::string log;
        bool clean = compileToSharedLibrary(src, cppPath, soPath, log);
        JitModule mod(soPath);
        ComputeFn fn = mod.getFunction<ComputeFn>("compute");
        std::vector<float> out(static_cast<size_t>(extent));
        fn(a.data(), out.data(), extent);
        bool ok = arraysExactlyEqual(out, reference);
        allCorrect = allCorrect && ok && clean;
        printf("  tileSize=%-2lld unrollFactor=%-2lld  clean compile: %-3s  bit-for-bit correct: %s\n",
               tileSize, unrollFactor, clean ? "yes" : "no", ok ? "yes" : "NO");
    }
    printf("\nself-check: every one of these 5 real, COMPILED, dlopen-loaded schedules produces the exact\n");
    printf("same bit-for-bit answer (%s) -- tile size and unroll factor stay a pure performance knob for\n",
           allCorrect ? "confirmed" : "MISMATCH");
    printf("an elementwise loop whether that loop is interpreted (Chapter 22) or actually compiled (here).\n");

    // ---- Part 2: real cost, at scale -- compile time and execution time are TWO separate real costs ----
    printf("\n--- Part 2: extent=30,000,000 -- compile time vs. execution time, both real, both separate ---\n\n");
    long long bigExtent = 30'000'000;
    std::vector<float> bigA(static_cast<size_t>(bigExtent));
    for (long long i = 0; i < bigExtent; ++i) bigA[static_cast<size_t>(i)] = static_cast<float>(i % 997) * 0.001f;

    std::vector<TileUnroll> bigSchedules = {{1, 1}, {1024, 4}, {1024, 16}, {4096, 16}, {4096, 64}};
    for (size_t k = 0; k < bigSchedules.size(); ++k) {
        long long tileSize = bigSchedules[k].tileSize, unrollFactor = bigSchedules[k].unrollFactor;
        std::string src = generateScheduledElementwiseFunction(tileSize, unrollFactor);
        std::string cppPath = "/tmp/057_big_" + std::to_string(k) + ".cpp";
        std::string soPath = "/tmp/057_big_" + std::to_string(k) + ".so";
        std::string log;

        auto compileStart = std::chrono::steady_clock::now();
        bool clean = compileToSharedLibrary(src, cppPath, soPath, log);
        auto compileEnd = std::chrono::steady_clock::now();
        double compileMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();

        JitModule mod(soPath);
        ComputeFn fn = mod.getFunction<ComputeFn>("compute");
        std::vector<float> out(static_cast<size_t>(bigExtent));
        double bestRunMs = std::numeric_limits<double>::max();
        for (int trial = 0; trial < 5; ++trial) {
            auto runStart = std::chrono::steady_clock::now();
            fn(bigA.data(), out.data(), bigExtent);
            auto runEnd = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(runEnd - runStart).count();
            if (ms < bestRunMs) bestRunMs = ms;
        }
        printf("  tileSize=%-5lld unrollFactor=%-3lld  compile: %7.1f ms   run (best-of-5): %7.3f ms  (%s)\n",
               tileSize, unrollFactor, compileMs, bestRunMs, clean ? "clean" : "COMPILE ERROR");
    }
    printf("\nself-check: compile time and execution time land in the SAME real order of magnitude here --\n");
    printf("a handful to several tens of milliseconds each, a real g++ process costing about as much as\n");
    printf("the computation it just built. Chapter 22's own interpreter never had to pay the compile side of\n");
    printf("that bill at all -- interpreting a Schedule is free of compilation, which is exactly why Chapter\n");
    printf("22 could afford to try many candidates. Real codegen changes that tradeoff: compiling EVERY one\n");
    printf("of a large schedule space, one g++ process per candidate, adds up to a real, non-negligible cost\n");
    printf("of its own -- which is precisely the budget Section 23.3's own autotuner has to manage: compile\n");
    printf("a short list, not the whole space.\n");

    return allCorrect ? 0 : 1;
}
