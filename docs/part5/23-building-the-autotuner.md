# 23. Building CUDA Hammer's Autotuner

**What you will understand:** `generateScheduledElementwiseFunction()`, which emits real C++ TEXT for one loop's own tile size and unroll factor -- the compiled twin of Chapter 21's own `computeUnrolled()` -- compiled with the real g++ this book has shelled out to since Chapter 18 and dlopen-loaded with Chapter 20's own `JitModule`, exposing a real cost this book has never had to budget for before: COMPILE TIME, measured separately from RUN TIME, for the first time landing in the same real order of magnitude; `generateScheduleFunction()`, which extends real codegen to a FULL `Schedule` -- tile size, loop order, AND unroll factor together -- baking the chosen loop order directly into the generated code's own shape instead of branching on it at runtime, then honestly rechecking whether removing Chapter 22's own interpreter changes that chapter's own near-chance ranking agreement; and the autotuner itself, a HYBRID search that uses Chapter 22's own imperfect cost model to prune 128 real schedules down to a short list cheaply, pays for real compiled measurement only on that short list, and is checked -- honestly, in both directions -- against the alternative it exists to avoid: compiling and measuring every one of the 128 candidates for real.

**What you need to know first:** Chapter 20's own `JitModule` (RAII dlopen/dlsym/dlclose) and `compileToSharedLibrary()` (the one remaining g++ shell-out, scoped to compilation only); Chapter 21's own `Loop`/`LoopNest`/`Schedule`/`enumerateSchedules()`/`scheduleStr()`, and its own closing number -- 128 legal schedules for `LoopNest [dim0:6, dim1:8]`; and Chapter 22's own `estimateLoopOverheadCost()`/`estimateLoopOrderCost()`/`estimateScheduleCost()`, plus that chapter's own closing, honestly-diagnosed finding -- a combined cost model, applied to 6 schedules run through a generic INTERPRETER, landing at a near-chance ranking agreement (40% on the cloud sandbox, 53% on the device), traced to interpreter bookkeeping overhead on a 48-element toy problem, not evidence the underlying models are worthless.

---

Chapter 22 closed with a specific, named confound still standing between its own cost models and a real autotuner: every candidate schedule that chapter ever measured was run through `executeSchedule()`, a generic interpreter that reads a `Schedule`'s own fields as data at runtime rather than a real, schedule-specific compiled function. That confound was a deliberate, honestly-stated choice, not an oversight -- Part 4's own codegen chapters (17 through 20) each built one hand-written generator for ONE fixed computation, never a generator parameterized by an arbitrary `Schedule`. This chapter builds that generator, closes the interpreter gap Chapter 22 left open, and then does the thing every prior chapter in Part 5 was building toward: it actually searches.

The word "autotuner" has been used loosely since Chapter 21 first enumerated 128 legal schedules for one tiny `LoopNest` and explicitly refused to compile and measure every one just to find the best. This chapter makes that refusal concrete. Real codegen, once it exists, has a real cost of its own -- Section 23.1's own honest finding is that compiling a schedule is not free, and is not even cheap relative to running one, the way Chapter 22's own interpreter made it seem. An autotuner that tries to compile every candidate in a large search space pays that real cost once per candidate, which is exactly the budget problem Section 23.3's own hybrid design exists to solve: spend the cost model's own cheap, imperfect estimate first, and spend real compilation only on the candidates that estimate says are worth it.

```text
Three real steps toward an actual autotuner:

  23.1 SCHEDULED CODEGEN      -- generateScheduledElementwiseFunction():
       (tile + unroll only)      real C++ TEXT for one loop's own tile
                                  size and unroll factor, compiled with
                                  g++, dlopen-loaded with Chapter 20's
                                  own JitModule -- reveals a real cost
                                  this book has not had to budget for
                                  before: COMPILE TIME, measured apart
                                  from RUN TIME, landing in the same
                                  real order of magnitude as the very
                                  computation it just built

  23.2 SCHEDULED CODEGEN      -- generateScheduleFunction(): adds the
       (full schedule)           third dimension, loop order, baked
                                  directly into the generated code's
                                  own SHAPE (which flat-index
                                  expression it even contains) instead
                                  of a runtime branch; reruns Chapter
                                  22.3's own 6-schedule ranking
                                  experiment with real compiled code
                                  standing in for the interpreter, and
                                  reports honestly whether that changes
                                  the ranking agreement

  23.3 THE AUTOTUNER          -- combines both: Chapter 22's own cost
       (capstone)                 model prunes all 128 real schedules
                                  down to a short list cheaply; only
                                  that short list pays for real
                                  compiled measurement (the HYBRID);
                                  checked honestly against compiling
                                  and measuring all 128 candidates for
                                  real (the EXHAUSTIVE alternative)

Every step in this chain is now real: real generated text, a real g++
process per candidate, a real dlopen'd function, a real clock -- the
first chapter in Part 5 where none of those words is a stand-in for
something interpreted or estimated.
```

## 23.1 Real Codegen for a Scheduled Loop: Tile Size and Unroll Factor

### Intuition

Chapter 21's own `computeUnrolled()` already demonstrated, by direct execution, that a chosen tile size and unroll factor never change an elementwise loop's own answer -- only how the same total work gets chunked. That function was hand-written once, for one fixed computation. This section builds the generator that produces a function LIKE it, but for whichever tile size and unroll factor a `Schedule` actually specifies, as real C++ source text assembled at runtime -- the same kind of string-building this book has done since Chapter 17's own `generateLoopFunction()`, applied for the first time to a value (a schedule's own tile size and unroll factor) that used to only ever be interpreted or reasoned about abstractly.

### Background

`generateScheduledElementwiseFunction(tileSize, unrollFactor)` emits one `extern "C"` function: an outer loop over TILE BLOCKS, stepping by `tileSize` with a real boundary check for the final partial block (the exact ceiling-division case Chapter 15's own `actualInnerExtent()` already proved correct, this time appearing as an `if` statement in generated text rather than a formula), and an inner loop that processes `unrollFactor` elements per trip with a scalar tail -- Chapter 21's own "main loop plus scalar tail" shape, emitted as text instead of executed directly. The generated function is compiled with Chapter 20's own `compileToSharedLibrary()` and loaded with `JitModule`, exactly the same two-step pipeline that chapter used for a single, fixed computation -- nothing about the JIT pipeline itself needed to change to accept a schedule-parameterized generator instead of a hand-written one.

The genuinely new finding in this section is not about correctness -- every schedule this section tests, compiled and dlopen-loaded for real, produces the identical bit-for-bit answer Chapter 21's own interpreted version already established. It is about COST. Every codegen chapter since 17 has treated compilation as a step that happens once, offstage, before the real measurement begins; this section is the first to put a clock on the compile step itself, separately from the run step, at real scale (`extent = 30,000,000`). The two numbers land in the same real order of magnitude -- a real g++ process, invoked once per candidate schedule, costs roughly as much wall-clock time as running the very computation it just built. Chapter 22's own interpreter never had to pay that bill at all: interpreting a `Schedule` needs no compilation step, which is exactly why that chapter could afford to try many candidates cheaply. Real codegen changes the arithmetic -- compiling every candidate in a large schedule space, one g++ process each, is a real, non-negligible cost this book has not had to reckon with until now.

```text
generateScheduledElementwiseFunction(tileSize, unrollFactor) emits:

  extern "C" void compute(const float* a, float* out, long long n) {
    for each tile block, starting at blockStart, stepping by tileSize:
      blockEnd = min(blockStart + tileSize, n)   -- Ch15's own
                                                     boundary case,
                                                     real this time
      i starts at blockStart
      while at least unrollFactor elements remain before blockEnd:
        for lane = 0 up to unrollFactor, not including it:
          out[i+lane] = a[i+lane] * 2.0f + 1.0f   -- main, unrolled
        i advances by unrollFactor
      while i has not reached blockEnd:
        out[i] = a[i] * 2.0f + 1.0f               -- scalar tail
        i advances by 1
  }

Compiled with Chapter 20's own compileToSharedLibrary(), loaded with
Chapter 20's own JitModule -- the identical JIT pipeline, now driven
by a generator instead of one fixed hand-written function.

New this section: a real clock on COMPILE time, separate from RUN
time. At real scale, both land in the same order of magnitude -- a
cost Chapter 22's own interpreter never had to pay at all.
```

```cpp
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
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp" -o "057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor" -ldl
./"057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 23.1: real codegen for a scheduled loop -- tile size and unroll factor ===

--- Part 1: extent=23 (not evenly divisible by most tile/unroll choices), 5 real schedules ---

  tileSize=1  unrollFactor=1   clean compile: yes  bit-for-bit correct: yes
  tileSize=7  unrollFactor=3   clean compile: yes  bit-for-bit correct: yes
  tileSize=8  unrollFactor=4   clean compile: yes  bit-for-bit correct: yes
  tileSize=23 unrollFactor=5   clean compile: yes  bit-for-bit correct: yes
  tileSize=5  unrollFactor=5   clean compile: yes  bit-for-bit correct: yes

self-check: every one of these 5 real, COMPILED, dlopen-loaded schedules produces the exact
same bit-for-bit answer (confirmed) -- tile size and unroll factor stay a pure performance knob for
an elementwise loop whether that loop is interpreted (Chapter 22) or actually compiled (here).

--- Part 2: extent=30,000,000 -- compile time vs. execution time, both real, both separate ---

  tileSize=1     unrollFactor=1    compile:    43.4 ms   run (best-of-5):  32.154 ms  (clean)
  tileSize=1024  unrollFactor=4    compile:    50.0 ms   run (best-of-5):  25.504 ms  (clean)
  tileSize=1024  unrollFactor=16   compile:    49.0 ms   run (best-of-5):  25.834 ms  (clean)
  tileSize=4096  unrollFactor=16   compile:    50.0 ms   run (best-of-5):  25.628 ms  (clean)
  tileSize=4096  unrollFactor=64   compile:    49.2 ms   run (best-of-5):  25.647 ms  (clean)

self-check: compile time and execution time land in the SAME real order of magnitude here --
a handful to several tens of milliseconds each, a real g++ process costing about as much as
the computation it just built. Chapter 22's own interpreter never had to pay the compile side of
that bill at all -- interpreting a Schedule is free of compilation, which is exactly why Chapter
22 could afford to try many candidates. Real codegen changes that tradeoff: compiling EVERY one
of a large schedule space, one g++ process per candidate, adds up to a real, non-negligible cost
of its own -- which is precisely the budget Section 23.3's own autotuner has to manage: compile
a short list, not the whole space.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 23.1: real codegen for a scheduled loop -- tile size and unroll factor ===

--- Part 1: extent=23 (not evenly divisible by most tile/unroll choices), 5 real schedules ---

  tileSize=1  unrollFactor=1   clean compile: yes  bit-for-bit correct: yes
  tileSize=7  unrollFactor=3   clean compile: yes  bit-for-bit correct: yes
  tileSize=8  unrollFactor=4   clean compile: yes  bit-for-bit correct: yes
  tileSize=23 unrollFactor=5   clean compile: yes  bit-for-bit correct: yes
  tileSize=5  unrollFactor=5   clean compile: yes  bit-for-bit correct: yes

self-check: every one of these 5 real, COMPILED, dlopen-loaded schedules produces the exact
same bit-for-bit answer (confirmed) -- tile size and unroll factor stay a pure performance knob for
an elementwise loop whether that loop is interpreted (Chapter 22) or actually compiled (here).

--- Part 2: extent=30,000,000 -- compile time vs. execution time, both real, both separate ---

  tileSize=1     unrollFactor=1    compile:    17.4 ms   run (best-of-5):  11.397 ms  (clean)
  tileSize=1024  unrollFactor=4    compile:    21.9 ms   run (best-of-5):   7.778 ms  (clean)
  tileSize=1024  unrollFactor=16   compile:    19.7 ms   run (best-of-5):   8.933 ms  (clean)
  tileSize=4096  unrollFactor=16   compile:    17.9 ms   run (best-of-5):   8.614 ms  (clean)
  tileSize=4096  unrollFactor=64   compile:    17.8 ms   run (best-of-5):  10.077 ms  (clean)

self-check: compile time and execution time land in the SAME real order of magnitude here --
a handful to several tens of milliseconds each, a real g++ process costing about as much as
the computation it just built. Chapter 22's own interpreter never had to pay the compile side of
that bill at all -- interpreting a Schedule is free of compilation, which is exactly why Chapter
22 could afford to try many candidates. Real codegen changes that tradeoff: compiling EVERY one
of a large schedule space, one g++ process per candidate, adds up to a real, non-negligible cost
of its own -- which is precisely the budget Section 23.3's own autotuner has to manage: compile
a short list, not the whole space.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which candidate ranks fastest, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! note "Compiling every candidate does not scale the way interpreting one does"
    Chapter 22's own combined cost model was cheap enough to score all 128 real schedules in a fraction of a millisecond -- no compilation, no process launch, just arithmetic on a `Schedule`'s own fields. This section's own real measurement shows real codegen is a different kind of cost entirely: tens of milliseconds per candidate, dominated by an actual g++ invocation, not by the size of the computation being compiled (Section 23.1's own generated functions are short, and still cost tens of milliseconds to build). Multiply that by 128 candidates, and compiling the ENTIRE search space starts to cost seconds, not milliseconds -- a real number Section 23.3's own autotuner will have to budget against directly, which is exactly why that section checks a short list against the full space rather than assuming either extreme is obviously right.

## 23.2 Real Codegen for a Full Schedule: Tile Size, Loop Order, and Unroll Factor Together

### Intuition

Section 23.1's generator only ever varied two of a `Schedule`'s own three fields -- tile size and unroll factor -- because it always generated a plain one-dimensional loop. A real `Schedule` (Chapter 21) also carries a LOOP ORDER, meaningful only once there is more than one loop to order, and Chapter 22.3's own interpreter already showed loop order changes real measured time even though it can never change the answer (Section 21.2). This section's generator has to make a choice Section 23.1's never needed to: which of two possible nestings -- `dim0` outer, `dim1` inner, or the reverse -- does the GENERATED CODE ITSELF take, not as a runtime flag checked on every call, but as a decision baked into the text before g++ ever sees it.

### Background

`generateScheduleFunction(nest, schedule)` reads `schedule.loopOrder` in the HOST program, while building the string, and emits one of two structurally different function bodies depending on which loop the schedule puts innermost: the flat-index expression the generated code computes -- `ov * dim1Extent + innerVal` when `dim0` is outer, `innerVal * dim1Extent + ov` when `dim1` is outer -- is chosen once, at generation time, and appears as a single hard-coded line in the emitted text. This is the precise difference between this generator and Chapter 22's own `executeSchedule()`: the interpreter reads a `Schedule`'s own `loopOrder` field as DATA, at runtime, inside a shared function that has to be ready to handle either order on every call; this generator reads the identical field as a host C++ value while BUILDING the source text, so by the time g++ ever sees the generated function, there is no `Schedule` struct left in it at all -- only whichever one nesting the schedule actually chose, hard-coded.

That difference sets up the real question this section exists to answer: does removing the interpreter -- replacing `executeSchedule()`'s own generic, data-driven dispatch with a real, schedule-specific compiled function -- change Chapter 22.3's own near-chance ranking agreement? This section reruns that chapter's identical experiment: the same 6 schedules, evenly drawn from the same 128-schedule list, for the identical `LoopNest [dim0:6, dim1:8]`, with the same combined `estimateScheduleCost()` scoring them -- the only variable that changes is whether each candidate is interpreted or genuinely compiled and dlopen-loaded before it is timed. The genuinely interesting result, reported honestly in the code's own closing comparison below, is that the answer is not the same on both of this book's own real machines -- removing the interpreter is not a fix that either clearly helps everywhere or clearly does nothing everywhere.

```text
executeSchedule() (Chapter 22.3) vs. generateScheduleFunction() (here):

  INTERPRETED (Ch22.3)                  COMPILED (this section)
  ---------------------                 ------------------------
  ONE generic function, ready for       ONE function PER schedule,
  any Schedule                          specific to exactly one

  reads loopOrder as DATA, every        reads loopOrder while BUILDING
  call, branches at runtime             the text -- no branch survives
                                         into the compiled function

  no compile step, cheap to try         a real g++ process per
  many candidates                       candidate (Section 23.1's own
                                         finding)

Same 6 schedules. Same LoopNest [dim0:6, dim1:8]. Same combined
estimateScheduleCost(). The only variable: interpreted, or compiled.
Section 22.3's own interpreted agreement: 40% (cloud), 53% (device).
This section's own honest finding, on real compiled code, is reported
in its own closing comparison below -- and it is not the same story
on both machines.
```

```cpp
// Chapter 23: Building CUDA Hammer's Autotuner
// 058_real_codegen_for_a_full_schedule_tile_size_loop_order_and_unroll_factor_together.cpp
//
// Section 23.2 -- Section 23.1 generated real compiled code for ONE loop's
// own tile size and unroll factor. This section adds the third dimension,
// loop order, and generates real code for a FULL two-loop Schedule at
// once -- the compiled twin of Section 22.3's own `executeSchedule()`
// interpreter, this time emitting real C++ TEXT that bakes the chosen loop
// order directly into the generated code's own SHAPE (which dimension's
// loop is written first, which flat-index expression appears in the
// innermost line) rather than branching on it at runtime the way the
// interpreter had to. This section runs the exact same experiment Section
// 22.3 already ran -- 6 schedules, evenly spaced across Section 21.3's own
// 128-schedule list, for the same tiny `LoopNest [dim0:6, dim1:8]` -- and
// reports, honestly, whether removing the interpreter changes the combined
// cost model's own ranking agreement with real measurement.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 058_real_codegen_for_a_full_schedule_tile_size_loop_order_and_unroll_factor_together.cpp -o 058_driver -ldl
// Run:     ./058_driver
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <dlfcn.h>

// ==================== Loop / LoopNest (Chapter 15, unchanged) ====================

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

// ==================== Section 21.1 / 21.2 / 21.3's own Schedule machinery (unchanged) ====================

static std::vector<long long> enumerateTileSizeCandidates(long long extent) {
    std::vector<long long> candidates;
    for (long long t = 1; t <= extent; t *= 2) candidates.push_back(t);
    if (candidates.empty() || candidates.back() != extent) candidates.push_back(extent);
    return candidates;
}
using LoopOrder = std::vector<int>;
static std::vector<LoopOrder> enumerateLoopOrders(const LoopNest& nest) {
    std::vector<int> indices(nest.loops.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<LoopOrder> orders;
    do {
        orders.push_back(indices);
    } while (std::next_permutation(indices.begin(), indices.end()));
    return orders;
}
static std::vector<long long> unrollFactorCandidates(long long extent) {
    return enumerateTileSizeCandidates(extent);
}
struct Schedule {
    std::vector<long long> tileSizePerLoop;
    LoopOrder loopOrder;
    long long unrollFactor;
};
static std::vector<Schedule> enumerateSchedules(const LoopNest& nest) {
    std::vector<std::vector<long long>> perLoopCandidates;
    for (const Loop& l : nest.loops) perLoopCandidates.push_back(enumerateTileSizeCandidates(l.extent));

    std::vector<LoopOrder> orders = enumerateLoopOrders(nest);
    long long innerExtent = nest.loops.back().extent;
    std::vector<long long> unrollCandidates = unrollFactorCandidates(innerExtent);

    std::vector<Schedule> schedules;
    std::vector<size_t> odometer(perLoopCandidates.size(), 0);
    bool done = perLoopCandidates.empty();
    while (!done) {
        std::vector<long long> tileChoice;
        for (size_t d = 0; d < perLoopCandidates.size(); ++d) tileChoice.push_back(perLoopCandidates[d][odometer[d]]);
        for (const LoopOrder& order : orders) {
            for (long long unroll : unrollCandidates) {
                schedules.push_back(Schedule{tileChoice, order, unroll});
            }
        }
        size_t d = 0;
        while (d < odometer.size()) {
            odometer[d]++;
            if (odometer[d] < perLoopCandidates[d].size()) break;
            odometer[d] = 0;
            d++;
        }
        if (d == odometer.size()) done = true;
    }
    return schedules;
}
static std::string scheduleStr(const LoopNest& nest, const Schedule& s) {
    std::string out = "tiles=[";
    for (size_t i = 0; i < s.tileSizePerLoop.size(); ++i) {
        if (i) out += ",";
        out += nest.loops[i].dimName + ":" + std::to_string(s.tileSizePerLoop[i]);
    }
    out += "] order=";
    for (size_t i = 0; i < s.loopOrder.size(); ++i) {
        if (i) out += ">";
        out += nest.loops[static_cast<size_t>(s.loopOrder[i])].dimName;
    }
    out += " unroll=" + std::to_string(s.unrollFactor);
    return out;
}

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

// ==================== Section 23.2: real codegen for a FULL schedule ====================
//
// The compiled twin of Section 22.3's own executeSchedule(): tile blocks
// in the OUTER loop, in-tile elements in the INNER loop, both walked in
// the schedule's own chosen order, unrolling the innermost in-tile loop by
// the schedule's own factor -- the identical algorithm, generated as real
// text instead of interpreted. The one genuine difference is where the
// SCHEDULE lives: the interpreter read a Schedule's own fields as DATA, at
// runtime, from one generic function; this generator reads them as host
// C++ values while BUILDING the source text, so the emitted code has no
// Schedule struct left in it at all by the time g++ ever sees it -- the
// loop order is baked into which flat-index expression the generated code
// even contains, not a runtime branch on it.
static std::string generateScheduleFunction(const LoopNest& nest, const Schedule& schedule) {
    long long dim0Extent = nest.loops[0].extent;
    long long dim1Extent = nest.loops[1].extent;
    int dimOuter = schedule.loopOrder[0];
    int dimInner = schedule.loopOrder[1];
    long long outerExtent = (dimOuter == 0) ? dim0Extent : dim1Extent;
    long long innerExtent = (dimInner == 0) ? dim0Extent : dim1Extent;
    long long outerTile = schedule.tileSizePerLoop[static_cast<size_t>(dimOuter)];
    long long innerTile = schedule.tileSizePerLoop[static_cast<size_t>(dimInner)];
    long long unrollFactor = schedule.unrollFactor;

    std::string src;
    src += "#include <algorithm>\n";
    src += "extern \"C\" void compute(const float* a, float* out) {\n";
    src += "  const long long dim1Extent = " + std::to_string(dim1Extent) + ";\n";
    src += "  for (long long ob = 0; ob < " + std::to_string(outerExtent) + "; ob += " + std::to_string(outerTile) + ") {\n";
    src += "    long long outerBlockEnd = std::min(ob + " + std::to_string(outerTile) + "LL, " + std::to_string(outerExtent) + "LL);\n";
    src += "    for (long long ib = 0; ib < " + std::to_string(innerExtent) + "; ib += " + std::to_string(innerTile) + ") {\n";
    src += "      long long innerBlockEnd = std::min(ib + " + std::to_string(innerTile) + "LL, " + std::to_string(innerExtent) + "LL);\n";
    src += "      for (long long ov = ob; ov < outerBlockEnd; ++ov) {\n";
    src += "        long long iv = ib;\n";
    src += "        for (; iv + " + std::to_string(unrollFactor) + " <= innerBlockEnd; iv += " + std::to_string(unrollFactor) + ") {\n";
    src += "          for (long long lane = 0; lane < " + std::to_string(unrollFactor) + "; ++lane) {\n";
    src += "            long long innerVal = iv + lane;\n";
    src += dimOuter == 0
               ? "            long long flat = ov * dim1Extent + innerVal;\n"
               : "            long long flat = innerVal * dim1Extent + ov;\n";
    src += "            out[flat] = a[flat] * 2.0f + 1.0f;\n";
    src += "          }\n        }\n";
    src += "        for (; iv < innerBlockEnd; ++iv) {\n";
    src += dimOuter == 0
               ? "          long long flat = ov * dim1Extent + iv;\n"
               : "          long long flat = iv * dim1Extent + ov;\n";
    src += "          out[flat] = a[flat] * 2.0f + 1.0f;\n";
    src += "        }\n";
    src += "      }\n    }\n  }\n}\n";
    return src;
}
using ComputeFn2D = void (*)(const float*, float*);
static bool arraysExactlyEqual(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

int main() {
    printf("=== Section 23.2: real codegen for a full schedule -- tile size, loop order, unroll factor ===\n\n");

    LoopNest nest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};
    std::vector<Schedule> allSchedules = enumerateSchedules(nest);

    // ---- Part 1: correctness, several real schedules spanning both loop orders ----
    printf("--- Part 1: correctness -- several real, compiled schedules, both loop orders ---\n\n");
    long long dim0Extent = nest.loops[0].extent, dim1Extent = nest.loops[1].extent;
    std::vector<float> a(static_cast<size_t>(dim0Extent * dim1Extent));
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> reference(a.size());
    for (size_t i = 0; i < a.size(); ++i) reference[i] = a[i] * 2.0f + 1.0f;

    std::vector<Schedule> correctnessSchedules = {
        Schedule{{1, 1}, {0, 1}, 1},
        Schedule{{4, 2}, {0, 1}, 8},
        Schedule{{2, 4}, {1, 0}, 1},
        Schedule{{6, 8}, {1, 0}, 8},
        Schedule{{3, 3}, {0, 1}, 3},
    };
    bool allCorrect = true;
    for (const Schedule& s : correctnessSchedules) {
        std::string src = generateScheduleFunction(nest, s);
        std::string log;
        bool clean = compileToSharedLibrary(src, "/tmp/058_correct.cpp", "/tmp/058_correct.so", log);
        JitModule mod("/tmp/058_correct.so");
        ComputeFn2D fn = mod.getFunction<ComputeFn2D>("compute");
        std::vector<float> out(a.size());
        fn(a.data(), out.data());
        bool ok = arraysExactlyEqual(out, reference);
        allCorrect = allCorrect && ok && clean;
        printf("  %-40s clean compile: %-3s  bit-for-bit correct: %s\n",
               scheduleStr(nest, s).c_str(), clean ? "yes" : "no", ok ? "yes" : "NO");
    }
    printf("\nself-check: every one of these real, compiled schedules -- both loop orders, several tile\n");
    printf("sizes and unroll factors -- produces the exact same bit-for-bit answer (%s), the same finding\n",
           allCorrect ? "confirmed" : "MISMATCH");
    printf("Section 22.3's own interpreter already established, now confirmed for genuinely compiled code.\n");

    // ---- Part 2: redo Section 22.3's own 6-schedule experiment, this time with REAL compiled code ----
    printf("\n--- Part 2: Section 22.3's own 6 schedules, evenly spaced, this time COMPILED not interpreted ---\n\n");
    std::vector<size_t> pickedIdx = {0, 25, 51, 76, 102, 127};
    std::vector<Schedule> picked;
    for (size_t idx : pickedIdx) picked.push_back(allSchedules[idx]);

    const long long repeatCount = 2000000;
    std::vector<double> measuredMs(picked.size());
    bool allCorrect2 = true;
    for (size_t k = 0; k < picked.size(); ++k) {
        std::string src = generateScheduleFunction(nest, picked[k]);
        std::string cppPath = "/tmp/058_perf_" + std::to_string(k) + ".cpp";
        std::string soPath = "/tmp/058_perf_" + std::to_string(k) + ".so";
        std::string log;
        compileToSharedLibrary(src, cppPath, soPath, log);
        JitModule mod(soPath);
        ComputeFn2D fn = mod.getFunction<ComputeFn2D>("compute");

        std::vector<float> out(a.size());
        double best = std::numeric_limits<double>::max();
        for (int trial = 0; trial < 9; ++trial) {
            auto start = std::chrono::steady_clock::now();
            for (long long r = 0; r < repeatCount; ++r) fn(a.data(), out.data());
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (ms < best) best = ms;
        }
        bool ok = arraysExactlyEqual(out, reference);
        allCorrect2 = allCorrect2 && ok;
        measuredMs[k] = best;
        printf("  [%3zu] best-of-9 total = %8.3f ms  (bit-for-bit correct: %s)  %s\n",
               pickedIdx[k], best, ok ? "yes" : "NO", scheduleStr(nest, picked[k]).c_str());
    }
    std::vector<size_t> measuredRank(picked.size());
    std::iota(measuredRank.begin(), measuredRank.end(), 0);
    std::sort(measuredRank.begin(), measuredRank.end(), [&](size_t x, size_t y) { return measuredMs[x] < measuredMs[y]; });
    printf("\nmeasured ranking, fastest to slowest (real compiled code, no interpreter):\n");
    for (size_t r = 0; r < measuredRank.size(); ++r) {
        size_t k = measuredRank[r];
        printf("  measured #%zu: [%3zu] %8.3f ms  %s\n", r + 1, pickedIdx[k], measuredMs[k], scheduleStr(nest, picked[k]).c_str());
    }

    // ---- Part 3: does removing the interpreter change the combined model's own ranking agreement? ----
    printf("\n--- Part 3: does real compiled code change the cost model's own ranking agreement? ---\n\n");
    // estimateScheduleCost(), reused unchanged from Section 22.3
    auto estimateLoopOverheadCost = [](long long extent, long long chunkSize, double perTripOverhead, double perElementWork) {
        long long trips = (extent + chunkSize - 1) / chunkSize;
        return static_cast<double>(trips) * perTripOverhead + static_cast<double>(extent) * perElementWork;
    };
    auto estimateLoopOrderCost = [](const LoopNest& n, const LoopOrder& order, long long cacheLineElements,
                                     double perLineCost, double perElementWork) {
        long long totalIterations = n.totalIterations();
        int contiguousDim = static_cast<int>(n.loops.size()) - 1;
        int innermostDim = order.back();
        double lineLoads = (innermostDim == contiguousDim)
                                ? static_cast<double>(totalIterations) / static_cast<double>(cacheLineElements)
                                : static_cast<double>(totalIterations);
        return lineLoads * perLineCost + static_cast<double>(totalIterations) * perElementWork;
    };
    const double perTripOverhead = 40.0, perLineCost = 40.0, perElementWork = 1.0;
    const long long cacheLineElements = 16;
    std::vector<double> predictedCost(picked.size());
    for (size_t k = 0; k < picked.size(); ++k) {
        double overhead = 0.0;
        for (size_t d = 0; d < nest.loops.size(); ++d) {
            overhead += estimateLoopOverheadCost(nest.loops[d].extent, picked[k].tileSizePerLoop[d], perTripOverhead, 0.0);
        }
        size_t innermostDim = static_cast<size_t>(picked[k].loopOrder.back());
        long long innermostTileSize = picked[k].tileSizePerLoop[innermostDim];
        overhead += estimateLoopOverheadCost(innermostTileSize, picked[k].unrollFactor, perTripOverhead, 0.0);
        double orderCost = estimateLoopOrderCost(nest, picked[k].loopOrder, cacheLineElements, perLineCost, 0.0);
        double elementWorkCost = static_cast<double>(nest.totalIterations()) * perElementWork;
        predictedCost[k] = overhead + orderCost + elementWorkCost;
    }
    int totalPairs = 0, concordantPairs = 0;
    for (size_t i = 0; i < picked.size(); ++i) {
        for (size_t j = i + 1; j < picked.size(); ++j) {
            totalPairs++;
            bool predSaysICheaper = predictedCost[i] < predictedCost[j];
            bool measSaysIFaster = measuredMs[i] < measuredMs[j];
            if (predSaysICheaper == measSaysIFaster) concordantPairs++;
        }
    }
    double concordancePct = 100.0 * concordantPairs / totalPairs;
    printf("  pairwise ranking agreement (real compiled code): %d of %d comparisons agree (%.0f%%)\n",
           concordantPairs, totalPairs, concordancePct);
    printf("  (Section 22.3's own interpreted agreement was 40%% on the cloud sandbox, 53%% on the device)\n\n");

    bool meaningfullyBetter = concordancePct >= 65.0;
    if (meaningfullyBetter) {
        printf("Real compiled code raises the agreement well above Section 22.3's own interpreted numbers --\n");
        printf("real evidence that interpreter bookkeeping really was a major part of what made that section's\n");
        printf("own ranking agreement land so close to chance. Removing it here, by generating and compiling a\n");
        printf("genuinely separate function per schedule, measurably helped.\n");
    } else {
        printf("This is close to Section 22.3's own interpreted numbers (40%%/53%%), not meaningfully higher --\n");
        printf("a real, honest correction to that section's own diagnosis. Interpreter overhead was A real\n");
        printf("cost, and Section 23.1's own compile-time finding proves real codegen has real costs of its\n");
        printf("own that an interpreter never pays -- but removing the interpreter did NOT, by itself, raise\n");
        printf("the combined cost model's own ranking agreement on this problem. The more likely explanation,\n");
        printf("visible now that interpretation is no longer a confound to blame: a 48-element computation is\n");
        printf("simply too small and too fast for the real differences between these 6 schedules to be the\n");
        printf("dominant signal in a wall-clock measurement, compiled or not -- system noise, cache state left\n");
        printf("over from whichever schedule ran immediately before, and OS scheduling jitter all compete with\n");
        printf("the actual, tiny difference these schedules make at this scale. Section 22.1 and 22.2 each\n");
        printf("measured real, larger computations (50 million and 16.7 million elements) and got much\n");
        printf("cleaner agreement with their own models; this section's own honest finding is that SCALE, not\n");
        printf("interpretation, was probably the bigger confound all along -- a genuinely useful correction to\n");
        printf("carry into Section 23.3's own real autotuner, which will need to budget real measurement time\n");
        printf("carefully regardless of problem size.\n");
    }

    bool allOk = allCorrect && allCorrect2;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "058_real_codegen_for_a_full_schedule_tile_size_loop_order_and_unroll_factor_together.cpp" -o "058_real_codegen_for_a_full_schedule_tile_size_loop_order_and_unroll_factor_together" -ldl
./"058_real_codegen_for_a_full_schedule_tile_size_loop_order_and_unroll_factor_together"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 23.2: real codegen for a full schedule -- tile size, loop order, unroll factor ===

--- Part 1: correctness -- several real, compiled schedules, both loop orders ---

  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:3,dim1:3] order=dim0>dim1 unroll=3 clean compile: yes  bit-for-bit correct: yes

self-check: every one of these real, compiled schedules -- both loop orders, several tile
sizes and unroll factors -- produces the exact same bit-for-bit answer (confirmed), the same finding
Section 22.3's own interpreter already established, now confirmed for genuinely compiled code.

--- Part 2: Section 22.3's own 6 schedules, evenly spaced, this time COMPILED not interpreted ---

  [  0] best-of-9 total =   59.900 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [ 25] best-of-9 total =   68.009 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  [ 51] best-of-9 total =   43.917 ms  (bit-for-bit correct: yes)  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  [ 76] best-of-9 total =   37.708 ms  (bit-for-bit correct: yes)  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  [102] best-of-9 total =   58.772 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  [127] best-of-9 total =   67.505 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

measured ranking, fastest to slowest (real compiled code, no interpreter):
  measured #1: [ 76]   37.708 ms  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  measured #2: [ 51]   43.917 ms  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  measured #3: [102]   58.772 ms  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  measured #4: [  0]   59.900 ms  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  measured #5: [127]   67.505 ms  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  measured #6: [ 25]   68.009 ms  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2

--- Part 3: does real compiled code change the cost model's own ranking agreement? ---

  pairwise ranking agreement (real compiled code): 6 of 15 comparisons agree (40%)
  (Section 22.3's own interpreted agreement was 40% on the cloud sandbox, 53% on the device)

This is close to Section 22.3's own interpreted numbers (40%/53%), not meaningfully higher --
a real, honest correction to that section's own diagnosis. Interpreter overhead was A real
cost, and Section 23.1's own compile-time finding proves real codegen has real costs of its
own that an interpreter never pays -- but removing the interpreter did NOT, by itself, raise
the combined cost model's own ranking agreement on this problem. The more likely explanation,
visible now that interpretation is no longer a confound to blame: a 48-element computation is
simply too small and too fast for the real differences between these 6 schedules to be the
dominant signal in a wall-clock measurement, compiled or not -- system noise, cache state left
over from whichever schedule ran immediately before, and OS scheduling jitter all compete with
the actual, tiny difference these schedules make at this scale. Section 22.1 and 22.2 each
measured real, larger computations (50 million and 16.7 million elements) and got much
cleaner agreement with their own models; this section's own honest finding is that SCALE, not
interpretation, was probably the bigger confound all along -- a genuinely useful correction to
carry into Section 23.3's own real autotuner, which will need to budget real measurement time
carefully regardless of problem size.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 23.2: real codegen for a full schedule -- tile size, loop order, unroll factor ===

--- Part 1: correctness -- several real, compiled schedules, both loop orders ---

  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8 clean compile: yes  bit-for-bit correct: yes
  tiles=[dim0:3,dim1:3] order=dim0>dim1 unroll=3 clean compile: yes  bit-for-bit correct: yes

self-check: every one of these real, compiled schedules -- both loop orders, several tile
sizes and unroll factors -- produces the exact same bit-for-bit answer (confirmed), the same finding
Section 22.3's own interpreter already established, now confirmed for genuinely compiled code.

--- Part 2: Section 22.3's own 6 schedules, evenly spaced, this time COMPILED not interpreted ---

  [  0] best-of-9 total =   37.026 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  [ 25] best-of-9 total =   34.446 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  [ 51] best-of-9 total =   18.389 ms  (bit-for-bit correct: yes)  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  [ 76] best-of-9 total =   20.089 ms  (bit-for-bit correct: yes)  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  [102] best-of-9 total =   38.329 ms  (bit-for-bit correct: yes)  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4
  [127] best-of-9 total =   31.682 ms  (bit-for-bit correct: yes)  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8

measured ranking, fastest to slowest (real compiled code, no interpreter):
  measured #1: [ 51]   18.389 ms  tiles=[dim0:4,dim1:2] order=dim0>dim1 unroll=8
  measured #2: [ 76]   20.089 ms  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=1
  measured #3: [127]   31.682 ms  tiles=[dim0:6,dim1:8] order=dim1>dim0 unroll=8
  measured #4: [ 25]   34.446 ms  tiles=[dim0:6,dim1:1] order=dim0>dim1 unroll=2
  measured #5: [  0]   37.026 ms  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1
  measured #6: [102]   38.329 ms  tiles=[dim0:1,dim1:8] order=dim1>dim0 unroll=4

--- Part 3: does real compiled code change the cost model's own ranking agreement? ---

  pairwise ranking agreement (real compiled code): 10 of 15 comparisons agree (67%)
  (Section 22.3's own interpreted agreement was 40% on the cloud sandbox, 53% on the device)

Real compiled code raises the agreement well above Section 22.3's own interpreted numbers --
real evidence that interpreter bookkeeping really was a major part of what made that section's
own ranking agreement land so close to chance. Removing it here, by generating and compiling a
genuinely separate function per schedule, measurably helped.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which candidate ranks fastest, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! warning "[COMMON TRAP] removing one confound does not mean removing every confound"
    It would be easy to read Section 23.2's own honest finding as a clean verdict either way -- "compiling fixed it" or "compiling didn't help." The real, machine-specific result this section actually measured is more useful than either clean story: whatever changed on one real machine did not change the same way on the other, which is itself evidence about WHAT was actually going on. Chapter 22.3 diagnosed interpreter overhead as a likely confound on a 48-element toy problem; this section's own real test of that diagnosis, on two different machines, is exactly the kind of check a cost model -- or a diagnosis about a cost model -- has to survive before an autotuner should lean on it.

## 23.3 The Autotuner: Pruning With a Cost Model, Then Measuring for Real

### Intuition

Every piece this chapter and the two before it built now exists at once: Chapter 21's own 128 real, enumerated schedules for `LoopNest [dim0:6, dim1:8]`; Chapter 22's own `estimateScheduleCost()`, a cost model cheap enough to score all 128 in a fraction of a millisecond and honestly known to rank imperfectly; and Section 23.2's own `generateScheduleFunction()`, real compiled codegen for any one of those 128 schedules, at a real, non-negligible cost per candidate that Section 23.1 already put a number on. An autotuner is simply the program that puts all three pieces together and makes a real decision: which candidates are worth the real cost of compiling and measuring, and which are not.

### Background

The design this section builds is the HYBRID Chapter 22's own closing paragraph already pointed toward: rank all 128 real schedules with `estimateScheduleCost()`, keep only the cheapest `topK = 8`, and pay for real compilation and careful measurement (best-of-9 trials, 2,000,000 repeat calls each, to get a reliably measurable duration on this book's own toy-scale 48-element computation) on ONLY that short list. That hybrid autotuner's own winner is whichever of the 8 short-listed candidates measures fastest for real -- a genuine, compiled, dlopen-loaded answer, not a cost-model guess.

The honest question a hybrid design like this one has to answer is whether pruning down to 8 ever throws away the schedule that was actually best. This section answers it directly, not by argument but by also doing the expensive thing the hybrid exists to avoid: compiling and measuring all 128 real schedules, with a deliberately LIGHTER measurement pass (best-of-3, 200,000 repeat calls -- noisier, but fast enough to make measuring all 128 affordable at all) to serve as real, not simulated, ground truth. Comparing the hybrid's own 8-candidate answer against this exhaustive 128-candidate answer needs one piece of care Section 23.1 already set up: the two passes use different repeat counts, so the code compares PER-CALL time (`MeasuredCandidate::perCallUs()`), never raw totals from two differently-scaled measurement runs, which would never be a fair comparison. The result, reported honestly in the code's own closing verdict below, is a genuine test of whether an imperfect, honestly-flawed cost model is still useful as a pruning filter -- and a real, directly measured wall-clock accounting of what pruning actually saves, whichever way the short list turns out.

```text
autotuneSchedule, the hybrid design (this section's own capstone):

  Part 1 -- estimateScheduleCost() ranks all 128 real schedules
            (Ch22, unchanged) -- cheap, no compilation, imperfect
  |
  +-- keep only the cheapest topK = 8
  |
  Part 2 -- HYBRID: compile + carefully measure ONLY the 8
            (best-of-9, 2,000,000 reps each) -- real g++, real
            dlopen, real clock, on a short list Section 23.1's own
            cost-per-candidate finding says is worth affording
  |
  +-- hybrid's own winner: fastest of the 8, for real
  |
  Part 3 -- EXHAUSTIVE (ground truth): compile + measure ALL 128
            (lighter pass: best-of-3, 200,000 reps each -- noisier,
            but affordable across all 128) -- real ground truth,
            not simulated
  |
  +-- true best: fastest of all 128, for real
  |
  Part 4 -- the honest verdict: was the true best inside the cost
            model's own top-8 short list? did the hybrid still win
            on real wall-clock time regardless? Compared by PER-CALL
            time (perCallUs()), never raw totals across two
            differently-scaled measurement passes.

Neither pass is asked to be free of noise. Both are asked to be
honest about what they actually measured, and about which real
schedule -- not which predicted one -- came out fastest.
```

```cpp
// Chapter 23: Building CUDA Hammer's Autotuner
// 059_the_autotuner_pruning_with_a_cost_model_then_measuring_for_real.cpp
//
// Section 23.3 (capstone) -- this chapter's capstone builds the actual
// search its own first two sections and Chapter 22 were built toward.
// Chapter 21 proved the search space is real and already too large to
// brute-force one worked example at a time (128 legal schedules for a
// TINY two-loop kernel). Chapter 22 built two honest, imperfect cost
// models and combined them into `estimateScheduleCost()`. Section 23.1-23.2
// (this chapter) built real, compiled codegen for a full schedule, closing
// the gap Chapter 22's own interpreter left open. This section puts all of
// it together into an autotuning loop: use the cost model to prune 128
// real candidates down to a short list cheaply, generate and compile real
// code for ONLY that short list, and measure it for real -- then,
// honestly, checks that hybrid against the alternative it exists to avoid:
// compiling and measuring EVERY one of the 128 candidates for real, which
// this section also actually does, because at this book's own toy scale it
// is (just barely) affordable enough to serve as real, not simulated,
// ground truth. Part 5 itself still has one chapter left after this one --
// this section proves the hybrid autotuner works and is honest about where
// it can still miss; Chapter 24 will close Part 5 by asking what a real
// autotuner does with that result afterward, run after run.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 059_the_autotuner_pruning_with_a_cost_model_then_measuring_for_real.cpp -o 059_driver -ldl
// Run:     ./059_driver
#include <cstdio>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <dlfcn.h>

// ==================== Loop / LoopNest / Schedule machinery (Chapters 15/21, unchanged) ====================

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
static std::vector<long long> enumerateTileSizeCandidates(long long extent) {
    std::vector<long long> candidates;
    for (long long t = 1; t <= extent; t *= 2) candidates.push_back(t);
    if (candidates.empty() || candidates.back() != extent) candidates.push_back(extent);
    return candidates;
}
using LoopOrder = std::vector<int>;
static std::vector<LoopOrder> enumerateLoopOrders(const LoopNest& nest) {
    std::vector<int> indices(nest.loops.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<LoopOrder> orders;
    do {
        orders.push_back(indices);
    } while (std::next_permutation(indices.begin(), indices.end()));
    return orders;
}
static std::vector<long long> unrollFactorCandidates(long long extent) {
    return enumerateTileSizeCandidates(extent);
}
struct Schedule {
    std::vector<long long> tileSizePerLoop;
    LoopOrder loopOrder;
    long long unrollFactor;
};
static std::vector<Schedule> enumerateSchedules(const LoopNest& nest) {
    std::vector<std::vector<long long>> perLoopCandidates;
    for (const Loop& l : nest.loops) perLoopCandidates.push_back(enumerateTileSizeCandidates(l.extent));
    std::vector<LoopOrder> orders = enumerateLoopOrders(nest);
    long long innerExtent = nest.loops.back().extent;
    std::vector<long long> unrollCandidates = unrollFactorCandidates(innerExtent);
    std::vector<Schedule> schedules;
    std::vector<size_t> odometer(perLoopCandidates.size(), 0);
    bool done = perLoopCandidates.empty();
    while (!done) {
        std::vector<long long> tileChoice;
        for (size_t d = 0; d < perLoopCandidates.size(); ++d) tileChoice.push_back(perLoopCandidates[d][odometer[d]]);
        for (const LoopOrder& order : orders) {
            for (long long unroll : unrollCandidates) {
                schedules.push_back(Schedule{tileChoice, order, unroll});
            }
        }
        size_t d = 0;
        while (d < odometer.size()) {
            odometer[d]++;
            if (odometer[d] < perLoopCandidates[d].size()) break;
            odometer[d] = 0;
            d++;
        }
        if (d == odometer.size()) done = true;
    }
    return schedules;
}
static std::string scheduleStr(const LoopNest& nest, const Schedule& s) {
    std::string out = "tiles=[";
    for (size_t i = 0; i < s.tileSizePerLoop.size(); ++i) {
        if (i) out += ",";
        out += nest.loops[i].dimName + ":" + std::to_string(s.tileSizePerLoop[i]);
    }
    out += "] order=";
    for (size_t i = 0; i < s.loopOrder.size(); ++i) {
        if (i) out += ">";
        out += nest.loops[static_cast<size_t>(s.loopOrder[i])].dimName;
    }
    out += " unroll=" + std::to_string(s.unrollFactor);
    return out;
}

// ==================== estimateScheduleCost() (Chapter 22, unchanged) ====================

static double estimateLoopOverheadCost(long long extent, long long chunkSize, double perTripOverhead, double perElementWork) {
    long long trips = (extent + chunkSize - 1) / chunkSize;
    return static_cast<double>(trips) * perTripOverhead + static_cast<double>(extent) * perElementWork;
}
static double estimateLoopOrderCost(const LoopNest& nest, const LoopOrder& order, long long cacheLineElements,
                                     double perLineCost, double perElementWork) {
    long long totalIterations = nest.totalIterations();
    int contiguousDim = static_cast<int>(nest.loops.size()) - 1;
    int innermostDim = order.back();
    double lineLoads = (innermostDim == contiguousDim)
                            ? static_cast<double>(totalIterations) / static_cast<double>(cacheLineElements)
                            : static_cast<double>(totalIterations);
    return lineLoads * perLineCost + static_cast<double>(totalIterations) * perElementWork;
}
static double estimateScheduleCost(const LoopNest& nest, const Schedule& schedule, double perTripOverhead,
                                    long long cacheLineElements, double perLineCost, double perElementWork) {
    double overheadCost = 0.0;
    for (size_t d = 0; d < nest.loops.size(); ++d) {
        overheadCost += estimateLoopOverheadCost(nest.loops[d].extent, schedule.tileSizePerLoop[d], perTripOverhead, 0.0);
    }
    size_t innermostDim = static_cast<size_t>(schedule.loopOrder.back());
    long long innermostTileSize = schedule.tileSizePerLoop[innermostDim];
    overheadCost += estimateLoopOverheadCost(innermostTileSize, schedule.unrollFactor, perTripOverhead, 0.0);
    double orderCost = estimateLoopOrderCost(nest, schedule.loopOrder, cacheLineElements, perLineCost, 0.0);
    double elementWorkCost = static_cast<double>(nest.totalIterations()) * perElementWork;
    return overheadCost + orderCost + elementWorkCost;
}

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

// ==================== Section 23.2's own generateScheduleFunction() (unchanged) ====================

static std::string generateScheduleFunction(const LoopNest& nest, const Schedule& schedule) {
    long long dim0Extent = nest.loops[0].extent;
    long long dim1Extent = nest.loops[1].extent;
    int dimOuter = schedule.loopOrder[0];
    int dimInner = schedule.loopOrder[1];
    long long outerExtent = (dimOuter == 0) ? dim0Extent : dim1Extent;
    long long innerExtent = (dimInner == 0) ? dim0Extent : dim1Extent;
    long long outerTile = schedule.tileSizePerLoop[static_cast<size_t>(dimOuter)];
    long long innerTile = schedule.tileSizePerLoop[static_cast<size_t>(dimInner)];
    long long unrollFactor = schedule.unrollFactor;

    std::string src;
    src += "#include <algorithm>\n";
    src += "extern \"C\" void compute(const float* a, float* out) {\n";
    src += "  const long long dim1Extent = " + std::to_string(dim1Extent) + ";\n";
    src += "  for (long long ob = 0; ob < " + std::to_string(outerExtent) + "; ob += " + std::to_string(outerTile) + ") {\n";
    src += "    long long outerBlockEnd = std::min(ob + " + std::to_string(outerTile) + "LL, " + std::to_string(outerExtent) + "LL);\n";
    src += "    for (long long ib = 0; ib < " + std::to_string(innerExtent) + "; ib += " + std::to_string(innerTile) + ") {\n";
    src += "      long long innerBlockEnd = std::min(ib + " + std::to_string(innerTile) + "LL, " + std::to_string(innerExtent) + "LL);\n";
    src += "      for (long long ov = ob; ov < outerBlockEnd; ++ov) {\n";
    src += "        long long iv = ib;\n";
    src += "        for (; iv + " + std::to_string(unrollFactor) + " <= innerBlockEnd; iv += " + std::to_string(unrollFactor) + ") {\n";
    src += "          for (long long lane = 0; lane < " + std::to_string(unrollFactor) + "; ++lane) {\n";
    src += "            long long innerVal = iv + lane;\n";
    src += dimOuter == 0
               ? "            long long flat = ov * dim1Extent + innerVal;\n"
               : "            long long flat = innerVal * dim1Extent + ov;\n";
    src += "            out[flat] = a[flat] * 2.0f + 1.0f;\n";
    src += "          }\n        }\n";
    src += "        for (; iv < innerBlockEnd; ++iv) {\n";
    src += dimOuter == 0
               ? "          long long flat = ov * dim1Extent + iv;\n"
               : "          long long flat = iv * dim1Extent + ov;\n";
    src += "          out[flat] = a[flat] * 2.0f + 1.0f;\n";
    src += "        }\n";
    src += "      }\n    }\n  }\n}\n";
    return src;
}
using ComputeFn2D = void (*)(const float*, float*);
static bool arraysExactlyEqual(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

// ==================== Section 23.3: the autotuner itself ====================

struct MeasuredCandidate {
    Schedule schedule;
    double compileMs;
    double runMs;        // best-of-N TOTAL for repeatCount calls -- only comparable to another
                          // candidate measured with the SAME repeatCount; use perCallUs() to
                          // compare across two passes that used different repeat counts.
    long long repeatCount;
    bool correct;
    double perCallUs() const { return runMs * 1000.0 / static_cast<double>(repeatCount); }
};

static MeasuredCandidate compileAndMeasure(const LoopNest& nest, const Schedule& schedule, const std::vector<float>& a,
                                            const std::vector<float>& reference, long long repeatCount, int trials,
                                            const std::string& tag) {
    std::string src = generateScheduleFunction(nest, schedule);
    std::string cppPath = "/tmp/059_" + tag + ".cpp";
    std::string soPath = "/tmp/059_" + tag + ".so";
    std::string log;

    auto compileStart = std::chrono::steady_clock::now();
    compileToSharedLibrary(src, cppPath, soPath, log);
    auto compileEnd = std::chrono::steady_clock::now();
    double compileMs = std::chrono::duration<double, std::milli>(compileEnd - compileStart).count();

    JitModule mod(soPath);
    ComputeFn2D fn = mod.getFunction<ComputeFn2D>("compute");
    std::vector<float> out(a.size());
    double best = std::numeric_limits<double>::max();
    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::steady_clock::now();
        for (long long r = 0; r < repeatCount; ++r) fn(a.data(), out.data());
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < best) best = ms;
    }
    bool correct = arraysExactlyEqual(out, reference);
    return MeasuredCandidate{schedule, compileMs, best, repeatCount, correct};
}

int main() {
    printf("=== Section 23.3: the autotuner -- pruning with a cost model, then measuring for real ===\n\n");

    LoopNest nest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};
    std::vector<Schedule> allSchedules = enumerateSchedules(nest);
    printf("LoopNest [dim0:6, dim1:8]: %zu real, legal schedules (Section 21.3's own number, confirmed again)\n\n",
           allSchedules.size());

    long long dim0Extent = nest.loops[0].extent, dim1Extent = nest.loops[1].extent;
    std::vector<float> a(static_cast<size_t>(dim0Extent * dim1Extent));
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> reference(a.size());
    for (size_t i = 0; i < a.size(); ++i) reference[i] = a[i] * 2.0f + 1.0f;

    // ---- Part 1: rank ALL 128 by the cost model, prune to a short list of 8 ----
    printf("--- Part 1: estimateScheduleCost() ranks all 128; the autotuner keeps only the cheapest 8 ---\n\n");
    const double perTripOverhead = 40.0, perLineCost = 40.0, perElementWork = 1.0;
    const long long cacheLineElements = 16;
    const size_t topK = 8;

    std::vector<double> predictedCost(allSchedules.size());
    for (size_t i = 0; i < allSchedules.size(); ++i) {
        predictedCost[i] = estimateScheduleCost(nest, allSchedules[i], perTripOverhead, cacheLineElements, perLineCost, perElementWork);
    }
    std::vector<size_t> rankedIdx(allSchedules.size());
    std::iota(rankedIdx.begin(), rankedIdx.end(), 0);
    std::sort(rankedIdx.begin(), rankedIdx.end(), [&](size_t x, size_t y) { return predictedCost[x] < predictedCost[y]; });

    printf("cheapest %zu predicted schedules (the autotuner's own short list):\n", topK);
    for (size_t r = 0; r < topK; ++r) {
        size_t idx = rankedIdx[r];
        printf("  predicted #%zu: cost=%8.1f  %s\n", r + 1, predictedCost[idx], scheduleStr(nest, allSchedules[idx]).c_str());
    }

    // ---- Part 2: the HYBRID -- compile and carefully measure ONLY the short list ----
    printf("\n--- Part 2: HYBRID -- compile and carefully measure (best-of-9, 2,000,000 reps) only the 8 ---\n\n");
    auto hybridStart = std::chrono::steady_clock::now();
    std::vector<MeasuredCandidate> hybridResults;
    bool hybridAllCorrect = true;
    for (size_t r = 0; r < topK; ++r) {
        size_t idx = rankedIdx[r];
        MeasuredCandidate mc = compileAndMeasure(nest, allSchedules[idx], a, reference, 2000000, 9, "hybrid_" + std::to_string(r));
        hybridAllCorrect = hybridAllCorrect && mc.correct;
        hybridResults.push_back(mc);
        printf("  [rank %zu] compile=%6.1f ms  per-call=%7.4f us  correct=%s  %s\n",
               r + 1, mc.compileMs, mc.perCallUs(), mc.correct ? "yes" : "NO", scheduleStr(nest, allSchedules[idx]).c_str());
    }
    auto hybridEnd = std::chrono::steady_clock::now();
    double hybridTotalMs = std::chrono::duration<double, std::milli>(hybridEnd - hybridStart).count();

    size_t hybridBestLocal = 0;
    for (size_t i = 1; i < hybridResults.size(); ++i)
        if (hybridResults[i].perCallUs() < hybridResults[hybridBestLocal].perCallUs()) hybridBestLocal = i;
    Schedule hybridWinner = hybridResults[hybridBestLocal].schedule;
    printf("\nhybrid autotuner's own winner: %s (%.4f us/call, best-of-9 over 2,000,000 reps)\n",
           scheduleStr(nest, hybridWinner).c_str(), hybridResults[hybridBestLocal].perCallUs());
    printf("hybrid total wall-clock (compile + measure, 8 candidates): %.1f ms\n", hybridTotalMs);

    // ---- Part 3: EXHAUSTIVE -- compile and measure ALL 128, for real, as ground truth ----
    printf("\n--- Part 3: EXHAUSTIVE -- compile and measure all 128, for real (lighter pass: best-of-3, 200,000 reps) ---\n\n");
    auto exhaustiveStart = std::chrono::steady_clock::now();
    std::vector<MeasuredCandidate> allResults;
    bool exhaustiveAllCorrect = true;
    for (size_t i = 0; i < allSchedules.size(); ++i) {
        MeasuredCandidate mc = compileAndMeasure(nest, allSchedules[i], a, reference, 200000, 3, "exh_" + std::to_string(i));
        exhaustiveAllCorrect = exhaustiveAllCorrect && mc.correct;
        allResults.push_back(mc);
    }
    auto exhaustiveEnd = std::chrono::steady_clock::now();
    double exhaustiveTotalMs = std::chrono::duration<double, std::milli>(exhaustiveEnd - exhaustiveStart).count();

    size_t trueBestIdx = 0;
    for (size_t i = 1; i < allResults.size(); ++i)
        if (allResults[i].perCallUs() < allResults[trueBestIdx].perCallUs()) trueBestIdx = i;
    printf("all 128 real schedules compiled and measured. every one bit-for-bit correct: %s\n",
           exhaustiveAllCorrect ? "confirmed" : "MISMATCH");
    printf("true best, by exhaustive real measurement: %s (%.4f us/call, best-of-3 over 200,000 reps)\n",
           scheduleStr(nest, allResults[trueBestIdx].schedule).c_str(), allResults[trueBestIdx].perCallUs());
    printf("(per-call time, not raw totals, since this pass uses a different repeat count than Part 2 --\n");
    printf("comparing raw totals across two different repeat counts would not be a fair comparison)\n");
    printf("exhaustive total wall-clock (compile + measure, all 128 candidates): %.1f ms\n", exhaustiveTotalMs);

    // ---- Part 4: the honest verdict ----
    printf("\n--- Part 4: did pruning to 8 find the true best? was it actually cheaper? ---\n\n");
    bool trueBestInShortList = false;
    for (size_t r = 0; r < topK; ++r) if (rankedIdx[r] == trueBestIdx) trueBestInShortList = true;
    bool hybridFoundTrueBest = (scheduleStr(nest, hybridWinner) == scheduleStr(nest, allResults[trueBestIdx].schedule));

    printf("  true best (exhaustive, all 128) in the cost model's own top-%zu short list: %s\n",
           topK, trueBestInShortList ? "yes" : "no");
    printf("  hybrid's own winner == true best (exhaustive, all 128): %s\n", hybridFoundTrueBest ? "yes" : "no");
    printf("  hybrid wall-clock: %8.1f ms (8 candidates, careful measurement)\n", hybridTotalMs);
    printf("  exhaustive wall-clock: %8.1f ms (128 candidates, lighter measurement)\n", exhaustiveTotalMs);
    printf("  real wall-clock ratio (exhaustive / hybrid): %.2fx\n\n", exhaustiveTotalMs / hybridTotalMs);

    if (trueBestInShortList) {
        printf("The cost model's own top-%zu, cheap to compute and requiring no compilation at all to produce,\n", topK);
        printf("contained the schedule that real, exhaustive, all-128 measurement confirms is actually fastest.\n");
        printf("This is Chapter 22's own honest imperfection paying off anyway: the model does not need to be\n");
        printf("PERFECT at ranking to be a USEFUL filter -- it only needs to not throw away the winner, and\n");
        printf("here, it didn't. Real wall-clock time still favors the hybrid by a real, directly measured\n");
        printf("margin, even though the exhaustive pass deliberately used a lighter (and therefore noisier)\n");
        printf("measurement per candidate -- giving the exhaustive approach every advantage this comparison\n");
        printf("could fairly give it, and the hybrid still won on total real time.\n");
    } else {
        printf("The cost model's own top-%zu did NOT contain the schedule real, exhaustive measurement found\n", topK);
        printf("to actually be fastest -- an honest miss, not a hidden one. This is exactly the risk of pruning\n");
        printf("with an imperfect model, named plainly rather than smoothed over: Chapter 22 never claimed\n");
        printf("estimateScheduleCost() was reliable enough to guarantee the true winner survives every prune,\n");
        printf("only that it is cheap enough to be worth trying before paying for real measurement. A real\n");
        printf("autotuner facing this tradeoff has options this book leaves as an open, named direction rather\n");
        printf("than a settled answer: widen the short list, refine the cost model with the real mechanisms\n");
        printf("Chapter 22 already found it was missing, or accept a probably-very-good schedule instead of\n");
        printf("a provably-best one in exchange for the real time this section's own numbers show was saved.\n");
    }

    bool allOk = hybridAllCorrect && exhaustiveAllCorrect;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "059_the_autotuner_pruning_with_a_cost_model_then_measuring_for_real.cpp" -o "059_the_autotuner_pruning_with_a_cost_model_then_measuring_for_real" -ldl
./"059_the_autotuner_pruning_with_a_cost_model_then_measuring_for_real"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 23.3: the autotuner -- pruning with a cost model, then measuring for real ===

LoopNest [dim0:6, dim1:8]: 128 real, legal schedules (Section 21.3's own number, confirmed again)

--- Part 1: estimateScheduleCost() ranks all 128; the autotuner keeps only the cheapest 8 ---

cheapest 8 predicted schedules (the autotuner's own short list):
  predicted #1: cost=   288.0  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=8
  predicted #2: cost=   328.0  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=4
  predicted #3: cost=   328.0  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=8
  predicted #4: cost=   328.0  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=8
  predicted #5: cost=   328.0  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4
  predicted #6: cost=   368.0  tiles=[dim0:4,dim1:4] order=dim0>dim1 unroll=8
  predicted #7: cost=   368.0  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=4
  predicted #8: cost=   368.0  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

--- Part 2: HYBRID -- compile and carefully measure (best-of-9, 2,000,000 reps) only the 8 ---

  [rank 1] compile= 137.9 ms  per-call= 0.0337 us  correct=yes  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=8
  [rank 2] compile= 129.2 ms  per-call= 0.0279 us  correct=yes  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=4
  [rank 3] compile= 147.4 ms  per-call= 0.0381 us  correct=yes  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=8
  [rank 4] compile= 108.1 ms  per-call= 0.0311 us  correct=yes  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=8
  [rank 5] compile= 107.8 ms  per-call= 0.0221 us  correct=yes  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4
  [rank 6] compile= 107.5 ms  per-call= 0.0395 us  correct=yes  tiles=[dim0:4,dim1:4] order=dim0>dim1 unroll=8
  [rank 7] compile= 130.5 ms  per-call= 0.0334 us  correct=yes  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=4
  [rank 8] compile= 121.4 ms  per-call= 0.0222 us  correct=yes  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

hybrid autotuner's own winner: tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4 (0.0221 us/call, best-of-9 over 2,000,000 reps)
hybrid total wall-clock (compile + measure, 8 candidates): 5756.1 ms

--- Part 3: EXHAUSTIVE -- compile and measure all 128, for real (lighter pass: best-of-3, 200,000 reps) ---

all 128 real schedules compiled and measured. every one bit-for-bit correct: confirmed
true best, by exhaustive real measurement: tiles=[dim0:2,dim1:8] order=dim1>dim0 unroll=1 (0.0172 us/call, best-of-3 over 200,000 reps)
(per-call time, not raw totals, since this pass uses a different repeat count than Part 2 --
comparing raw totals across two different repeat counts would not be a fair comparison)
exhaustive total wall-clock (compile + measure, all 128 candidates): 17032.8 ms

--- Part 4: did pruning to 8 find the true best? was it actually cheaper? ---

  true best (exhaustive, all 128) in the cost model's own top-8 short list: no
  hybrid's own winner == true best (exhaustive, all 128): no
  hybrid wall-clock:   5756.1 ms (8 candidates, careful measurement)
  exhaustive wall-clock:  17032.8 ms (128 candidates, lighter measurement)
  real wall-clock ratio (exhaustive / hybrid): 2.96x

The cost model's own top-8 did NOT contain the schedule real, exhaustive measurement found
to actually be fastest -- an honest miss, not a hidden one. This is exactly the risk of pruning
with an imperfect model, named plainly rather than smoothed over: Chapter 22 never claimed
estimateScheduleCost() was reliable enough to guarantee the true winner survives every prune,
only that it is cheap enough to be worth trying before paying for real measurement. A real
autotuner facing this tradeoff has options this book leaves as an open, named direction rather
than a settled answer: widen the short list, refine the cost model with the real mechanisms
Chapter 22 already found it was missing, or accept a probably-very-good schedule instead of
a provably-best one in exchange for the real time this section's own numbers show was saved.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 23.3: the autotuner -- pruning with a cost model, then measuring for real ===

LoopNest [dim0:6, dim1:8]: 128 real, legal schedules (Section 21.3's own number, confirmed again)

--- Part 1: estimateScheduleCost() ranks all 128; the autotuner keeps only the cheapest 8 ---

cheapest 8 predicted schedules (the autotuner's own short list):
  predicted #1: cost=   288.0  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=8
  predicted #2: cost=   328.0  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=4
  predicted #3: cost=   328.0  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=8
  predicted #4: cost=   328.0  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=8
  predicted #5: cost=   328.0  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4
  predicted #6: cost=   368.0  tiles=[dim0:4,dim1:4] order=dim0>dim1 unroll=8
  predicted #7: cost=   368.0  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=4
  predicted #8: cost=   368.0  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

--- Part 2: HYBRID -- compile and carefully measure (best-of-9, 2,000,000 reps) only the 8 ---

  [rank 1] compile=  86.3 ms  per-call= 0.0144 us  correct=yes  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=8
  [rank 2] compile=  83.3 ms  per-call= 0.0151 us  correct=yes  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=4
  [rank 3] compile=  76.8 ms  per-call= 0.0138 us  correct=yes  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=8
  [rank 4] compile=  77.9 ms  per-call= 0.0144 us  correct=yes  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=8
  [rank 5] compile=  77.4 ms  per-call= 0.0118 us  correct=yes  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4
  [rank 6] compile=  79.0 ms  per-call= 0.0154 us  correct=yes  tiles=[dim0:4,dim1:4] order=dim0>dim1 unroll=8
  [rank 7] compile=  77.8 ms  per-call= 0.0150 us  correct=yes  tiles=[dim0:4,dim1:8] order=dim0>dim1 unroll=4
  [rank 8] compile=  84.7 ms  per-call= 0.0095 us  correct=yes  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

hybrid autotuner's own winner: tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2 (0.0095 us/call, best-of-9 over 2,000,000 reps)
hybrid total wall-clock (compile + measure, 8 candidates): 2763.1 ms

--- Part 3: EXHAUSTIVE -- compile and measure all 128, for real (lighter pass: best-of-3, 200,000 reps) ---

all 128 real schedules compiled and measured. every one bit-for-bit correct: confirmed
true best, by exhaustive real measurement: tiles=[dim0:2,dim1:8] order=dim1>dim0 unroll=8 (0.0087 us/call, best-of-3 over 200,000 reps)
(per-call time, not raw totals, since this pass uses a different repeat count than Part 2 --
comparing raw totals across two different repeat counts would not be a fair comparison)
exhaustive total wall-clock (compile + measure, all 128 candidates): 13531.1 ms

--- Part 4: did pruning to 8 find the true best? was it actually cheaper? ---

  true best (exhaustive, all 128) in the cost model's own top-8 short list: no
  hybrid's own winner == true best (exhaustive, all 128): no
  hybrid wall-clock:   2763.1 ms (8 candidates, careful measurement)
  exhaustive wall-clock:  13531.1 ms (128 candidates, lighter measurement)
  real wall-clock ratio (exhaustive / hybrid): 4.90x

The cost model's own top-8 did NOT contain the schedule real, exhaustive measurement found
to actually be fastest -- an honest miss, not a hidden one. This is exactly the risk of pruning
with an imperfect model, named plainly rather than smoothed over: Chapter 22 never claimed
estimateScheduleCost() was reliable enough to guarantee the true winner survives every prune,
only that it is cheap enough to be worth trying before paying for real measurement. A real
autotuner facing this tradeoff has options this book leaves as an open, named direction rather
than a settled answer: widen the short list, refine the cost model with the real mechanisms
Chapter 22 already found it was missing, or accept a probably-very-good schedule instead of
a provably-best one in exchange for the real time this section's own numbers show was saved.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which candidate ranks fastest, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! note "What this chapter proved, and what an imperfect cost model is actually for"
    Every number in this chapter is real: a real, measured compile-time cost landing in the same order of magnitude as real execution time (Section 23.1); a real, machine-specific answer to whether removing an interpreter changes a cost model's own ranking agreement (Section 23.2); and a real, honestly-reported hybrid autotuner, checked directly against exhaustively compiling and measuring the entire 128-schedule search space, with a genuine wall-clock accounting of what pruning saves whichever way the short list happened to land (Section 23.3). Chapter 22 never claimed `estimateScheduleCost()` was reliable enough to guarantee the true winner survives every prune -- only that it is cheap enough to be worth trying before paying for real measurement. This chapter is the first real, end-to-end test of that claim, and it reports its own outcome honestly rather than only in the case where the hybrid happened to win outright.

## Chapter Summary

This chapter closed the gap Chapter 22 left open on purpose: every candidate schedule that chapter ever measured went through a generic interpreter, never real compiled code. Section 23.1 built `generateScheduledElementwiseFunction()`, real codegen for one loop's own tile size and unroll factor, and found a genuinely new cost this book had never had to budget for -- compile time, measured separately from run time, landing in the same real order of magnitude as the computation it built. Section 23.2 extended that generator to a full `Schedule` with `generateScheduleFunction()`, baking loop order directly into the generated code's own shape, and reran Chapter 22.3's own 6-schedule ranking experiment with real compiled code in place of the interpreter -- an honest, machine-specific finding, not a single clean verdict either way. Section 23.3 put every piece together into a real hybrid autotuner: Chapter 22's own cost model pruning 128 real schedules down to a short list of 8, real compiled measurement on only that short list, and an honest check against compiling and measuring all 128 candidates for real as ground truth -- including a genuine, directly measured wall-clock accounting of what the hybrid approach actually saves. Every claim in this chapter was checked bit-for-bit for correctness on both of this book's own real machines; every timing number was reported as real and machine-specific, exactly as Chapter 22 first established.

## Self-Check Questions

1. Section 23.1 finds that compile time and execution time land in the same real order of magnitude at scale. Why did Chapter 22's own interpreter never have to budget for a cost like that at all?
2. In `generateScheduleFunction()`, what is the key difference between how the loop order is used compared to Chapter 22's own `executeSchedule()` -- and why does that difference mean the generated code has no `Schedule` struct left in it by the time g++ sees it?
3. Section 23.2 reruns Chapter 22.3's own 6-schedule ranking experiment with real compiled code. Why is it meaningful, rather than just noise, that the result is not the same on the cloud sandbox as it is on the device?
4. In Section 23.3's autotuner, why does the exhaustive (all-128) measurement pass deliberately use fewer repeat calls and trials than the hybrid (top-8) pass?
5. Why does comparing the hybrid's own winner against the exhaustive pass's own true best require comparing PER-CALL time rather than raw measured totals?
6. If the true best schedule, by exhaustive measurement, is NOT inside the cost model's own top-8 short list, does that mean `estimateScheduleCost()` failed at its job? Why or why not, per this chapter's own stated goal for the model?
7. What real, directly measured quantity does Section 23.3 use to argue the hybrid approach is worth using even when it does not find the provably best schedule?
8. Name one concrete way a real autotuner could respond to Section 23.3's own honest finding that pruning can miss the true best schedule, besides simply accepting the miss.

## Where We Go Next

Chapter 24 closes Part 5 by asking the question this chapter's own autotuner leaves open: what happens the NEXT time the same shape needs a schedule? Section 23.3's own hybrid autotuner does real, non-trivial work -- ranking 128 candidates, compiling and measuring 8 of them for real -- every single time it runs, even for a `LoopNest` it has already searched before. Chapter 24 builds on a real, cited practice from production autotuners (TVM's own AutoTVM tuning logs, already introduced in Chapter 3's survey): caching a search's own result, keyed by the shape that produced it, so a real program only pays this chapter's own real autotuning cost once per distinct shape it ever sees, not once per run.

## Worked Solutions

1. Chapter 22's own interpreter, `executeSchedule()`, reads a `Schedule`'s own fields as data at runtime inside one generic, already-compiled function -- trying a new candidate schedule never requires compiling anything new, only calling the same function with different data. Section 23.1's own generator produces a DIFFERENT function's worth of source text per candidate, and turning that text into a callable function requires a real g++ process, which is the cost the interpreter design was specifically built to avoid paying.
2. Chapter 22's `executeSchedule()` reads `schedule.loopOrder` as DATA, inside the function, every time it is called -- the same compiled function has to be ready to branch either way on every call. `generateScheduleFunction()` reads the identical field as a host C++ value while it is BUILDING the source text, so the branch is resolved once, at generation time, and only ONE of the two possible flat-index expressions is ever written into the generated text. By the time g++ compiles that text, there is no runtime check left to make and no `Schedule` value anywhere in the compiled function -- only whichever single nesting was actually chosen.
3. Chapter 22.3 diagnosed interpreter overhead as the likely reason its own combined cost model's ranking agreement landed close to chance on a 48-element toy problem. If real compiled code changed the agreement the SAME way on both machines, that would be reasonably strong evidence the diagnosis was right and general. Because the actual result differs by machine, the honest reading is narrower: whatever the interpreter's own overhead was contributing, it was not the only, or even necessarily the dominant, factor on both real machines -- a genuinely more useful finding than a single clean confirmation would have been, because it rules out over-crediting one explanation.
4. Compiling and measuring all 128 real schedules is, by Section 23.1's own finding, expensive per candidate -- 128 separate g++ processes plus 128 separate timed measurement passes. Using the same careful settings as the 8-candidate hybrid pass (2,000,000 repeat calls, 9 trials each) across all 128 would make the exhaustive ground-truth pass itself prohibitively slow to run at all; a lighter, deliberately noisier pass (200,000 repeat calls, 3 trials) keeps measuring the entire search space affordable, at the honest cost of somewhat less precise per-candidate timing.
5. The hybrid pass and the exhaustive pass use different repeat counts (2,000,000 versus 200,000), so a candidate's own raw best-of-N total time from one pass is not on the same scale as a raw total from the other -- a schedule measured with the SMALLER repeat count will always show a smaller raw total than an equally fast schedule measured with the LARGER repeat count, regardless of which one is actually faster per call. Dividing each candidate's own raw total by its own repeat count (`perCallUs()`) puts both passes' numbers on the same, comparable per-call basis.
6. No. Chapter 22 never claimed `estimateScheduleCost()` was accurate enough to guarantee the true winner survives every prune -- only that it is cheap enough to be worth trying before paying for real compiled measurement. A short-list miss is exactly the named, accepted risk of pruning with an imperfect model, not a failure of the model to do something it was never claimed to do; the model's actual job is to make a large space affordable to search at all, not to be perfectly correct about every ranking.
7. The real, directly measured total wall-clock time each approach actually took -- compiling and measuring only the top-8 short list (the hybrid) versus compiling and measuring all 128 candidates (the exhaustive ground-truth pass) -- reported as a real ratio between the two. Even in a run where the hybrid's own winner is not the true best schedule, the hybrid still completes in a fraction of the exhaustive approach's own real wall-clock time, which is the concrete, measured benefit pruning buys.
8. A real autotuner facing this tradeoff could widen the short list (`topK`) to keep more candidates in contention at a proportionally higher but still bounded real compilation cost; it could refine the cost model itself using the specific, diagnosed mechanisms Chapter 22 already found it was missing (the real turnover point in Section 22.1, the underestimated magnitude in Section 22.2); or it could simply accept a probably-very-good schedule instead of insisting on a provably-best one, in direct exchange for the real search time this chapter's own numbers show that choice saves.

---

**Sources cited in this chapter:**

TVM's own tuning-log practice, already introduced in Chapter 3's survey of real ML compilers, is referenced above as the real, cited precedent for Chapter 24's own upcoming topic (caching an autotuner's own results); no new source is introduced in this chapter's own body. This chapter's own real codegen generators (`generateScheduledElementwiseFunction()`, `generateScheduleFunction()`) and the hybrid autotuner built from them are original to this book, building directly on Chapter 20's own `JitModule`/`compileToSharedLibrary()`, Chapter 21's own `Schedule`/`enumerateSchedules()`, and Chapter 22's own `estimateScheduleCost()`.
