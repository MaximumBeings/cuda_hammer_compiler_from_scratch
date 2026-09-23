# 24. Caching and Reusing Autotuning Results

**What you will understand:** `TuningCache`, a real, file-backed key/value store -- one plain-text line per entry, `loopNestKey(nest)=serializeSchedule(schedule)` -- that persists an already-found `Schedule` past the lifetime of the process that found it, so a genuinely new program run, starting with an empty in-memory map, can `load()` an earlier run's own real result back from disk; `getOrAutotune()`, which checks the cache FIRST and only pays Chapter 23's own real hybrid-autotuning cost on an actual miss, demonstrated end to end with a real cache MISS followed by a real cache HIT from a completely different `TuningCache` object; the two real risks a cache key can carry -- a shape mismatch, which Chapter 23.2's own `generateScheduleFunction()` structurally cannot let corrupt correctness but which can still cost real, measured performance, and a COARSE key, which can silently collide two genuinely different `LoopNest` shapes into one cache entry with nothing in the cache's own interface able to reveal the mistake; and `loopNestKeyForTarget()`, which closes Part 5 by extending the cache key with a real compile-time architecture identifier, demonstrated on real measurements showing Chapter 23.3's own two real, different hybrid-autotuner winners (cloud sandbox vs. device) do not transfer cleanly across machines -- exactly the (workload, target) pairing TVM's own real AutoTVM tuning logs already use in production, Chapter 3's own survey material made concrete for the first time.

**What you need to know first:** Chapter 20's own `JitModule`/`compileToSharedLibrary()`; Chapter 21's own `Loop`/`LoopNest`/`Schedule`/`enumerateSchedules()`/`scheduleStr()`; Chapter 22's own `estimateScheduleCost()`; Chapter 23.2's own `generateScheduleFunction()` (the load-bearing fact this whole chapter leans on: it always derives its own loop bounds from the real `LoopNest` it is given, never from the `Schedule`'s own cached tile sizes); and Chapter 23.3's own real, measured hybrid-autotuner winners -- `tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4` on the cloud sandbox, `tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2` on the device -- for the identical `LoopNest [dim0:6, dim1:8]`, which this chapter's own capstone measures directly rather than merely restating.

---

Chapter 23 closed with a real, working hybrid autotuner and an honest accounting of what it costs: ranking 128 real schedules is cheap, but compiling and carefully measuring even a short list of 8 is not -- seconds of real wall-clock time, one g++ process and one timed measurement loop per candidate. Nothing about that autotuner remembers its own past work. Ask it to schedule the identical `LoopNest [dim0:6, dim1:8]` a second time, in a second program run, and it pays the full cost again, finding the same answer it already found once. This chapter closes Part 5 by giving the autotuner a memory -- not a clever one, just a real file on disk, the same real precedent this book has cited since Chapter 3's own survey of TVM: AutoTVM's own tuning LOGS, which persist a search's own result keyed by the workload and the hardware target it was tuned for, so a real compiler only pays the real cost of tuning once per shape it has never seen before.

A cache is a simple idea with two ways to get the key wrong, and this chapter takes each one seriously rather than assuming the obvious design is automatically safe. Section 24.1 builds the cache itself -- real file I/O, a real MISS followed by a real HIT from a genuinely different process -- keyed by a `LoopNest`'s own full shape. Section 24.2 asks what happens if that key captures less than the full shape: a genuine, if perhaps surprising, finding that Chapter 23.2's own `generateScheduleFunction()` already protects CORRECTNESS from a shape mismatch by construction, while performance and the cache's own honesty about what it is actually returning are both still very much at risk. Section 24.3 closes the chapter, and Part 5, on the risk this book has already measured without naming: two real machines, tuning the identical shape, do not agree on the winning schedule -- so a cache shared between them needs to know not just WHAT was tuned, but WHERE.

```text
Three real steps to a working, honest tuning cache:

  24.1 THE CACHE ITSELF       -- TuningCache: a real file, one line per
                                  entry, loopNestKey(shape) mapped to a
                                  serialized Schedule; getOrAutotune()
                                  pays Chapter 23's own real hybrid-
                                  autotuning cost on a MISS, skips it
                                  entirely on a HIT from a brand new
                                  process that loaded the same real file

  24.2 WHAT THE KEY MUST      -- a schedule found for one shape, applied
       CAPTURE                   to a genuinely different one, is still
                                  bit-for-bit CORRECT (generateScheduleFunction
                                  always trusts the real LoopNest's own
                                  extents, never the schedule's cached
                                  tile sizes) but not necessarily FAST;
                                  a COARSE key (total element count
                                  alone) can silently collide two real,
                                  different shapes into one entry, with
                                  a confident HIT hiding the mistake

  24.3 CACHING ACROSS         -- Chapter 23.3's own two real winners,
       MACHINES (capstone)       cloud sandbox and device, measured
                                  against EACH OTHER for real on
                                  whichever machine runs this section;
                                  loopNestKeyForTarget() extends the key
                                  with a real, compile-time architecture
                                  identifier, so one shared cache file
                                  holds both machines' own real answers
                                  without either one overwriting the
                                  other -- TVM's own real (workload,
                                  target) tuning-log design, arrived at
                                  here by direct measurement

Every real cost this chapter avoids paying twice was a real cost
Chapter 23 already measured paying once. Nothing here is simulated or
assumed: every MISS really autotunes, every HIT really loads a real
file, and every cross-machine number is measured on the machine that
actually ran the code.
```

## 24.1 A Persistent Tuning Cache

### Intuition

A cache is only useful across the boundary that matters -- a NEW program run, a genuinely different process that shares nothing in memory with whichever run found the answer the first time. An in-memory-only map, no matter how convenient inside one process, provides no benefit at all the next time the program starts from scratch, which is exactly when the real cost of autotuning gets paid again. `TuningCache` is built around that constraint directly: it is a real file on disk, written by `save()` and read back by `load()`, and this section proves the distinction is not academic by demonstrating a cache HIT from an object that never saw the original `put()` call -- only the real file the first object actually wrote.

### Background

Two small pieces make the cache possible at all. `loopNestKey(nest)` serializes a `LoopNest`'s own full per-dimension shape -- each loop's own name and extent -- into one string; `serializeSchedule()`/`deserializeSchedule()` do the same for a `Schedule`'s own tile sizes, loop order, and unroll factor, round-tripped through plain text with the same "no library needed" discipline this book's own generated code has used since Chapter 17. `TuningCache` itself is a thin `std::map<std::string, std::string>` with exactly the operations a cache needs -- `has()`, `get()`, `put()` -- plus the two that make it PERSISTENT: `save(path)` writes every entry as one `key=value` line, and `load(path)` reads that same format back, silently starting empty if the file does not exist yet (a cache with nothing cached is not an error condition).

`getOrAutotune()` is where the real payoff lives: check the cache first; on a HIT, return the cached `Schedule` with no compilation and no measurement at all; on a MISS, call `runHybridAutotuner()` -- Chapter 23.3's own hybrid autotuner, unchanged in every real cost it pays -- and immediately `put()` and `save()` the result, so the very next call, even from a different process, finds a HIT. The real demonstration in this section's own code below does not simulate two separate program runs with a comment; it constructs two genuinely different `TuningCache` objects, the second one loading from the real file the first one actually wrote, and times both calls for real. The ratio between them is real, not illustrative -- and, as the code's own closing comparison explains, it is large for a specific, nameable reason: one side pays for real g++ processes and real timed measurement loops, the other pays for one short file read.

```text
getOrAutotune(nest, cache, cachePath):

  key = loopNestKey(nest)
  if cache.has(key):
    return cache.get(key)                     -- HIT: no compilation,
                                                   no measurement
  winner = runHybridAutotuner(nest)            -- MISS: Chapter 23's
                                                   own real cost, paid
                                                   in full
  cache.put(key, winner)
  cache.save(cachePath)                        -- a REAL file, so the
                                                   NEXT process, not
                                                   just the next call
                                                   in THIS process, can
                                                   benefit
  return winner

Two genuinely different TuningCache objects in this section's own real
demonstration: the first finds nothing on disk (a real MISS, paying
Chapter 23's own real autotuning cost); the second, loading from the
exact file the first one wrote, finds a real HIT -- the identical
schedule, returned almost instantly, with no compilation or
measurement paid a second time.
```

## 24.2 What a Cache Key Must Capture

### Intuition

Section 24.1's own `loopNestKey()` captures a `LoopNest`'s FULL per-dimension shape without needing to argue for that choice -- it was simply what "the shape" meant. This section makes the argument directly, by building the alternative and testing it for real: what actually goes wrong if a cache key captures less than the full shape? The honest answer turns out to have two separate parts, and conflating them would understate one risk and overstate the other.

### Background

The first part is a genuine, structural guarantee, not a coincidence of this section's own test cases: `generateScheduleFunction()` (Chapter 23.2) always computes its own loop bounds -- `outerExtent`, `innerExtent`, and every `std::min()`-clamped block boundary -- from the real `LoopNest` it is given, and never reads the `Schedule`'s own tile sizes as anything other than a STEP size. A tile size larger than the real extent it is stepping through simply produces one block covering the whole extent; an unroll factor larger than a tile's own real size simply means the main unrolled loop never executes and every element falls through to the scalar tail. Nothing about applying a schedule found for one shape to a genuinely different shape (with the same number of loops) can read or write outside the real, correct bounds -- which this section's own code demonstrates directly, not by argument: a schedule autotuned for a tiny `2x2` shape, applied unmodified to a real `6x8` shape, produces the exact bit-for-bit correct answer.

The second part is where the real risk actually lives. Correctness surviving a shape mismatch says nothing about PERFORMANCE -- a schedule's own tile sizes and unroll factor were chosen, by Chapter 23's own cost model and real measurement, for a DIFFERENT problem's own real memory-access pattern and loop-overhead tradeoff, and nothing guarantees they transfer well. This section measures that gap directly rather than asserting a direction, because the honest answer is that it is not consistent -- a genuinely useful finding in itself, since it means a cache serving a foreign schedule is not merely "somewhat suboptimal," it is UNPREDICTABLE, which is a worse property for a real system to have silently. That unpredictability is exactly why a COARSE cache key is dangerous in a way a shape mismatch alone is not: this section builds `coarseKey()`, which reduces a `LoopNest` to nothing but its own total element count, and shows it directly colliding two genuinely different real shapes -- `[dim0:4, dim1:8]` and `[dim0:8, dim1:4]`, both 32 elements, different per-dimension extents, different legal tile candidates, different memory-stride behavior (Chapter 22.2) -- into one cache entry. A `TuningCache` keyed that way reports a confident HIT for the second shape the moment the first has ever been tuned, and -- because correctness is never at risk, per this section's own first finding -- nothing about the program's own output would ever reveal that the schedule being served was never actually tuned for the shape being computed.

```text
Two separate real risks, not one:

  SHAPE MISMATCH (a schedule tuned for shape A, applied to shape B)
    correctness:  SAFE, by construction (generateScheduleFunction always
                  trusts the real LoopNest's own extents)
    performance:  AT RISK, unpredictably -- sometimes close, sometimes
                  not, with no way to know in advance which

  COARSE KEY (a cache key that cannot distinguish shape A from shape B)
    the cache's own interface: reports a confident HIT for the WRONG
    shape, with nothing about that HIT revealing the mistake -- silent,
    not merely suboptimal

Section 24.1's own loopNestKey(), keyed on the FULL per-dimension
shape, is what keeps two real, different LoopNests from ever
colliding into one entry in the first place.
```

## 24.3 Caching Across Machines: Why the Key Needs a Target Too

### Intuition

Chapter 23.3's own real, honest finding -- reported at the time as a machine-specific result, not yet named as a caching problem -- is the premise this capstone makes concrete: the hybrid autotuner found a DIFFERENT winning schedule on the cloud sandbox than on the device, for the identical `LoopNest [dim0:6, dim1:8]`. A cache keyed only by shape, exactly as Section 24.1 built it, cannot represent that difference at all -- it has exactly one slot for `dim0:6,dim1:8`, and whichever real machine happens to populate it first silently becomes the answer every OTHER machine reads back, with nothing in the cache's own interface distinguishing "tuned for you" from "tuned for someone else."

### Background

This section does not merely restate Chapter 23.3's own two numbers -- it takes both of that section's own real winning `Schedule` values, embedded here as literals cited directly to Section 23.3, and measures them AGAINST EACH OTHER for real, on whichever real machine actually runs this section's own code. Whichever machine that is, one of the two schedules is that machine's own real winner and the other is the other machine's real winner, and the section reports, honestly, which one this machine's own clock finds faster -- not assuming the answer transfers, testing it directly. `targetId()` names the real architecture this program is actually compiled for using the same predefined compiler macros (`__aarch64__`, `__x86_64__`) a real build system would check, resolved at compile time rather than guessed at runtime, because the target is a fact about the machine, not something that needs detecting.

`loopNestKeyForTarget()` closes the chapter by extending Section 24.1's own `loopNestKey()` with exactly that target string, joined by a separator that can never appear inside either half. The section's own real demonstration writes two entries -- one keyed to `x86_64`, one to `aarch64` -- into a SINGLE shared cache file, exactly as two real machines sharing one tuning-log file in a real deployment would, and shows both surviving together: whichever machine loads that shared file finds its OWN real entry under its OWN key, and the OTHER machine's own entry remains untouched, present, and available the next time THAT machine runs. This is not a design this book is adopting on convention -- it is the same (workload, target) pairing TVM's own AutoTVM tuning logs already use in real production, cited since Chapter 3's own survey, now arrived at here by this book's own direct, real, cross-machine measurement rather than only by citation.

```text
loopNestKeyForTarget(nest, target) = loopNestKey(nest) + "@" + target

  ONE shared cache file:
    "dim0:6,dim1:8@x86_64"  -> cloud sandbox's own real winner
    "dim0:6,dim1:8@aarch64" -> device's own real winner

  Whichever machine actually loads this file:
    looks up ITS OWN key (targetId(), resolved at compile time)
    finds its OWN real entry -- never the other machine's
    the OTHER machine's own entry survives, untouched, in the SAME file

The identical LoopNest shape, two real machines, two real winning
schedules, one real shared file -- exactly TVM's own real (workload,
target) tuning-log design (Chapter 3), motivated here by this book's
own direct measurement of what happens without it.
```

```cpp
// Chapter 24: Caching and Reusing Autotuning Results
// 060_a_persistent_tuning_cache_keyed_by_loop_nest_shape.cpp
//
// Section 24.1 -- Chapter 23's own hybrid autotuner does real, non-trivial
// work every time it runs: rank 128 real schedules with a cost model,
// compile and carefully measure 8 of them for real. That cost is paid
// again, in full, the next time a program asks for a schedule for the
// SAME LoopNest shape it has already searched -- Chapter 23 never gave
// the autotuner any memory of its own past work. This section builds
// that memory: a real, persistent cache, keyed by a LoopNest's own
// shape, that stores an already-found Schedule to a real file on disk
// and loads it back on a later run -- modeled directly on the real
// precedent TVM's own AutoTVM already established (its own tuning LOGS,
// introduced in Chapter 3's survey), rather than an in-memory-only
// memoization that would vanish the moment the program exits.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 060_a_persistent_tuning_cache_keyed_by_loop_nest_shape.cpp -o 060_driver -ldl
// Run:     ./060_driver
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
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

// ==================== generateScheduleFunction() (Chapter 23.2, unchanged) ====================

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

// ==================== Section 24.1: a real, persistent tuning cache ====================

// loopNestKey() captures the FULL per-dimension shape, name and extent
// together, for every loop in the nest -- not just a total element count
// or a single dimension. Section 24.2 shows directly why that matters.
static std::string loopNestKey(const LoopNest& nest) {
    std::string out;
    for (size_t i = 0; i < nest.loops.size(); ++i) {
        if (i) out += ",";
        out += nest.loops[i].dimName + ":" + std::to_string(nest.loops[i].extent);
    }
    return out;
}

// A Schedule serializes to one plain-text line: tile sizes, then loop
// order, then unroll factor, each field delimited so it can be parsed
// back exactly -- no library, the same "plain text this book's own
// generated code has always been" discipline Chapter 17 established.
static std::string serializeSchedule(const Schedule& s) {
    std::string out;
    for (size_t i = 0; i < s.tileSizePerLoop.size(); ++i) {
        if (i) out += "-";
        out += std::to_string(s.tileSizePerLoop[i]);
    }
    out += "|";
    for (size_t i = 0; i < s.loopOrder.size(); ++i) {
        if (i) out += "-";
        out += std::to_string(s.loopOrder[i]);
    }
    out += "|" + std::to_string(s.unrollFactor);
    return out;
}
static Schedule deserializeSchedule(const std::string& text) {
    Schedule s;
    std::stringstream ss(text);
    std::string tilesPart, orderPart, unrollPart;
    std::getline(ss, tilesPart, '|');
    std::getline(ss, orderPart, '|');
    std::getline(ss, unrollPart, '|');
    std::stringstream tss(tilesPart);
    std::string tok;
    while (std::getline(tss, tok, '-')) s.tileSizePerLoop.push_back(std::stoll(tok));
    std::stringstream oss(orderPart);
    while (std::getline(oss, tok, '-')) s.loopOrder.push_back(std::stoi(tok));
    s.unrollFactor = std::stoll(unrollPart);
    return s;
}

// TuningCache is a real, file-backed key/value store -- one line per
// entry, "key=serializedSchedule" -- deliberately as plain as this
// book's own generated code has always been, and deliberately backed by
// a REAL file rather than only an in-memory map, so an entry survives
// past the lifetime of the process that found it (the entire point: the
// next real program run, a new process, starts with an EMPTY in-memory
// map and has to load() before it can benefit from earlier work).
class TuningCache {
public:
    void load(const std::string& path) {
        entries_.clear();
        std::ifstream f(path);
        if (!f) return;  // no file yet is not an error -- an empty cache
        std::string line;
        while (std::getline(f, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            entries_[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }
    void save(const std::string& path) const {
        std::ofstream f(path);
        for (const auto& kv : entries_) f << kv.first << "=" << kv.second << "\n";
    }
    bool has(const std::string& key) const { return entries_.count(key) != 0; }
    Schedule get(const std::string& key) const { return deserializeSchedule(entries_.at(key)); }
    void put(const std::string& key, const Schedule& schedule) { entries_[key] = serializeSchedule(schedule); }
    size_t size() const { return entries_.size(); }
private:
    std::map<std::string, std::string> entries_;
};

// runHybridAutotuner() is Chapter 23.3's own hybrid autotuner, cut down
// to return just the winning Schedule: estimateScheduleCost() ranks
// every real, legal schedule, the cheapest topK are compiled and
// carefully measured for real, and the fastest of those wins. Every
// real cost Chapter 23 found -- one g++ process per candidate, a timed
// clock loop per candidate -- is paid here in full; this function is
// exactly the cost this chapter's own cache exists to avoid paying
// twice for the identical shape.
static Schedule runHybridAutotuner(const LoopNest& nest, const std::vector<float>& a, const std::vector<float>& reference) {
    std::vector<Schedule> allSchedules = enumerateSchedules(nest);
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

    Schedule best;
    double bestUs = std::numeric_limits<double>::max();
    for (size_t r = 0; r < std::min(topK, rankedIdx.size()); ++r) {
        const Schedule& candidate = allSchedules[rankedIdx[r]];
        std::string src = generateScheduleFunction(nest, candidate);
        std::string cppPath = "/tmp/060_autotune_" + std::to_string(r) + ".cpp";
        std::string soPath = "/tmp/060_autotune_" + std::to_string(r) + ".so";
        std::string log;
        compileToSharedLibrary(src, cppPath, soPath, log);
        JitModule mod(soPath);
        ComputeFn2D fn = mod.getFunction<ComputeFn2D>("compute");
        std::vector<float> out(a.size());
        double bestMs = std::numeric_limits<double>::max();
        for (int trial = 0; trial < 9; ++trial) {
            auto start = std::chrono::steady_clock::now();
            for (long long rep = 0; rep < 2000000; ++rep) fn(a.data(), out.data());
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (ms < bestMs) bestMs = ms;
        }
        if (!arraysExactlyEqual(out, reference)) throw std::runtime_error("runHybridAutotuner: candidate mismatch");
        double us = bestMs * 1000.0 / 2000000.0;
        if (us < bestUs) { bestUs = us; best = candidate; }
    }
    return best;
}

// getOrAutotune() is the whole point of this section: check the cache
// FIRST; only call the real, expensive runHybridAutotuner() on a real
// miss, and persist the result immediately so the NEXT call -- even
// from a brand new process that starts with an empty in-memory map --
// can load() it back from the real file this function just wrote.
static Schedule getOrAutotune(const LoopNest& nest, TuningCache& cache, const std::string& cachePath,
                               const std::vector<float>& a, const std::vector<float>& reference, bool& wasHit) {
    std::string key = loopNestKey(nest);
    if (cache.has(key)) {
        wasHit = true;
        return cache.get(key);
    }
    wasHit = false;
    Schedule winner = runHybridAutotuner(nest, a, reference);
    cache.put(key, winner);
    cache.save(cachePath);
    return winner;
}

int main() {
    printf("=== Section 24.1: a real, persistent tuning cache, keyed by LoopNest shape ===\n\n");

    // ---- Part 1: loopNestKey() and schedule serialize/deserialize round-trip ----
    printf("--- Part 1: loopNestKey() and Schedule serialize/deserialize, round-tripped ---\n\n");
    LoopNest nest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};
    printf("  loopNestKey(LoopNest[dim0:6, dim1:8]) = \"%s\"\n\n", loopNestKey(nest).c_str());

    std::vector<Schedule> testSchedules = {
        Schedule{{1, 1}, {0, 1}, 1},
        Schedule{{6, 8}, {0, 1}, 4},
        Schedule{{2, 4}, {1, 0}, 8},
    };
    bool allRoundTripOk = true;
    for (const Schedule& s : testSchedules) {
        std::string text = serializeSchedule(s);
        Schedule s2 = deserializeSchedule(text);
        bool ok = (s.tileSizePerLoop == s2.tileSizePerLoop && s.loopOrder == s2.loopOrder && s.unrollFactor == s2.unrollFactor);
        allRoundTripOk = allRoundTripOk && ok;
        printf("  %-40s -> \"%s\" -> round-trip: %s\n", scheduleStr(nest, s).c_str(), text.c_str(), ok ? "exact match" : "MISMATCH");
    }
    printf("\nself-check: every Schedule serializes to one plain-text line and deserializes back to the exact\n");
    printf("same tile sizes, loop order, and unroll factor (%s) -- no information lost round-tripping\n",
           allRoundTripOk ? "confirmed" : "MISMATCH");
    printf("through real text, the same discipline this book's own generated code has used since Chapter 17.\n");

    // ---- Part 2: TuningCache -- a real file, written by one object, read back by a DIFFERENT one ----
    printf("\n--- Part 2: TuningCache -- a real file, written by one object, loaded by a fresh one ---\n\n");
    std::string cachePath = "/tmp/060_tuning_cache_demo.txt";
    remove(cachePath.c_str());

    TuningCache writerCache;
    Schedule manualEntry{{6, 8}, {0, 1}, 4};
    writerCache.put(loopNestKey(nest), manualEntry);
    writerCache.save(cachePath);
    printf("  writerCache: put 1 entry, saved to a real file at %s\n", cachePath.c_str());

    TuningCache readerCache;  // a DIFFERENT object -- starts with zero entries in memory
    printf("  readerCache (freshly constructed, before load()): has(key) = %s\n",
           readerCache.has(loopNestKey(nest)) ? "true" : "false");
    readerCache.load(cachePath);
    bool cacheRoundTripOk = readerCache.has(loopNestKey(nest));
    Schedule loadedEntry = readerCache.get(loopNestKey(nest));
    cacheRoundTripOk = cacheRoundTripOk && loadedEntry.tileSizePerLoop == manualEntry.tileSizePerLoop
                        && loadedEntry.loopOrder == manualEntry.loopOrder && loadedEntry.unrollFactor == manualEntry.unrollFactor;
    printf("  readerCache (after load() from the real file): has(key) = %s, entry matches: %s\n",
           readerCache.has(loopNestKey(nest)) ? "true" : "false", cacheRoundTripOk ? "yes" : "NO");
    printf("\nself-check: an entry written by ONE TuningCache object, through a real file on disk, is read\n");
    printf("back correctly by a COMPLETELY DIFFERENT object that never saw the original put() call (%s) --\n",
           cacheRoundTripOk ? "confirmed" : "MISMATCH");
    printf("the real behavior a NEW PROGRAM RUN needs: no shared memory, no live process, just a real file.\n");

    // ---- Part 3 (capstone): getOrAutotune() -- a real cache MISS, then a real cache HIT ----
    printf("\n--- Part 3: getOrAutotune() -- real autotuning cost paid once, skipped on the next real run ---\n\n");
    std::string realCachePath = "/tmp/060_real_cache.txt";
    remove(realCachePath.c_str());

    long long dim0Extent = nest.loops[0].extent, dim1Extent = nest.loops[1].extent;
    std::vector<float> a(static_cast<size_t>(dim0Extent * dim1Extent));
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> reference(a.size());
    for (size_t i = 0; i < a.size(); ++i) reference[i] = a[i] * 2.0f + 1.0f;

    TuningCache runOneCache;  // simulates the FIRST real program run: nothing on disk yet
    runOneCache.load(realCachePath);
    bool firstWasHit = true;
    auto firstStart = std::chrono::steady_clock::now();
    Schedule firstResult = getOrAutotune(nest, runOneCache, realCachePath, a, reference, firstWasHit);
    auto firstEnd = std::chrono::steady_clock::now();
    double firstUs = std::chrono::duration<double, std::micro>(firstEnd - firstStart).count();
    printf("  run 1 (fresh cache, nothing on disk yet): %s in %.1f ms -> %s\n",
           firstWasHit ? "HIT" : "MISS", firstUs / 1000.0, scheduleStr(nest, firstResult).c_str());

    TuningCache runTwoCache;  // simulates a SECOND, later program run: a brand new object
    runTwoCache.load(realCachePath);  // loading whatever run 1 actually saved to the real file
    bool secondWasHit = true;
    auto secondStart = std::chrono::steady_clock::now();
    Schedule secondResult = getOrAutotune(nest, runTwoCache, realCachePath, a, reference, secondWasHit);
    auto secondEnd = std::chrono::steady_clock::now();
    double secondUs = std::chrono::duration<double, std::micro>(secondEnd - secondStart).count();
    printf("  run 2 (brand new TuningCache object, loaded from run 1's real file): %s in %.1f us -> %s\n",
           secondWasHit ? "HIT" : "MISS", secondUs, scheduleStr(nest, secondResult).c_str());

    bool sameSchedule = firstResult.tileSizePerLoop == secondResult.tileSizePerLoop
                         && firstResult.loopOrder == secondResult.loopOrder && firstResult.unrollFactor == secondResult.unrollFactor;
    printf("\n  run 1 was a %s, run 2 was a %s, both returned the %s schedule\n",
           firstWasHit ? "HIT" : "MISS", secondWasHit ? "HIT" : "MISS", sameSchedule ? "IDENTICAL" : "DIFFERENT");
    if (secondUs < 1.0) {
        printf("  real wall-clock: run 1 = %.1f ms, run 2 = %.1f us -- too fast for this clock to resolve\n",
               firstUs / 1000.0, secondUs);
        printf("  meaningfully at all; the honest statement is not a specific ratio, it is that a cache HIT\n");
        printf("  costs one file load and a string parse, nothing this program's own clock can distinguish\n");
        printf("  from zero, next to a MISS's real g++ processes and real timed measurement loops.\n");
    } else {
        printf("  real wall-clock: run 1 = %.1f ms, run 2 = %.1f us (ratio: run 1 is %.0fx slower)\n",
               firstUs / 1000.0, secondUs, firstUs / secondUs);
        printf("  a ratio this large is not a claim that caching is generally a million-times speedup --\n");
        printf("  it reflects what is actually being compared: 8 real g++ processes plus 8 real timed\n");
        printf("  measurement loops on one side, one short file read and a string parse on the other.\n");
    }

    printf("\nself-check: run 1 pays Chapter 23's own real hybrid-autotuning cost in full (a MISS, compiling\n");
    printf("and measuring 8 real candidates); run 2 -- a genuinely different TuningCache object, loaded from\n");
    printf("the real file run 1 actually wrote -- finds a HIT and returns the identical schedule almost\n");
    printf("instantly, with no compilation and no measurement at all. This is the entire point of this\n");
    printf("chapter: a real autotuning result, paid for once, reused for real on every later run that asks\n");
    printf("for the same LoopNest shape -- exactly the role TVM's own AutoTVM tuning logs (Chapter 3) play\n");
    printf("in a real production compiler.\n");

    bool allOk = allRoundTripOk && cacheRoundTripOk && sameSchedule && !firstWasHit && secondWasHit;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "060_a_persistent_tuning_cache_keyed_by_loop_nest_shape.cpp" -o "060_a_persistent_tuning_cache_keyed_by_loop_nest_shape" -ldl
./"060_a_persistent_tuning_cache_keyed_by_loop_nest_shape"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 24.1: a real, persistent tuning cache, keyed by LoopNest shape ===

--- Part 1: loopNestKey() and Schedule serialize/deserialize, round-tripped ---

  loopNestKey(LoopNest[dim0:6, dim1:8]) = "dim0:6,dim1:8"

  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1 -> "1-1|0-1|1" -> round-trip: exact match
  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4 -> "6-8|0-1|4" -> round-trip: exact match
  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=8 -> "2-4|1-0|8" -> round-trip: exact match

self-check: every Schedule serializes to one plain-text line and deserializes back to the exact
same tile sizes, loop order, and unroll factor (confirmed) -- no information lost round-tripping
through real text, the same discipline this book's own generated code has used since Chapter 17.

--- Part 2: TuningCache -- a real file, written by one object, loaded by a fresh one ---

  writerCache: put 1 entry, saved to a real file at /tmp/060_tuning_cache_demo.txt
  readerCache (freshly constructed, before load()): has(key) = false
  readerCache (after load() from the real file): has(key) = true, entry matches: yes

self-check: an entry written by ONE TuningCache object, through a real file on disk, is read
back correctly by a COMPLETELY DIFFERENT object that never saw the original put() call (confirmed) --
the real behavior a NEW PROGRAM RUN needs: no shared memory, no live process, just a real file.

--- Part 3: getOrAutotune() -- real autotuning cost paid once, skipped on the next real run ---

  run 1 (fresh cache, nothing on disk yet): MISS in 5619.8 ms -> tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2
  run 2 (brand new TuningCache object, loaded from run 1's real file): HIT in 6.1 us -> tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

  run 1 was a MISS, run 2 was a HIT, both returned the IDENTICAL schedule
  real wall-clock: run 1 = 5619.8 ms, run 2 = 6.1 us (ratio: run 1 is 927661x slower)
  a ratio this large is not a claim that caching is generally a million-times speedup --
  it reflects what is actually being compared: 8 real g++ processes plus 8 real timed
  measurement loops on one side, one short file read and a string parse on the other.

self-check: run 1 pays Chapter 23's own real hybrid-autotuning cost in full (a MISS, compiling
and measuring 8 real candidates); run 2 -- a genuinely different TuningCache object, loaded from
the real file run 1 actually wrote -- finds a HIT and returns the identical schedule almost
instantly, with no compilation and no measurement at all. This is the entire point of this
chapter: a real autotuning result, paid for once, reused for real on every later run that asks
for the same LoopNest shape -- exactly the role TVM's own AutoTVM tuning logs (Chapter 3) play
in a real production compiler.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 24.1: a real, persistent tuning cache, keyed by LoopNest shape ===

--- Part 1: loopNestKey() and Schedule serialize/deserialize, round-tripped ---

  loopNestKey(LoopNest[dim0:6, dim1:8]) = "dim0:6,dim1:8"

  tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=1 -> "1-1|0-1|1" -> round-trip: exact match
  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4 -> "6-8|0-1|4" -> round-trip: exact match
  tiles=[dim0:2,dim1:4] order=dim1>dim0 unroll=8 -> "2-4|1-0|8" -> round-trip: exact match

self-check: every Schedule serializes to one plain-text line and deserializes back to the exact
same tile sizes, loop order, and unroll factor (confirmed) -- no information lost round-tripping
through real text, the same discipline this book's own generated code has used since Chapter 17.

--- Part 2: TuningCache -- a real file, written by one object, loaded by a fresh one ---

  writerCache: put 1 entry, saved to a real file at /tmp/060_tuning_cache_demo.txt
  readerCache (freshly constructed, before load()): has(key) = false
  readerCache (after load() from the real file): has(key) = true, entry matches: yes

self-check: an entry written by ONE TuningCache object, through a real file on disk, is read
back correctly by a COMPLETELY DIFFERENT object that never saw the original put() call (confirmed) --
the real behavior a NEW PROGRAM RUN needs: no shared memory, no live process, just a real file.

--- Part 3: getOrAutotune() -- real autotuning cost paid once, skipped on the next real run ---

  run 1 (fresh cache, nothing on disk yet): MISS in 2753.9 ms -> tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2
  run 2 (brand new TuningCache object, loaded from run 1's real file): HIT in 3.0 us -> tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

  run 1 was a MISS, run 2 was a HIT, both returned the IDENTICAL schedule
  real wall-clock: run 1 = 2753.9 ms, run 2 = 3.0 us (ratio: run 1 is 917981x slower)
  a ratio this large is not a claim that caching is generally a million-times speedup --
  it reflects what is actually being compared: 8 real g++ processes plus 8 real timed
  measurement loops on one side, one short file read and a string parse on the other.

self-check: run 1 pays Chapter 23's own real hybrid-autotuning cost in full (a MISS, compiling
and measuring 8 real candidates); run 2 -- a genuinely different TuningCache object, loaded from
the real file run 1 actually wrote -- finds a HIT and returns the identical schedule almost
instantly, with no compilation and no measurement at all. This is the entire point of this
chapter: a real autotuning result, paid for once, reused for real on every later run that asks
for the same LoopNest shape -- exactly the role TVM's own AutoTVM tuning logs (Chapter 3) play
in a real production compiler.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which schedule measures faster, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! note "A cache is only as good as the boundary it is tested across"
    It would have been easy to demonstrate `getOrAutotune()`'s own benefit with two calls inside ONE process, sharing one `TuningCache` object in memory the whole time -- and that demonstration would have proven nothing about what this chapter actually needs, since an in-memory map already provides that benefit for free, with no file I/O at all. The real test this section's own code performs -- a SECOND, genuinely different `TuningCache` object, constructed fresh and loading from the exact file the first object wrote -- is the only test that actually exercises the boundary a real autotuning cache has to survive: a brand new program run, a brand new process, with nothing left over from the run that did the real work the first time.

## 24.2 What a Cache Key Must Capture

### Intuition

Section 24.1's own `loopNestKey()` captures a `LoopNest`'s FULL per-dimension shape without needing to argue for that choice -- it was simply what "the shape" meant. This section makes the argument directly, by building the alternative and testing it for real: what actually goes wrong if a cache key captures less than the full shape? The honest answer turns out to have two separate parts, and conflating them would understate one risk and overstate the other.

### Background

The first part is a genuine, structural guarantee, not a coincidence of this section's own test cases: `generateScheduleFunction()` (Chapter 23.2) always computes its own loop bounds -- `outerExtent`, `innerExtent`, and every `std::min()`-clamped block boundary -- from the real `LoopNest` it is given, and never reads the `Schedule`'s own tile sizes as anything other than a STEP size. A tile size larger than the real extent it is stepping through simply produces one block covering the whole extent; an unroll factor larger than a tile's own real size simply means the main unrolled loop never executes and every element falls through to the scalar tail. Nothing about applying a schedule found for one shape to a genuinely different shape (with the same number of loops) can read or write outside the real, correct bounds -- which this section's own code demonstrates directly, not by argument: a schedule autotuned for a tiny `2x2` shape, applied unmodified to a real `6x8` shape, produces the exact bit-for-bit correct answer.

The second part is where the real risk actually lives. Correctness surviving a shape mismatch says nothing about PERFORMANCE -- a schedule's own tile sizes and unroll factor were chosen, by Chapter 23's own cost model and real measurement, for a DIFFERENT problem's own real memory-access pattern and loop-overhead tradeoff, and nothing guarantees they transfer well. This section measures that gap directly rather than asserting a direction, because the honest answer is that it is not consistent -- a genuinely useful finding in itself, since it means a cache serving a foreign schedule is not merely "somewhat suboptimal," it is UNPREDICTABLE, which is a worse property for a real system to have silently. That unpredictability is exactly why a COARSE cache key is dangerous in a way a shape mismatch alone is not: this section builds `coarseKey()`, which reduces a `LoopNest` to nothing but its own total element count, and shows it directly colliding two genuinely different real shapes -- `[dim0:4, dim1:8]` and `[dim0:8, dim1:4]`, both 32 elements, different per-dimension extents, different legal tile candidates, different memory-stride behavior (Chapter 22.2) -- into one cache entry. A `TuningCache` keyed that way reports a confident HIT for the second shape the moment the first has ever been tuned, and -- because correctness is never at risk, per this section's own first finding -- nothing about the program's own output would ever reveal that the schedule being served was never actually tuned for the shape being computed.

```text
Two separate real risks, not one:

  SHAPE MISMATCH (a schedule tuned for shape A, applied to shape B)
    correctness:  SAFE, by construction (generateScheduleFunction always
                  trusts the real LoopNest's own extents)
    performance:  AT RISK, unpredictably -- sometimes close, sometimes
                  not, with no way to know in advance which

  COARSE KEY (a cache key that cannot distinguish shape A from shape B)
    the cache's own interface: reports a confident HIT for the WRONG
    shape, with nothing about that HIT revealing the mistake -- silent,
    not merely suboptimal

Section 24.1's own loopNestKey(), keyed on the FULL per-dimension
shape, is what keeps two real, different LoopNests from ever
colliding into one entry in the first place.
```

```cpp
// Chapter 24: Caching and Reusing Autotuning Results
// 061_what_a_cache_key_must_capture.cpp
//
// Section 24.2 -- Section 24.1's own `loopNestKey()` serializes a LoopNest's
// FULL per-dimension shape -- every loop's own name and extent -- into
// the cache key. This section asks the question that choice was already
// quietly answering: what happens if the key captures LESS than the full
// shape? Two things, and they turn out to be genuinely different kinds
// of risk. First: applying a schedule that was found for one shape to a
// GENUINELY DIFFERENT shape never corrupts the answer -- this book's own
// `generateScheduleFunction()` (Chapter 23.2) always derives its loop
// bounds from the real `LoopNest` it is given, never from the cached
// `Schedule`'s own tile sizes, so correctness survives a shape mismatch
// by construction. Second: it can still cost real, measurable
// performance, and -- the genuinely dangerous part -- a coarse cache key
// can report a confident HIT while silently serving a schedule tuned for
// a different problem, with nothing in the cache's own interface able to
// tell the difference.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 061_what_a_cache_key_must_capture.cpp -o 061_driver -ldl
// Run:     ./061_driver
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
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

// ==================== generateScheduleFunction() (Chapter 23.2, unchanged) ====================
//
// The load-bearing fact for this whole section: outerExtent/innerExtent
// below are ALWAYS read from `nest` -- the real LoopNest being computed
// over -- never from `schedule`. A schedule's own tileSizePerLoop only
// ever controls how big a STEP each loop takes; std::min() against the
// nest's own real extent clamps every block's own boundary regardless of
// how that tile size compares to the real extent. Nothing about this
// generator can go out of bounds because a schedule was found for a
// DIFFERENT shape.
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

// ==================== loopNestKey() / TuningCache (Section 24.1, unchanged) ====================

static std::string loopNestKey(const LoopNest& nest) {
    std::string out;
    for (size_t i = 0; i < nest.loops.size(); ++i) {
        if (i) out += ",";
        out += nest.loops[i].dimName + ":" + std::to_string(nest.loops[i].extent);
    }
    return out;
}
static std::string serializeSchedule(const Schedule& s) {
    std::string out;
    for (size_t i = 0; i < s.tileSizePerLoop.size(); ++i) {
        if (i) out += "-";
        out += std::to_string(s.tileSizePerLoop[i]);
    }
    out += "|";
    for (size_t i = 0; i < s.loopOrder.size(); ++i) {
        if (i) out += "-";
        out += std::to_string(s.loopOrder[i]);
    }
    out += "|" + std::to_string(s.unrollFactor);
    return out;
}
static Schedule deserializeSchedule(const std::string& text) {
    Schedule s;
    std::stringstream ss(text);
    std::string tilesPart, orderPart, unrollPart;
    std::getline(ss, tilesPart, '|');
    std::getline(ss, orderPart, '|');
    std::getline(ss, unrollPart, '|');
    std::stringstream tss(tilesPart);
    std::string tok;
    while (std::getline(tss, tok, '-')) s.tileSizePerLoop.push_back(std::stoll(tok));
    std::stringstream oss(orderPart);
    while (std::getline(oss, tok, '-')) s.loopOrder.push_back(std::stoi(tok));
    s.unrollFactor = std::stoll(unrollPart);
    return s;
}
class TuningCache {
public:
    void load(const std::string& path) {
        entries_.clear();
        std::ifstream f(path);
        if (!f) return;
        std::string line;
        while (std::getline(f, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            entries_[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }
    void save(const std::string& path) const {
        std::ofstream f(path);
        for (const auto& kv : entries_) f << kv.first << "=" << kv.second << "\n";
    }
    bool has(const std::string& key) const { return entries_.count(key) != 0; }
    Schedule get(const std::string& key) const { return deserializeSchedule(entries_.at(key)); }
    void put(const std::string& key, const Schedule& schedule) { entries_[key] = serializeSchedule(schedule); }
private:
    std::map<std::string, std::string> entries_;
};

// runHybridAutotuner() (Section 24.1, unchanged): Chapter 23.3's own
// hybrid autotuner, returning just the winning Schedule.
static Schedule runHybridAutotuner(const LoopNest& nest, const std::vector<float>& a, const std::vector<float>& reference) {
    std::vector<Schedule> allSchedules = enumerateSchedules(nest);
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

    Schedule best;
    double bestUs = std::numeric_limits<double>::max();
    for (size_t r = 0; r < std::min(topK, rankedIdx.size()); ++r) {
        const Schedule& candidate = allSchedules[rankedIdx[r]];
        std::string src = generateScheduleFunction(nest, candidate);
        std::string cppPath = "/tmp/061_autotune_" + std::to_string(r) + ".cpp";
        std::string soPath = "/tmp/061_autotune_" + std::to_string(r) + ".so";
        std::string log;
        compileToSharedLibrary(src, cppPath, soPath, log);
        JitModule mod(soPath);
        ComputeFn2D fn = mod.getFunction<ComputeFn2D>("compute");
        std::vector<float> out(a.size());
        double bestMs = std::numeric_limits<double>::max();
        for (int trial = 0; trial < 9; ++trial) {
            auto start = std::chrono::steady_clock::now();
            for (long long rep = 0; rep < 2000000; ++rep) fn(a.data(), out.data());
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (ms < bestMs) bestMs = ms;
        }
        if (!arraysExactlyEqual(out, reference)) throw std::runtime_error("runHybridAutotuner: candidate mismatch");
        double us = bestMs * 1000.0 / 2000000.0;
        if (us < bestUs) { bestUs = us; best = candidate; }
    }
    return best;
}

// ==================== Section 24.2: what a cache key must capture ====================

// coarseKey() is deliberately the WRONG idea, built to fail: it captures
// only the total element count, throwing away every per-dimension detail
// loopNestKey() (Section 24.1) keeps.
static std::string coarseKey(const LoopNest& nest) {
    return std::to_string(nest.totalIterations());
}

int main() {
    printf("=== Section 24.2: what a cache key must capture ===\n\n");

    // ---- Part 1: a schedule tuned for one shape, applied to a genuinely different shape -- still correct ----
    printf("--- Part 1: correctness under a shape mismatch -- still bit-for-bit correct ---\n\n");
    LoopNest smallNest{{Loop{"dim0", 2}, Loop{"dim1", 2}}};
    LoopNest largeNest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};

    std::vector<float> smallA(static_cast<size_t>(smallNest.totalIterations()));
    for (size_t i = 0; i < smallA.size(); ++i) smallA[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> smallReference(smallA.size());
    for (size_t i = 0; i < smallA.size(); ++i) smallReference[i] = smallA[i] * 2.0f + 1.0f;

    Schedule foreignSchedule = runHybridAutotuner(smallNest, smallA, smallReference);
    printf("  schedule actually found FOR the small shape %s:\n", loopNestKey(smallNest).c_str());
    printf("    %s\n\n", scheduleStr(smallNest, foreignSchedule).c_str());

    long long largeDim0 = largeNest.loops[0].extent, largeDim1 = largeNest.loops[1].extent;
    std::vector<float> largeA(static_cast<size_t>(largeDim0 * largeDim1));
    for (size_t i = 0; i < largeA.size(); ++i) largeA[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> largeReference(largeA.size());
    for (size_t i = 0; i < largeA.size(); ++i) largeReference[i] = largeA[i] * 2.0f + 1.0f;

    std::string foreignSrc = generateScheduleFunction(largeNest, foreignSchedule);
    std::string log1;
    compileToSharedLibrary(foreignSrc, "/tmp/061_foreign.cpp", "/tmp/061_foreign.so", log1);
    JitModule foreignMod("/tmp/061_foreign.so");
    ComputeFn2D foreignFn = foreignMod.getFunction<ComputeFn2D>("compute");
    std::vector<float> foreignOut(largeA.size());
    foreignFn(largeA.data(), foreignOut.data());
    bool foreignCorrect = arraysExactlyEqual(foreignOut, largeReference);
    printf("  applying that SAME schedule, unmodified, to the genuinely different shape %s:\n", loopNestKey(largeNest).c_str());
    printf("    %s -- bit-for-bit correct: %s\n", scheduleStr(largeNest, foreignSchedule).c_str(), foreignCorrect ? "yes" : "NO");
    printf("\nself-check: a schedule tuned for a %lldx%lld shape, applied without modification to a %lldx%lld\n",
           smallNest.loops[0].extent, smallNest.loops[1].extent, largeNest.loops[0].extent, largeNest.loops[1].extent);
    printf("shape, still produces the exact correct answer (%s). This is not luck: generateScheduleFunction()\n",
           foreignCorrect ? "confirmed" : "MISMATCH");
    printf("(Chapter 23.2) always derives its own loop bounds from the real LoopNest passed to it, never from\n");
    printf("the Schedule's own cached tile sizes -- a shape mismatch cannot corrupt the answer, ONLY the\n");
    printf("performance, for any two nests with the same number of loops.\n");

    // ---- Part 2: performance under a shape mismatch -- real, measured, honestly reported ----
    printf("\n--- Part 2: performance under a shape mismatch -- real measurement, whichever way it lands ---\n\n");
    Schedule properSchedule = runHybridAutotuner(largeNest, largeA, largeReference);
    printf("  schedule actually found FOR the large shape %s:\n", loopNestKey(largeNest).c_str());
    printf("    %s\n\n", scheduleStr(largeNest, properSchedule).c_str());

    auto measureUs = [&](const Schedule& s) -> double {
        std::string src = generateScheduleFunction(largeNest, s);
        std::string log;
        compileToSharedLibrary(src, "/tmp/061_measure.cpp", "/tmp/061_measure.so", log);
        JitModule mod("/tmp/061_measure.so");
        ComputeFn2D fn = mod.getFunction<ComputeFn2D>("compute");
        std::vector<float> out(largeA.size());
        double bestMs = std::numeric_limits<double>::max();
        for (int trial = 0; trial < 9; ++trial) {
            auto start = std::chrono::steady_clock::now();
            for (long long rep = 0; rep < 2000000; ++rep) fn(largeA.data(), out.data());
            auto end = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            if (ms < bestMs) bestMs = ms;
        }
        if (!arraysExactlyEqual(out, largeReference)) throw std::runtime_error("measureUs: mismatch");
        return bestMs * 1000.0 / 2000000.0;
    };
    double foreignUs = measureUs(foreignSchedule);
    double properUs = measureUs(properSchedule);
    printf("  foreign schedule (tuned for %s, applied to %s): %.4f us/call\n",
           loopNestKey(smallNest).c_str(), loopNestKey(largeNest).c_str(), foreignUs);
    printf("  proper schedule  (tuned directly for %s):        %.4f us/call\n", loopNestKey(largeNest).c_str(), properUs);
    printf("\nself-check: correctness held in Part 1 either way -- this is purely a real, measured performance\n");
    printf("comparison, reported honestly whichever way it lands. Nothing about a shape mismatch guarantees a\n");
    printf("large gap for every possible pair of shapes -- some foreign schedules will happen to still be\n");
    printf("decent choices -- but nothing about it guarantees a GOOD one either, which is exactly why a cache\n");
    printf("that quietly served a foreign schedule as if it had been tuned for the real shape would be trading\n");
    printf("away a real, unmeasured amount of performance without ever admitting it did so.\n");

    // ---- Part 3: a coarse key can report a confident HIT for the WRONG shape ----
    printf("\n--- Part 3: a coarse key -- total element count only -- silently collides two real shapes ---\n\n");
    LoopNest shapeA{{Loop{"dim0", 4}, Loop{"dim1", 8}}};
    LoopNest shapeB{{Loop{"dim0", 8}, Loop{"dim1", 4}}};
    printf("  shapeA = %s, totalIterations = %lld\n", loopNestKey(shapeA).c_str(), shapeA.totalIterations());
    printf("  shapeB = %s, totalIterations = %lld\n", loopNestKey(shapeB).c_str(), shapeB.totalIterations());
    printf("  coarseKey(shapeA)  = \"%s\"\n", coarseKey(shapeA).c_str());
    printf("  coarseKey(shapeB)  = \"%s\"\n", coarseKey(shapeB).c_str());
    printf("  loopNestKey(shapeA) = \"%s\"\n", loopNestKey(shapeA).c_str());
    printf("  loopNestKey(shapeB) = \"%s\"\n", loopNestKey(shapeB).c_str());

    bool coarseCollides = (coarseKey(shapeA) == coarseKey(shapeB));
    bool fullDistinguishes = (loopNestKey(shapeA) != loopNestKey(shapeB));
    printf("\n  coarseKey collides shapeA and shapeB into ONE cache entry: %s\n", coarseCollides ? "yes" : "no");
    printf("  loopNestKey keeps shapeA and shapeB as two DISTINCT cache entries: %s\n", fullDistinguishes ? "yes" : "no");

    TuningCache coarseCache;
    coarseCache.put(coarseKey(shapeA), foreignSchedule);  // stands in for "whatever shapeA's own real winner was"
    bool wouldReportHit = coarseCache.has(coarseKey(shapeB));
    printf("\n  a TuningCache keyed by coarseKey(), asked about shapeB after only shapeA was ever tuned: %s\n",
           wouldReportHit ? "reports a confident HIT" : "reports a miss");
    printf("\nself-check: shapeA and shapeB are genuinely different LoopNests -- different per-dimension\n");
    printf("extents, different legal tile-size candidates, different memory-stride behavior (Chapter 22.2) --\n");
    printf("that merely happen to multiply out to the same total element count. A cache keyed on that total\n");
    printf("alone cannot tell them apart: it reports shapeB as already tuned the moment shapeA has been, and\n");
    printf("-- as Part 1 already proved -- the schedule it hands back will still be CORRECT, so nothing about\n");
    printf("the program's own output would ever reveal the mistake. Only Section 24.1's own loopNestKey(),\n");
    printf("keyed on the full per-dimension shape, keeps these two real problems apart.\n");

    bool allOk = foreignCorrect && coarseCollides && fullDistinguishes && wouldReportHit;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "061_what_a_cache_key_must_capture.cpp" -o "061_what_a_cache_key_must_capture" -ldl
./"061_what_a_cache_key_must_capture"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 24.2: what a cache key must capture ===

--- Part 1: correctness under a shape mismatch -- still bit-for-bit correct ---

  schedule actually found FOR the small shape dim0:2,dim1:2:
    tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=2

  applying that SAME schedule, unmodified, to the genuinely different shape dim0:6,dim1:8:
    tiles=[dim0:1,dim1:1] order=dim0>dim1 unroll=2 -- bit-for-bit correct: yes

self-check: a schedule tuned for a 2x2 shape, applied without modification to a 6x8
shape, still produces the exact correct answer (confirmed). This is not luck: generateScheduleFunction()
(Chapter 23.2) always derives its own loop bounds from the real LoopNest passed to it, never from
the Schedule's own cached tile sizes -- a shape mismatch cannot corrupt the answer, ONLY the
performance, for any two nests with the same number of loops.

--- Part 2: performance under a shape mismatch -- real measurement, whichever way it lands ---

  schedule actually found FOR the large shape dim0:6,dim1:8:
    tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

  foreign schedule (tuned for dim0:2,dim1:2, applied to dim0:6,dim1:8): 0.0296 us/call
  proper schedule  (tuned directly for dim0:6,dim1:8):        0.0218 us/call

self-check: correctness held in Part 1 either way -- this is purely a real, measured performance
comparison, reported honestly whichever way it lands. Nothing about a shape mismatch guarantees a
large gap for every possible pair of shapes -- some foreign schedules will happen to still be
decent choices -- but nothing about it guarantees a GOOD one either, which is exactly why a cache
that quietly served a foreign schedule as if it had been tuned for the real shape would be trading
away a real, unmeasured amount of performance without ever admitting it did so.

--- Part 3: a coarse key -- total element count only -- silently collides two real shapes ---

  shapeA = dim0:4,dim1:8, totalIterations = 32
  shapeB = dim0:8,dim1:4, totalIterations = 32
  coarseKey(shapeA)  = "32"
  coarseKey(shapeB)  = "32"
  loopNestKey(shapeA) = "dim0:4,dim1:8"
  loopNestKey(shapeB) = "dim0:8,dim1:4"

  coarseKey collides shapeA and shapeB into ONE cache entry: yes
  loopNestKey keeps shapeA and shapeB as two DISTINCT cache entries: yes

  a TuningCache keyed by coarseKey(), asked about shapeB after only shapeA was ever tuned: reports a confident HIT

self-check: shapeA and shapeB are genuinely different LoopNests -- different per-dimension
extents, different legal tile-size candidates, different memory-stride behavior (Chapter 22.2) --
that merely happen to multiply out to the same total element count. A cache keyed on that total
alone cannot tell them apart: it reports shapeB as already tuned the moment shapeA has been, and
-- as Part 1 already proved -- the schedule it hands back will still be CORRECT, so nothing about
the program's own output would ever reveal the mistake. Only Section 24.1's own loopNestKey(),
keyed on the full per-dimension shape, keeps these two real problems apart.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 24.2: what a cache key must capture ===

--- Part 1: correctness under a shape mismatch -- still bit-for-bit correct ---

  schedule actually found FOR the small shape dim0:2,dim1:2:
    tiles=[dim0:1,dim1:2] order=dim0>dim1 unroll=1

  applying that SAME schedule, unmodified, to the genuinely different shape dim0:6,dim1:8:
    tiles=[dim0:1,dim1:2] order=dim0>dim1 unroll=1 -- bit-for-bit correct: yes

self-check: a schedule tuned for a 2x2 shape, applied without modification to a 6x8
shape, still produces the exact correct answer (confirmed). This is not luck: generateScheduleFunction()
(Chapter 23.2) always derives its own loop bounds from the real LoopNest passed to it, never from
the Schedule's own cached tile sizes -- a shape mismatch cannot corrupt the answer, ONLY the
performance, for any two nests with the same number of loops.

--- Part 2: performance under a shape mismatch -- real measurement, whichever way it lands ---

  schedule actually found FOR the large shape dim0:6,dim1:8:
    tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

  foreign schedule (tuned for dim0:2,dim1:2, applied to dim0:6,dim1:8): 0.0086 us/call
  proper schedule  (tuned directly for dim0:6,dim1:8):        0.0091 us/call

self-check: correctness held in Part 1 either way -- this is purely a real, measured performance
comparison, reported honestly whichever way it lands. Nothing about a shape mismatch guarantees a
large gap for every possible pair of shapes -- some foreign schedules will happen to still be
decent choices -- but nothing about it guarantees a GOOD one either, which is exactly why a cache
that quietly served a foreign schedule as if it had been tuned for the real shape would be trading
away a real, unmeasured amount of performance without ever admitting it did so.

--- Part 3: a coarse key -- total element count only -- silently collides two real shapes ---

  shapeA = dim0:4,dim1:8, totalIterations = 32
  shapeB = dim0:8,dim1:4, totalIterations = 32
  coarseKey(shapeA)  = "32"
  coarseKey(shapeB)  = "32"
  loopNestKey(shapeA) = "dim0:4,dim1:8"
  loopNestKey(shapeB) = "dim0:8,dim1:4"

  coarseKey collides shapeA and shapeB into ONE cache entry: yes
  loopNestKey keeps shapeA and shapeB as two DISTINCT cache entries: yes

  a TuningCache keyed by coarseKey(), asked about shapeB after only shapeA was ever tuned: reports a confident HIT

self-check: shapeA and shapeB are genuinely different LoopNests -- different per-dimension
extents, different legal tile-size candidates, different memory-stride behavior (Chapter 22.2) --
that merely happen to multiply out to the same total element count. A cache keyed on that total
alone cannot tell them apart: it reports shapeB as already tuned the moment shapeA has been, and
-- as Part 1 already proved -- the schedule it hands back will still be CORRECT, so nothing about
the program's own output would ever reveal the mistake. Only Section 24.1's own loopNestKey(),
keyed on the full per-dimension shape, keeps these two real problems apart.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which schedule measures faster, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! warning "[COMMON TRAP] correctness surviving a mistake is not the same as the mistake being safe"
    It would be easy to read Part 1's own finding -- a shape-mismatched schedule still produces the exact correct answer -- as license to be casual about cache keys, on the theory that "worst case, it's just slower." Part 3's own finding is the direct rebuttal: a coarse key does not merely risk slower performance, it risks a cache that CONFIDENTLY REPORTS SUCCESS while silently serving the wrong problem's own answer, with no signal anywhere in the cache's own interface -- not an exception, not a warning, not a different return type for "close enough" versus "exact match" -- to reveal it. A system that fails by being slower is a performance bug; a system that fails by looking like it worked is a much harder one to ever notice at all.

## 24.3 Caching Across Machines: Why the Key Needs a Target Too

### Intuition

Chapter 23.3's own real, honest finding -- reported at the time as a machine-specific result, not yet named as a caching problem -- is the premise this capstone makes concrete: the hybrid autotuner found a DIFFERENT winning schedule on the cloud sandbox than on the device, for the identical `LoopNest [dim0:6, dim1:8]`. A cache keyed only by shape, exactly as Section 24.1 built it, cannot represent that difference at all -- it has exactly one slot for `dim0:6,dim1:8`, and whichever real machine happens to populate it first silently becomes the answer every OTHER machine reads back, with nothing in the cache's own interface distinguishing "tuned for you" from "tuned for someone else."

### Background

This section does not merely restate Chapter 23.3's own two numbers -- it takes both of that section's own real winning `Schedule` values, embedded here as literals cited directly to Section 23.3, and measures them AGAINST EACH OTHER for real, on whichever real machine actually runs this section's own code. Whichever machine that is, one of the two schedules is that machine's own real winner and the other is the other machine's real winner, and the section reports, honestly, which one this machine's own clock finds faster -- not assuming the answer transfers, testing it directly. `targetId()` names the real architecture this program is actually compiled for using the same predefined compiler macros (`__aarch64__`, `__x86_64__`) a real build system would check, resolved at compile time rather than guessed at runtime, because the target is a fact about the machine, not something that needs detecting.

`loopNestKeyForTarget()` closes the chapter by extending Section 24.1's own `loopNestKey()` with exactly that target string, joined by a separator that can never appear inside either half. The section's own real demonstration writes two entries -- one keyed to `x86_64`, one to `aarch64` -- into a SINGLE shared cache file, exactly as two real machines sharing one tuning-log file in a real deployment would, and shows both surviving together: whichever machine loads that shared file finds its OWN real entry under its OWN key, and the OTHER machine's own entry remains untouched, present, and available the next time THAT machine runs. This is not a design this book is adopting on convention -- it is the same (workload, target) pairing TVM's own AutoTVM tuning logs already use in real production, cited since Chapter 3's own survey, now arrived at here by this book's own direct, real, cross-machine measurement rather than only by citation.

```text
loopNestKeyForTarget(nest, target) = loopNestKey(nest) + "@" + target

  ONE shared cache file:
    "dim0:6,dim1:8@x86_64"  -> cloud sandbox's own real winner
    "dim0:6,dim1:8@aarch64" -> device's own real winner

  Whichever machine actually loads this file:
    looks up ITS OWN key (targetId(), resolved at compile time)
    finds its OWN real entry -- never the other machine's
    the OTHER machine's own entry survives, untouched, in the SAME file

The identical LoopNest shape, two real machines, two real winning
schedules, one real shared file -- exactly TVM's own real (workload,
target) tuning-log design (Chapter 3), motivated here by this book's
own direct measurement of what happens without it.
```

```cpp
// Chapter 24: Caching and Reusing Autotuning Results
// 062_caching_across_machines_why_the_key_needs_a_target_too.cpp
//
// Section 24.3 (capstone) -- Section 24.2 proved a cache key must capture
// a LoopNest's own full shape, not a coarse summary of it, or two
// genuinely different problems collide into one cache entry. This
// section closes Part 5 on a real problem this book has already
// measured without naming it: Chapter 23.3's own hybrid autotuner found
// a DIFFERENT winning schedule on the cloud sandbox than on the device,
// for the IDENTICAL LoopNest [dim0:6, dim1:8] -- cloud favored
// tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4; the device favored
// tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2. A cache keyed only by
// shape (Section 24.1's own loopNestKey()) cannot represent that
// difference at all -- whichever machine happens to populate the entry
// first "wins" it for every machine that reads it afterward. This
// section measures, for real, on whichever machine actually runs this
// program, what it costs to reuse a schedule found on a DIFFERENT real
// machine -- then extends the cache key with a target identifier,
// exactly the (workload, target) pairing TVM's own AutoTVM tuning logs
// already use in real production (Chapter 3's survey), so two machines
// can share one cache file without overwriting each other's own real,
// machine-specific answer.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 062_caching_across_machines_why_the_key_needs_a_target_too.cpp -o 062_driver -ldl
// Run:     ./062_driver
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <sstream>
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
using LoopOrder = std::vector<int>;
struct Schedule {
    std::vector<long long> tileSizePerLoop;
    LoopOrder loopOrder;
    long long unrollFactor;
};
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

// ==================== generateScheduleFunction() (Chapter 23.2, unchanged) ====================

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

// ==================== loopNestKey() / serialize / TuningCache (Section 24.1, unchanged) ====================

static std::string loopNestKey(const LoopNest& nest) {
    std::string out;
    for (size_t i = 0; i < nest.loops.size(); ++i) {
        if (i) out += ",";
        out += nest.loops[i].dimName + ":" + std::to_string(nest.loops[i].extent);
    }
    return out;
}
static std::string serializeSchedule(const Schedule& s) {
    std::string out;
    for (size_t i = 0; i < s.tileSizePerLoop.size(); ++i) {
        if (i) out += "-";
        out += std::to_string(s.tileSizePerLoop[i]);
    }
    out += "|";
    for (size_t i = 0; i < s.loopOrder.size(); ++i) {
        if (i) out += "-";
        out += std::to_string(s.loopOrder[i]);
    }
    out += "|" + std::to_string(s.unrollFactor);
    return out;
}
static Schedule deserializeSchedule(const std::string& text) {
    Schedule s;
    std::stringstream ss(text);
    std::string tilesPart, orderPart, unrollPart;
    std::getline(ss, tilesPart, '|');
    std::getline(ss, orderPart, '|');
    std::getline(ss, unrollPart, '|');
    std::stringstream tss(tilesPart);
    std::string tok;
    while (std::getline(tss, tok, '-')) s.tileSizePerLoop.push_back(std::stoll(tok));
    std::stringstream oss(orderPart);
    while (std::getline(oss, tok, '-')) s.loopOrder.push_back(std::stoi(tok));
    s.unrollFactor = std::stoll(unrollPart);
    return s;
}
class TuningCache {
public:
    void load(const std::string& path) {
        entries_.clear();
        std::ifstream f(path);
        if (!f) return;
        std::string line;
        while (std::getline(f, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            entries_[line.substr(0, eq)] = line.substr(eq + 1);
        }
    }
    void save(const std::string& path) const {
        std::ofstream f(path);
        for (const auto& kv : entries_) f << kv.first << "=" << kv.second << "\n";
    }
    bool has(const std::string& key) const { return entries_.count(key) != 0; }
    Schedule get(const std::string& key) const { return deserializeSchedule(entries_.at(key)); }
    void put(const std::string& key, const Schedule& schedule) { entries_[key] = serializeSchedule(schedule); }
    size_t size() const { return entries_.size(); }
private:
    std::map<std::string, std::string> entries_;
};

// ==================== Section 24.3: caching across machines ====================

// targetId() names the real architecture this program is actually
// compiled for, using the same predefined compiler macros a real build
// system would check -- no runtime detection needed, because the target
// is fixed at compile time. This book has exactly two real ones so far.
static std::string targetId() {
#if defined(__aarch64__)
    return "aarch64";
#elif defined(__x86_64__)
    return "x86_64";
#else
    return "unknown";
#endif
}

// loopNestKeyForTarget() extends Section 24.1's own loopNestKey() with
// the target this entry was actually measured on -- the (workload,
// target) pairing TVM's own AutoTVM tuning logs already use in real
// production (Chapter 3), motivated directly by what Part 1 below
// measures.
static std::string loopNestKeyForTarget(const LoopNest& nest, const std::string& target) {
    return loopNestKey(nest) + "@" + target;
}

static double measureUs(const LoopNest& nest, const Schedule& s, const std::vector<float>& a, const std::vector<float>& reference) {
    std::string src = generateScheduleFunction(nest, s);
    std::string log;
    compileToSharedLibrary(src, "/tmp/062_measure.cpp", "/tmp/062_measure.so", log);
    JitModule mod("/tmp/062_measure.so");
    ComputeFn2D fn = mod.getFunction<ComputeFn2D>("compute");
    std::vector<float> out(a.size());
    double bestMs = std::numeric_limits<double>::max();
    for (int trial = 0; trial < 9; ++trial) {
        auto start = std::chrono::steady_clock::now();
        for (long long rep = 0; rep < 2000000; ++rep) fn(a.data(), out.data());
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < bestMs) bestMs = ms;
    }
    if (!arraysExactlyEqual(out, reference)) throw std::runtime_error("measureUs: mismatch");
    return bestMs * 1000.0 / 2000000.0;
}

int main() {
    printf("=== Section 24.3: caching across machines -- why the key needs a target too ===\n\n");
    printf("this program's own real target (compile-time, via predefined macros): %s\n\n", targetId().c_str());

    LoopNest nest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};
    long long dim0Extent = nest.loops[0].extent, dim1Extent = nest.loops[1].extent;
    std::vector<float> a(static_cast<size_t>(dim0Extent * dim1Extent));
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> reference(a.size());
    for (size_t i = 0; i < a.size(); ++i) reference[i] = a[i] * 2.0f + 1.0f;

    // ---- Part 1: Chapter 23.3's own two REAL, machine-specific hybrid-autotuner winners ----
    printf("--- Part 1: Chapter 23.3's own real winners, both measured for real on THIS machine ---\n\n");
    // Both Schedule literals below are exactly what Section 23.3's own
    // hybrid autotuner actually found, real and measured, on each real
    // machine -- not illustrative placeholders.
    Schedule cloudWinner{{6, 8}, {0, 1}, 4};   // Chapter 23.3's own real cloud-sandbox (x86-64) winner
    Schedule deviceWinner{{6, 4}, {0, 1}, 2};  // Chapter 23.3's own real device (aarch64) winner
    printf("  cloud sandbox's own real winner (Section 23.3):  %s\n", scheduleStr(nest, cloudWinner).c_str());
    printf("  device's own real winner        (Section 23.3):  %s\n\n", scheduleStr(nest, deviceWinner).c_str());

    double cloudWinnerUs = measureUs(nest, cloudWinner, a, reference);
    double deviceWinnerUs = measureUs(nest, deviceWinner, a, reference);
    printf("  measured HERE, on this program's own real machine (%s):\n", targetId().c_str());
    printf("    cloud sandbox's schedule:  %.4f us/call\n", cloudWinnerUs);
    printf("    device's schedule:         %.4f us/call\n", deviceWinnerUs);

    bool nativeIsCloud = (targetId() == "x86_64");
    double nativeUs = nativeIsCloud ? cloudWinnerUs : deviceWinnerUs;
    double foreignUs = nativeIsCloud ? deviceWinnerUs : cloudWinnerUs;
    printf("\nself-check: both schedules produce the exact correct answer on this machine, exactly as Section\n");
    printf("24.2 already proved a shape-appropriate schedule always must -- a schedule found on a DIFFERENT\n");
    printf("real machine is never a correctness risk. This machine's OWN real winner measures %.4f us/call\n",
           nativeUs);
    printf("here; the OTHER real machine's own winner measures %.4f us/call here -- a real, measured\n", foreignUs);
    printf("difference this program's own clock actually recorded, not a hypothetical one.\n");

    // ---- Part 2: a cache key with a target -- two real entries, one shared file, no collision ----
    printf("\n--- Part 2: loopNestKeyForTarget() -- two real entries in ONE shared cache file, no collision ---\n\n");
    std::string sharedCachePath = "/tmp/062_shared_cross_machine_cache.txt";
    remove(sharedCachePath.c_str());

    TuningCache sharedCache;
    sharedCache.load(sharedCachePath);
    // Simulates what a real cloud-sandbox run and a real device run would
    // each, independently, have already put into this SAME shared file.
    sharedCache.put(loopNestKeyForTarget(nest, "x86_64"), cloudWinner);
    sharedCache.put(loopNestKeyForTarget(nest, "aarch64"), deviceWinner);
    sharedCache.save(sharedCachePath);
    printf("  wrote %zu real entries to one shared file: \"%s\" and \"%s\"\n",
           sharedCache.size(), loopNestKeyForTarget(nest, "x86_64").c_str(), loopNestKeyForTarget(nest, "aarch64").c_str());

    TuningCache thisMachineCache;  // a fresh object, as if THIS program had just started and loaded the shared file
    thisMachineCache.load(sharedCachePath);
    std::string myKey = loopNestKeyForTarget(nest, targetId());
    bool hitForMyTarget = thisMachineCache.has(myKey);
    Schedule myEntry = hitForMyTarget ? thisMachineCache.get(myKey) : Schedule{};
    bool myEntryCorrect = hitForMyTarget
        && myEntry.tileSizePerLoop == (nativeIsCloud ? cloudWinner.tileSizePerLoop : deviceWinner.tileSizePerLoop)
        && myEntry.loopOrder == (nativeIsCloud ? cloudWinner.loopOrder : deviceWinner.loopOrder)
        && myEntry.unrollFactor == (nativeIsCloud ? cloudWinner.unrollFactor : deviceWinner.unrollFactor);
    printf("  this machine (target=\"%s\") looks up its OWN key \"%s\" in the shared file: %s, entry matches\n",
           targetId().c_str(), myKey.c_str(), hitForMyTarget ? "HIT" : "MISS");
    printf("  this machine's own real winner exactly: %s\n\n", myEntryCorrect ? "yes" : "NO");

    std::string otherTarget = nativeIsCloud ? "aarch64" : "x86_64";
    std::string otherKey = loopNestKeyForTarget(nest, otherTarget);
    bool otherStillPresent = thisMachineCache.has(otherKey);
    printf("  the OTHER real machine's own entry, key \"%s\", is still present in the SAME shared file: %s\n",
           otherKey.c_str(), otherStillPresent ? "yes, untouched" : "NO -- lost");

    printf("\nself-check: one shared cache FILE, two real (workload, target) entries, keyed by\n");
    printf("loopNestKeyForTarget() -- extending Section 24.1's own loopNestKey() with exactly the target\n");
    printf("identifier this section's own Part 1 measurement just showed matters. Each machine reads back\n");
    printf("its OWN real winner (%s) without ever colliding with, or overwriting, the other machine's own\n",
           myEntryCorrect ? "confirmed" : "MISMATCH");
    printf("entry in the identical file (%s). This is precisely the (workload, target) pairing TVM's own\n",
           otherStillPresent ? "confirmed" : "MISMATCH");
    printf("AutoTVM tuning logs already use in real production (Chapter 3's own survey) -- this section's own\n");
    printf("real, measured cross-machine gap in Part 1 is exactly the reason that design decision is\n");
    printf("necessary, not merely a convention this book is choosing to follow without evidence.\n");

    bool allOk = myEntryCorrect && otherStillPresent;
    return allOk ? 0 : 1;
}
```

```bash
g++ -std=c++17 -Wall -Wextra -O2 "062_caching_across_machines_why_the_key_needs_a_target_too.cpp" -o "062_caching_across_machines_why_the_key_needs_a_target_too" -ldl
./"062_caching_across_machines_why_the_key_needs_a_target_too"
```

**Output (cloud sandbox, x86-64 -- real, machine-specific timing):**

```text
=== Section 24.3: caching across machines -- why the key needs a target too ===

this program's own real target (compile-time, via predefined macros): x86_64

--- Part 1: Chapter 23.3's own real winners, both measured for real on THIS machine ---

  cloud sandbox's own real winner (Section 23.3):  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4
  device's own real winner        (Section 23.3):  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

  measured HERE, on this program's own real machine (x86_64):
    cloud sandbox's schedule:  0.0218 us/call
    device's schedule:         0.0221 us/call

self-check: both schedules produce the exact correct answer on this machine, exactly as Section
24.2 already proved a shape-appropriate schedule always must -- a schedule found on a DIFFERENT
real machine is never a correctness risk. This machine's OWN real winner measures 0.0218 us/call
here; the OTHER real machine's own winner measures 0.0221 us/call here -- a real, measured
difference this program's own clock actually recorded, not a hypothetical one.

--- Part 2: loopNestKeyForTarget() -- two real entries in ONE shared cache file, no collision ---

  wrote 2 real entries to one shared file: "dim0:6,dim1:8@x86_64" and "dim0:6,dim1:8@aarch64"
  this machine (target="x86_64") looks up its OWN key "dim0:6,dim1:8@x86_64" in the shared file: HIT, entry matches
  this machine's own real winner exactly: yes

  the OTHER real machine's own entry, key "dim0:6,dim1:8@aarch64", is still present in the SAME shared file: yes, untouched

self-check: one shared cache FILE, two real (workload, target) entries, keyed by
loopNestKeyForTarget() -- extending Section 24.1's own loopNestKey() with exactly the target
identifier this section's own Part 1 measurement just showed matters. Each machine reads back
its OWN real winner (confirmed) without ever colliding with, or overwriting, the other machine's own
entry in the identical file (confirmed). This is precisely the (workload, target) pairing TVM's own
AutoTVM tuning logs already use in real production (Chapter 3's own survey) -- this section's own
real, measured cross-machine gap in Part 1 is exactly the reason that design decision is
necessary, not merely a convention this book is choosing to follow without evidence.
```

**Output (device, aarch64 Linux VM -- real, machine-specific timing):**

```text
=== Section 24.3: caching across machines -- why the key needs a target too ===

this program's own real target (compile-time, via predefined macros): aarch64

--- Part 1: Chapter 23.3's own real winners, both measured for real on THIS machine ---

  cloud sandbox's own real winner (Section 23.3):  tiles=[dim0:6,dim1:8] order=dim0>dim1 unroll=4
  device's own real winner        (Section 23.3):  tiles=[dim0:6,dim1:4] order=dim0>dim1 unroll=2

  measured HERE, on this program's own real machine (aarch64):
    cloud sandbox's schedule:  0.0120 us/call
    device's schedule:         0.0088 us/call

self-check: both schedules produce the exact correct answer on this machine, exactly as Section
24.2 already proved a shape-appropriate schedule always must -- a schedule found on a DIFFERENT
real machine is never a correctness risk. This machine's OWN real winner measures 0.0088 us/call
here; the OTHER real machine's own winner measures 0.0120 us/call here -- a real, measured
difference this program's own clock actually recorded, not a hypothetical one.

--- Part 2: loopNestKeyForTarget() -- two real entries in ONE shared cache file, no collision ---

  wrote 2 real entries to one shared file: "dim0:6,dim1:8@x86_64" and "dim0:6,dim1:8@aarch64"
  this machine (target="aarch64") looks up its OWN key "dim0:6,dim1:8@aarch64" in the shared file: HIT, entry matches
  this machine's own real winner exactly: yes

  the OTHER real machine's own entry, key "dim0:6,dim1:8@x86_64", is still present in the SAME shared file: yes, untouched

self-check: one shared cache FILE, two real (workload, target) entries, keyed by
loopNestKeyForTarget() -- extending Section 24.1's own loopNestKey() with exactly the target
identifier this section's own Part 1 measurement just showed matters. Each machine reads back
its OWN real winner (confirmed) without ever colliding with, or overwriting, the other machine's own
entry in the identical file (confirmed). This is precisely the (workload, target) pairing TVM's own
AutoTVM tuning logs already use in real production (Chapter 3's own survey) -- this section's own
real, measured cross-machine gap in Part 1 is exactly the reason that design decision is
necessary, not merely a convention this book is choosing to follow without evidence.
```

*Correctness (the bit-for-bit self-checks above) matches on both machines, as it has every chapter since 17. The millisecond figures, and in places even which schedule measures faster, are expected to differ between the two -- that is real data about two different machines, not a discrepancy to reconcile.*

!!! note "What Part 5 built, start to finish"
    Chapter 21 proved the search space is real and already too large to brute-force one worked example at a time. Chapter 22 built two honest, imperfect cost models and tested them against real hardware. Chapter 23 closed the interpreter gap those models were tested through and built a real, working hybrid autotuner, checked honestly against exhaustive search. This chapter gave that autotuner a memory: a real, persistent cache, a demonstrated understanding of what its own key must and must not lose, and a closing, measured proof that the key needs to know not just the shape being tuned but the machine it was tuned on. Every number in Part 5, from Chapter 21's own 128 enumerated schedules to this chapter's own real cross-machine measurement, came from code that actually ran -- interpreted, then compiled, then cached -- on two real, different machines, with correctness checked exactly and timing reported honestly, including every place the two machines disagreed.

## Chapter Summary

This chapter closed Part 5 by giving Chapter 23's own hybrid autotuner a real memory. Section 24.1 built `TuningCache`, a persistent, file-backed key/value store keyed by `loopNestKey()`, and `getOrAutotune()`, demonstrated end to end with a real cache MISS (paying Chapter 23's own full hybrid-autotuning cost) followed by a real cache HIT from a genuinely different process, loaded from the exact file the first run wrote. Section 24.2 examined what the cache key must capture, finding two separable risks: a shape mismatch cannot corrupt correctness, because `generateScheduleFunction()` (Chapter 23.2) always derives its own loop bounds from the real `LoopNest` it is given rather than from a schedule's own cached tile sizes, but it can still cost real, unpredictable performance; and a COARSE key -- total element count alone -- can silently collide two genuinely different real shapes into one entry, with a confident cache HIT hiding the mistake from the program's own output entirely. Section 24.3 closed the chapter and Part 5 by measuring Chapter 23.3's own two real, machine-specific hybrid-autotuner winners against each other, for real, on whichever machine ran the section's own code, and extended the cache key with `loopNestKeyForTarget()` -- a real, compile-time architecture identifier -- demonstrating one shared cache file correctly holding both machines' own real answers without collision, the same (workload, target) design TVM's own AutoTVM tuning logs already use in real production, cited since Chapter 3.

## Self-Check Questions

1. Why does Section 24.1's own demonstration construct a SECOND, genuinely different `TuningCache` object rather than reusing the first one in memory to show the benefit of caching?
2. `TuningCache::load()` silently does nothing if the file at `path` does not exist yet. Why is that the correct behavior, rather than an error?
3. Section 24.2 finds that a schedule tuned for one shape, applied to a different shape, always produces the correct answer. What specific property of `generateScheduleFunction()` (Chapter 23.2) makes that true, and would it still be true if the two shapes had a different NUMBER of loops?
4. Section 24.2's own real measurement of a shape-mismatched schedule's performance does not always show a large gap. Why does the chapter treat that as a meaningful finding rather than a disappointing one?
5. Why is a coarse cache key (Section 24.2's own `coarseKey()`, total element count only) a more dangerous mistake than simply choosing a bad tile size, given that both can hurt performance?
6. In Section 24.3, why does `targetId()` use compile-time predefined macros (`__aarch64__`, `__x86_64__`) rather than a runtime check?
7. `loopNestKeyForTarget()` joins the shape key and the target with an `@` character. What would go wrong if the two keys were concatenated with no separator at all, for some hypothetical dimension name or target string?
8. Section 24.3 measures Chapter 23.3's own two real winning schedules against each other on whichever machine actually runs the code, rather than simply asserting that a schedule should be re-tuned per machine. What does actually measuring it add that the assertion alone would not?

## Where We Go Next

Part 5 closes here, with a real, working autotoolchain: a search space (Chapter 21), honest cost models (Chapter 22), a hybrid autotuner built on real compiled code (Chapter 23), and a real, persistent, correctly-keyed cache (this chapter). Part 6 turns from CUDA Hammer's own internals to real, existing compilers -- case studies grounded in the same evidence-first standard this book has held since Chapter 1, examining how production systems such as XLA, TVM, and Triton (first surveyed all the way back in Chapter 3) make the same kinds of decisions this book's own chapters have made from scratch, at a scale and with engineering resources this book's own toy examples were never trying to match.

## Worked Solutions

1. An in-memory map already provides the benefit of skipping repeated autotuning WITHIN one process, with no file I/O needed at all -- that would prove nothing about what a real persistent cache adds. The real boundary a tuning cache has to survive is a brand new program run: a genuinely different process, with an empty in-memory map, that has to load a PAST run's own result from a real file. Constructing a second, independent `TuningCache` object and loading it from the first object's own saved file is the only test that actually exercises that boundary.
2. A cache with nothing cached yet is the normal starting state for the very first real run of a new shape -- not a malfunction. Treating a missing file as an error would make the cache unusable on its own first call for any shape, defeating the purpose of `getOrAutotune()`'s own fallback to a real autotuning MISS.
3. `generateScheduleFunction()` always computes its own loop bounds (`outerExtent`, `innerExtent`, and every `std::min()`-clamped block boundary) from the real `LoopNest` passed into it, and only ever reads the `Schedule`'s own tile sizes as a STEP size -- never as a bound to trust directly. That guarantee is specific to two nests with the SAME number of loops, since the generator is written for exactly two nested loops (an outer and an inner); a schedule built for a different number of dimensions would not even have the right number of `tileSizePerLoop`/`loopOrder` entries to apply at all.
4. If shape-mismatched performance always showed a large, consistent gap, that would actually be a MORE forgiving finding -- a predictable penalty a real system could budget for. The genuinely harder finding is that the gap is inconsistent: sometimes small, sometimes not, with no way to know in advance which, for a given pair of shapes. That unpredictability, not a guaranteed loss, is what makes silently serving a foreign schedule a real risk rather than a bounded, acceptable one.
5. A bad tile size, chosen by a real (if imperfect) autotuning search for the ACTUAL shape being computed, is still an honest answer to the right question -- it may be a suboptimal schedule, but it was tuned for this problem. A coarse-key collision serves a schedule that was never tuned for this shape at all, while reporting a confident cache HIT that implies otherwise -- the system is not merely wrong, it is wrong while claiming to be right, with nothing in its own interface able to distinguish that case from a genuine, correctly-keyed hit.
6. The target architecture a program is compiled for is fixed at compile time -- it does not change while the program runs, and it is already known with certainty by the compiler itself through its own predefined macros. A runtime check (parsing `uname` output, for instance) would be doing real work to rediscover a fact the compiler already had for free, and would risk disagreeing with what the code was actually compiled to run as.
7. Without a separator, a shape key ending in a target-like suffix could concatenate with an actual target string to form a key indistinguishable from a different, unintended shape-plus-target combination -- for example, a dimension named in a way that happens to end the same way a target string begins. The `@` character is chosen specifically because it never appears inside a `loopNestKey()`'s own output (built only from dimension names, digits, and colons) or inside a target string like `x86_64`/`aarch64`, so the two halves can always be split back apart unambiguously.
8. Asserting that a schedule should be re-tuned per machine is a reasonable-sounding claim that could still turn out to be overly cautious -- perhaps the two machines' own real winners are close enough in practice that reusing one on the other would cost little. Actually measuring both schedules against each other, for real, on the machine that runs the code, replaces that assumption with a real number: this section's own measurement shows the gap is real and, on at least one of this book's own two machines, large enough to matter -- the same standard of evidence this book has applied to every other claim since Chapter 1, now applied to the design of the cache itself.

---

**Sources cited in this chapter:**

TVM's own real AutoTVM tuning-log practice, keyed by (workload, target), was introduced in Chapter 3's survey of real ML compilers and is the direct, cited precedent for this entire chapter's own design -- Section 24.3's own capstone measurement is this book's first time arriving at that same design by direct evidence rather than only by citation. `TuningCache`, `loopNestKey()`/`loopNestKeyForTarget()`, `serializeSchedule()`/`deserializeSchedule()`, and `getOrAutotune()` are all original to this book, building directly on Chapter 20's own `JitModule`/`compileToSharedLibrary()`, Chapter 21's own `Schedule`/`enumerateSchedules()`, Chapter 22's own `estimateScheduleCost()`, and Chapter 23's own `generateScheduleFunction()` and hybrid autotuner.
