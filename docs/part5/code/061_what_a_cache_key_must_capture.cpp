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
