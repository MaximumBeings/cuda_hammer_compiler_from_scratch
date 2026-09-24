// Appendix F: From torch.compile and TVM's Python API to C++: A Rosetta
// Stone
// 108_the_same_autotune_and_cache_problem_through_cuda_hammers_own_real_hybrid_autotuner.cpp
//
// Section F.2 (capstone) -- the CUDA Hammer half of this appendix's second
// side-by-side pair. File 107 measures TVM's own real split+vectorize
// schedule on a real n=48 vector-add compute, then commits that real
// measurement to TVM's own real `meta_schedule.database.JSONDatabase`
// (Chapter 26.3's own File 068, reused unchanged), and reads it back with a
// FRESH `Database` object -- a real MISS-then-HIT shape. This file runs the
// exact same real shape through CUDA Hammer's own machinery instead:
// Chapter 23.3's own real hybrid autotuner (`estimateScheduleCost()` ranks
// every legal `Schedule`, the cheapest 8 are compiled with a real `g++`
// subprocess and carefully measured) and Chapter 24.1's own real,
// file-backed `TuningCache`, on `LoopNest{{"dim0",6},{"dim1",8}}` -- the
// SAME 6x8=48-element problem size TVM's own File 068/107 measures, even
// though the two systems represent it differently (TVM: one flat 1D
// `te.compute`, split 6x8 and vectorized; CUDA Hammer: a genuinely 2D
// `LoopNest`, matching Chapter 15's own tiling model) -- a real, honest
// structural difference this section's own closing synthesis names
// directly rather than papering over. `getOrAutotune()` (unchanged from
// File 060) is run twice with two SEPARATE, freshly-constructed
// `TuningCache` objects -- run 1 starts from an empty cache and pays
// Chapter 23's own real autotuning cost in full (a MISS); run 2, a
// genuinely different object that never saw run 1's own `put()` call,
// loads the real file run 1 wrote and returns the identical schedule
// almost instantly (a HIT) -- the exact same real shape as TVM's own
// fresh-`Database`-object demonstration in File 107.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 108_the_same_autotune_and_cache_problem_through_cuda_hammers_own_real_hybrid_autotuner.cpp -o 108_driver -ldl
// Run:     ./108_driver
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

// ==================== TuningCache / runHybridAutotuner() / getOrAutotune() (Chapters 23.3/24.1, unchanged) ====================

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
        std::string cppPath = "/tmp/108_autotune_" + std::to_string(r) + ".cpp";
        std::string soPath = "/tmp/108_autotune_" + std::to_string(r) + ".so";
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
    printf("=== Appendix F.2: the same autotune-and-cache problem, through CUDA Hammer's own real hybrid autotuner ===\n\n");

    // Same 6x8=48-element problem size as File 107's own TVM run (n=48,
    // split into factors=[6,8]) -- represented here as a genuinely 2D
    // LoopNest instead of TVM's own flat 1D compute split in two.
    LoopNest nest{{Loop{"dim0", 6}, Loop{"dim1", 8}}};
    printf("LoopNest: dim0=6, dim1=8 (%lld total elements, matching TVM's own File 107 n=48)\n",
           nest.totalIterations());
    printf("real legal schedule count for this LoopNest: %zu (Chapter 21's own enumerateSchedules())\n\n",
           enumerateSchedules(nest).size());

    std::vector<float> a(static_cast<size_t>(nest.totalIterations()));
    for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<float>(i) * 0.25f - 4.0f;
    std::vector<float> reference(a.size());
    for (size_t i = 0; i < a.size(); ++i) reference[i] = a[i] * 2.0f + 1.0f;

    std::string cachePath = "/tmp/108_real_cache.txt";
    remove(cachePath.c_str());

    printf("--- run 1: a fresh TuningCache, nothing on disk yet -- pays Chapter 23's own real hybrid-autotuning cost in full ---\n\n");
    TuningCache runOneCache;
    runOneCache.load(cachePath);
    bool firstWasHit = true;
    auto firstStart = std::chrono::steady_clock::now();
    Schedule firstResult = getOrAutotune(nest, runOneCache, cachePath, a, reference, firstWasHit);
    auto firstEnd = std::chrono::steady_clock::now();
    double firstUs = std::chrono::duration<double, std::micro>(firstEnd - firstStart).count();
    printf("  run 1: %s in %.1f ms -> %s\n", firstWasHit ? "HIT" : "MISS", firstUs / 1000.0, scheduleStr(nest, firstResult).c_str());

    printf("\n--- run 2: a BRAND NEW TuningCache object -- the real behavior a new program run needs ---\n\n");
    TuningCache runTwoCache;
    runTwoCache.load(cachePath);
    bool secondWasHit = true;
    auto secondStart = std::chrono::steady_clock::now();
    Schedule secondResult = getOrAutotune(nest, runTwoCache, cachePath, a, reference, secondWasHit);
    auto secondEnd = std::chrono::steady_clock::now();
    double secondUs = std::chrono::duration<double, std::micro>(secondEnd - secondStart).count();
    printf("  run 2: %s in %.1f us -> %s\n", secondWasHit ? "HIT" : "MISS", secondUs, scheduleStr(nest, secondResult).c_str());

    bool sameSchedule = firstResult.tileSizePerLoop == secondResult.tileSizePerLoop
                         && firstResult.loopOrder == secondResult.loopOrder && firstResult.unrollFactor == secondResult.unrollFactor;
    printf("\nrun 1 was a %s, run 2 was a %s, both returned the %s schedule\n",
           firstWasHit ? "HIT" : "MISS", secondWasHit ? "HIT" : "MISS", sameSchedule ? "IDENTICAL" : "DIFFERENT");
    if (secondUs > 0.0) {
        printf("real wall-clock: run 1 = %.1f ms, run 2 = %.1f us (ratio: run 1 is %.0fx slower)\n",
               firstUs / 1000.0, secondUs, firstUs / secondUs);
    }

    bool allOk = sameSchedule && !firstWasHit && secondWasHit;
    return allOk ? 0 : 1;
}
