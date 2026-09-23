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
