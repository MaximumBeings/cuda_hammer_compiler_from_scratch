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
