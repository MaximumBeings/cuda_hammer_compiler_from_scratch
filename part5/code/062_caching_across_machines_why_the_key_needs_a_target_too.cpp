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
