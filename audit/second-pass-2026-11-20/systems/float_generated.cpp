#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

void compute_out(const std::vector<const float*>& ext, const std::vector<long long>& extCounts, float* out) {
    for (long long i0 = 0; i0 < 1; ++i0) {
        float step0 = ext[0][(0 + i0 * 1) % extCounts[0]] * ext[1][(0 + i0 * 1) % extCounts[1]];
        out[0 + i0 * 1] = step0;
    }
}

int main() {
    std::map<std::string, std::vector<float>> buf;
    buf["a"] = {100000000.000000f};
    buf["c"] = {0.000000f};
    buf["out"] = std::vector<float>(1);
    {
        std::vector<const float*> ext; std::vector<long long> extCounts;
        ext.push_back(buf["a"].data()); extCounts.push_back(1);
        ext.push_back(buf["c"].data()); extCounts.push_back(1);
        compute_out(ext, extCounts, buf["out"].data());
    }
    printf("a:");
    for (float v : buf["a"]) printf(" %.6f", v);
    printf("\n");
    printf("c:");
    for (float v : buf["c"]) printf(" %.6f", v);
    printf("\n");
    printf("out:");
    for (float v : buf["out"]) printf(" %.6f", v);
    printf("\n");
    return 0;
}
