// Regression for finite-fixture comparisons: NaN must never pass an audit.
#define main hammer_example_main
#include "../docs/part4/code/047_a_whole_graph_vectorized_dispatch_reduction_and_a_real_fallback.cpp"
#undef main
#include <limits>
int main() {
    float nan = std::numeric_limits<float>::quiet_NaN();
    if (arraysMatch({nan}, {1.0f}) || arraysMatch({1.0f}, {nan})) return 1;
    if (!arraysMatch({1.0f}, {1.001f}) || arraysMatch({1.0f}, {2.0f})) return 2;
    std::puts("finite comparison regression: matching fixtures accepted; NaNs and mismatches rejected");
}
