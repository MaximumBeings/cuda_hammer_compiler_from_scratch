// Appendix A: Installation and Setup -- Building CUDA Hammer's Toolchain
// 093_how_much_of_cplusplus17_does_cuda_hammer_actually_need.cpp
//
// Section A.1 -- Every chapter in this book has compiled its own code
// with `-std=c++17` (stated in getting-started.md's own compile-line
// conventions). That flag has never once been questioned in 34
// chapters -- so before anything else, this appendix asks the honest
// question directly: does CUDA Hammer's own real source actually NEED
// C++17, or has this book been compiling with a newer standard than
// its own code requires?
//
// This file is a minimal, self-contained REPRODUCTION of the two real
// constructs a full sweep of this book's own 71 real .cpp files found
// (the sweep itself, and its exact real numbers, are reported in this
// section's own prose -- not fabricated, run for real this session):
//
//   1. `std::make_unique<T>()` -- used in 54 of this book's own 71
//      real files, starting with Graph::addNode() in File 005
//      (Chapter 4). Standard library, added in C++14.
//   2. digit-separator numeric literals like `1'000'000` -- used in 4
//      of this book's own real files (024, 025, 054, 057), each one a
//      cost-model or roofline chapter with a large element count.
//      Core language syntax, added in C++14.
//
// Neither construct needs C++17 -- both are C++14. This file compiles
// the SAME source three times, with three real `-std=` flags, and
// reports what actually happens each time -- not asserted, compiled.
//
// Compile (run three times, once per real standard flag):
//   g++ -std=c++11 -Wall -Wextra -O2 093_how_much_of_cplusplus17_does_cuda_hammer_actually_need.cpp -o 093_c11 2>&1; echo "exit: $?"
//   g++ -std=c++14 -Wall -Wextra -O2 093_how_much_of_cplusplus17_does_cuda_hammer_actually_need.cpp -o 093_c14 2>&1; echo "exit: $?"
//   g++ -std=c++17 -Wall -Wextra -O2 093_how_much_of_cplusplus17_does_cuda_hammer_actually_need.cpp -o 093_c17 2>&1; echo "exit: $?"
// Run (the two that build):
//   ./093_c14
//   ./093_c17

#include <cstdio>
#include <memory>
#include <vector>

// The same shape as Graph::addNode() (File 005, Chapter 4): allocate
// a small owned object with std::make_unique, exactly as CUDA
// Hammer's own Node storage has done since Chapter 4.
struct Node {
    int id;
    float value;
};

std::unique_ptr<Node> makeNode(int id, float value) {
    // std::make_unique<T>(...) -- C++14 (<memory>), not C++17.
    return std::make_unique<Node>(Node{id, value});
}

int main() {
    std::vector<std::unique_ptr<Node>> nodes;
    for (int i = 0; i < 3; ++i) {
        nodes.push_back(makeNode(i, static_cast<float>(i) * 1.5f));
    }

    // A digit-separator numeric literal -- the same real construct
    // File 024 (Chapter 12) and File 025 (Chapter 12) use for their
    // own real element counts. C++14 core language syntax, not C++17.
    const long long elementCount = 1'000'000;

    long long checksum = 0;
    for (const auto& n : nodes) checksum += n->id;

    printf("__cplusplus = %ldL\n", static_cast<long>(__cplusplus));
    printf("nodes allocated via std::make_unique: %zu (checksum of ids: %lld)\n",
           nodes.size(), checksum);
    printf("digit-separator literal elementCount = %lld\n", elementCount);
    printf("real result: this file compiled and ran under this standard.\n");
    return 0;
}
