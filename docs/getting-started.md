# Getting Started

This book builds CUDA Hammer entirely in C++17, with three real, distinct backends by the time Part 4 is done: a plain host C++ interpreter for CUDA Hammer's own IR (useful for testing a pass before trusting its output), a vectorized CPU code generator (real AVX2/FMA or NEON, whichever the machine actually has), and a CUDA C++ code generator that emits real kernels for `nvcc`. No external compiler framework -- no LLVM, no MLIR -- is used to build CUDA Hammer itself; Appendix C covers what a real production compiler gets by building on one of those instead, once CUDA Hammer's own from-scratch version exists to compare it against.

## Installing a toolchain

A working `g++` (or `clang++`) supporting C++17 is all Part 0 through Part 3 need. Part 4 onward needs two more pieces, and this book is explicit about which of its own two authoring machines has which:

```bash
g++ --version      # C++17 host compiler, needed for everything
nvcc --version      # CUDA C++ compiler, needed from Chapter 18 onward
```

A CPU that actually supports AVX2 and FMA (check with `grep avx2 /proc/cpuinfo` on Linux, or `sysctl -a | grep -i avx` on Intel Mac) is what Chapter 19's x86 path needs to genuinely compile *and run*; a NEON-capable ARM CPU (essentially every Apple Silicon Mac and modern ARM server, confirmed by compiling and running a real `<arm_neon.h>` program against `asimd` in `/proc/cpuinfo`) is what its ARM path needs for the same. `nvcc` needs a CUDA toolkit install (`apt-get install nvidia-cuda-toolkit` on Debian/Ubuntu, or the installer from developer.nvidia.com) -- compiling for a target architecture with `nvcc -arch=sm_XX` never requires a matching physical GPU to be present, only to *run* the result.

## The honesty discipline this book follows

This book's own two authoring machines are, between them, real and different: a cloud sandbox (x86_64, real AVX2/AVX512F/FMA, real `g++`, real `clang++`, and a real, complete `nvcc` toolchain) and a connected device (aarch64, real NEON, real `g++`, **no** `nvcc`, **no** physical GPU on either machine). That split is treated as a feature, not a limitation to route around -- it means CUDA Hammer's CPU backend gets to be genuinely, fully honest in a way its GPU backend structurally cannot:

- **CUDA Hammer's IR, its pass manager, and its host-side interpreter** (no vector intrinsics, no CUDA) are plain C++. They are genuinely compiled and genuinely run wherever they're touched, with exact output locked into the page and re-verified by a fresh recompile-and-rerun before publication.
- **CUDA Hammer's vectorized CPU backend** (Chapter 19 onward) is genuinely compiled *and genuinely run*, for real, on two real, different architectures: AVX2/FMA intrinsics on the cloud sandbox, NEON intrinsics on the connected device. A claim about what fused, vectorized code actually computes is never simulated in this book -- it is measured directly, cross-checked bit-for-bit against a plain scalar reference, on real hardware, on two real instruction sets.
- **CUDA Hammer's CUDA backend** (Chapter 18 onward) is genuinely compiled with `nvcc` for a real architecture (`-arch=sm_80` unless a chapter says otherwise) and checked instead by a **host-side reference implementation**: ordinary C++ that computes the exact same result the generated kernel's own loop nest would, checked independently. This is the one honest limitation this book carries, stated once here rather than repeated on every page: neither authoring machine has a physical NVIDIA GPU, so a generated CUDA kernel's compiled correctness (does it build, does its PTX look right) is verified for real, but its *runtime* result is verified by the same reference implementation that was used to design it, not by executing the kernel itself.
- **Autotuning search and cost models** (Part 5) are run for real against whichever backend a chapter is tuning -- the CPU backend's genuine on-hardware measurements are real, measured numbers; a CUDA-kernel cost model is a real, stated formula rather than a fabricated measured number, exactly like the sibling books' own standing rule against inventing timing data.
- **Real facts this book states about other compilers** -- an XLA fusion rule, a TVM scheduling primitive, a Triton language feature -- are cited to that project's own real, current documentation or source, not asserted from memory.

Every chapter states which of the above applies to each piece of its own code, so nothing is left for a reader to guess about how a claim was actually established.

## Compile-line conventions

- Plain C++ host files (`.cpp`): `g++ -std=c++17 -Wall -Wextra -O2 file.cpp -o binary`
- AVX2/FMA CPU-codegen files: `g++ -std=c++17 -O3 -mavx2 -mfma file.cpp -o binary` (x86_64 only)
- NEON CPU-codegen files: `g++ -std=c++17 -O3 file.cpp -o binary` (NEON is baseline on AArch64, no extra flag needed)
- CUDA-generated files (`.cu`): `nvcc -arch=sm_80 file.cu -o binary`
- A JIT-backend file that compiles and loads generated code at runtime shows its exact `dlopen`/linking recipe inline where it's introduced.

## Prerequisites

This book assumes working knowledge of C++ (classes, templates, smart pointers, RAII) and enough familiarity with how a tensor operation like matrix multiplication or a convolution is actually computed to follow why a particular fusion or tiling choice helps. No prior compiler-construction experience is assumed -- Part 0 builds the general vocabulary (IR, pass, lowering, codegen) from nothing before Part 1 ever starts building CUDA Hammer itself. No prior CUDA experience is assumed either; Chapter 18 introduces exactly the subset of the CUDA programming model CUDA Hammer's own generated kernels need, in place.
