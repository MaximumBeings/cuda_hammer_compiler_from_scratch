# CUDA Hammer: A Tensor Compiler in C++ -- Table of Contents (planning document)

Subtitle: From a Minimal IR to Fused, Autotuned Kernels

This is the full planned outline. Check this file first before starting any new chapter or appendix. Mirrors the sibling books' own build-verify-lock discipline (see getting-started.md for this book's own specific honesty discipline around the toolchain actually available while writing it).

## Part 0 -- Why Compile Tensor Programs At All
1. The Cost of Eager Execution
2. What a Compiler Pipeline Looks Like
3. A Tour of Real ML Compilers: XLA, TVM, and Triton

## Part 1 -- CUDA Hammer's IR: A Minimal Tensor Graph From Scratch
4. Representing Tensors and Operations as a Graph
5. Building the Graph: CUDA Hammer's Frontend
6. Graph Validation and Shape Inference
7. Printing and Visualizing CUDA Hammer's IR

## Part 2 -- Basic Optimization Passes
8. The Pass Manager: Structuring Compiler Transformations
9. Constant Folding and Dead Code Elimination
10. Common Subexpression Elimination for Tensor Graphs
11. Algebraic Simplification

## Part 3 -- Operator Fusion
12. Why Fusion Matters: Memory-Bound vs. Compute-Bound Kernels
13. Elementwise Fusion
14. Fusing Reductions
15. Loop Fusion and Tiling for Fused Kernels
16. Fusion Boundaries: What Can't Be Fused, and Why

## Part 4 -- Code Generation
17. Lowering CUDA Hammer's IR to Loops
18. Generating CUDA C++ From the Fused IR
19. Generating Vectorized CPU Code
20. A JIT Backend: Compiling and Loading Generated Code at Runtime

## Part 5 -- Autotuning
21. The Search Space: Tile Sizes, Loop Orders, and Unrolling
22. Cost Models vs. Measurement-Based Autotuning
23. Building CUDA Hammer's Autotuner
24. Caching and Reusing Tuned Schedules

## Part 6 -- Case Studies: How CUDA Hammer Compares to Real Compilers
25. XLA's HLO and Fusion Passes
26. TVM's Relay/TIR and Ansor Auto-Scheduling
27. Triton's Block-Level Programming Model
28. torch.compile and TorchInductor
29. Flash Attention as a Fusion Case Study
30. Quantization-Aware Codegen

## Appendices
- A. Installation and Setup -- Building CUDA Hammer's Toolchain
- B. Practice Quiz
- C. LLVM and MLIR: The Compiler Infrastructure You Get for Free
- D. Polyhedral Compilation and Advanced Loop Transformations
- E. Profiling and Benchmarking Generated Kernels
- F. From torch.compile and TVM's Python API to C++: A Rosetta Stone
- G. Common Failure Modes: Miscompilation, Fusion-Order Drift, and Autotuner Overfitting

## Status
Scaffolded 2026-09-21: mkdocs.yml, docs/index.md, docs/getting-started.md, this file.
Toolchain confirmed this session: cloud sandbox is x86_64, g++ 13.3.0, clang++ 18.1.3, real apt-installed nvcc 12.0 (genuine `nvcc`/`ptxas`/`cicc`, not the incomplete pip wheel the sibling books already documented), AVX2/AVX512F/FMA present in `/proc/cpuinfo`. Device (connected folder, isolated Linux VM on the user's Mac) is aarch64, g++ 11.4.0, no clang++, no nvcc -- but genuine NEON (`asimd`) confirmed by actually compiling and running an `arm_neon.h` test program there (output `3.0 3.0 3.0 3.0`, correct). This means Part 4's CPU-codegen chapter (Ch19) can genuinely compile AND RUN real vectorized kernels on two real, different architectures (AVX2/FMA on the cloud sandbox, NEON on the device) with zero simulation needed for that backend -- unlike the CUDA backend (Ch18), which compiles for a real arch via nvcc but cannot execute anywhere in this toolchain (no physical GPU on either side), the same honest limitation the sibling CUDA books already carry.
NEXT: Chapter 1, "The Cost of Eager Execution."
