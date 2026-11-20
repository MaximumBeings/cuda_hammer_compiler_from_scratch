# CUDA Hammer: A Tensor Compiler in C++

<!-- hammer:figure:book-overview:start -->
<figure class="chapter-figure" markdown="1">

[![From a tensor graph to a validated implementation](images/book-overview.svg?v=0f742bd23500)](images/book-overview.svg?v=0f742bd23500)

<figcaption>Compiler structure and validation form one continuous workflow.</figcaption>
</figure>
<!-- hammer:figure:book-overview:end -->

*From a Minimal IR to Fused, Autotuned Kernels*

An unfused eager tensor chain commonly materializes an array after each operation and submits several kernels. GPU submission can be asynchronous: the host need not wait after every launch, although operations retain their data dependencies. A compiler that sees a larger region can sometimes combine operations, reduce intermediate array traffic, and choose a useful iteration schedule. A hand-written kernel may already perform such fusion. This book builds a small teaching compiler, CUDA Hammer, from a graph representation through transformations, CPU and CUDA code generation, and schedule search. Each optimization remains subject to shape, numerical, resource and backend constraints.

This book is independent and builds the teaching compiler from Chapter 1. Downloadable sources and embedded listings accompany the examples. Its original edition recorded host execution and offline CUDA compilation; the current audit adds focused regressions and actual device validation for the stated cases. The validation record distinguishes fresh runs from historical framework transcripts, optional dependencies and unimplemented production features.

The current [validation status](validation-status.md) distinguishes fresh host/device runs from historical compiler transcripts and records the remaining limits. The [technical audit](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/main/TECHNICAL_ACCURACY_AUDIT.md) documents the corrections.

## What this book covers

- **Part 0 -- Why Compile Tensor Programs At All.** The real, measurable cost of eager execution, what a general compiler pipeline actually looks like end to end, and a tour of how three real production ML compilers -- XLA, TVM, and Triton -- are actually built.
- **Part 1 -- CUDA Hammer's IR: A Minimal Tensor Graph From Scratch.** Representing tensors and operations as a graph, a small frontend for building that graph, shape inference and validation, and tooling to print and visualize CUDA Hammer's own IR while developing it.
- **Part 2 -- Basic Optimization Passes.** A real pass manager for structuring transformations, constant folding and dead code elimination, common subexpression elimination over a tensor graph, and algebraic simplification.
- **Part 3 -- Operator Fusion.** Why fusion matters in the first place -- the real arithmetic-intensity argument behind memory-bound versus compute-bound kernels -- elementwise fusion, fusing reductions, loop fusion and tiling for the fused result, and precisely which fusion boundaries a real compiler cannot cross, and why.
- **Part 4 -- Code Generation.** Lowering CUDA Hammer's high-level IR to an explicit loop nest, generating real CUDA C++ from the fused IR, generating real vectorized CPU code, and a JIT backend that compiles and loads generated code at runtime.
- **Part 5 -- Autotuning.** The real search space a kernel's own schedule lives in (tile sizes, loop orders, unrolling), cost models versus genuine on-hardware measurement, building CUDA Hammer's own autotuner, and caching tuned schedules for reuse.
- **Part 6 -- Case Studies.** How CUDA Hammer's own design choices compare to XLA's HLO fusion passes, TVM's Relax/TIR stack and MetaSchedule, Triton's block-level programming model, torch.compile and TorchInductor, Flash Attention read as a fusion case study, and quantization-aware code generation.

- **Part 7 -- Fixed domain applications.** Preprocessing, a small bag-of-embeddings classifier, a heat stencil and binomial option pricing demonstrate the supported graph operations. These are bounded teaching applications, not complete production systems.

## How to read this book

Chapters combine intuition, implementations, worked calculations, common traps and self-check questions. The case studies and domain applications use somewhat different structures. Host examples are compiled and run, and useful values and graph states appear in the sample outputs. GPU compilation, CPU-reference checks and actual kernel execution are documented separately; none substitutes for the others. See [Validation Status](validation-status.md) for the tested cases, toolchains and remaining limits.
