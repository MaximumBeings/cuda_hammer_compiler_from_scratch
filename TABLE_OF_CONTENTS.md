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
Chapter 1 (Part 0, DONE, commit `b1f1933`): "The Cost of Eager Execution" -- opens the book. Two real, independent costs of eager execution quantified without any fabricated timing: 1.1/1.3 a genuinely counted memory-traffic model (eager(N) = N * 2 * array_bytes vs. fused = 2 * array_bytes, checked for N=1..8, EAGER and FUSED outputs verified bit-identical); 1.2 NVIDIA's own real cited kernel-launch-overhead figure (9.6us per kernel, "Getting Started with CUDA Graphs" blog post) applied via a real formula to the same chain lengths, plus a closing synthesis on a concrete 4-op/1M-element example that deliberately reports the two reductions SEPARATELY rather than fabricating a combined wall-clock speedup. 2 files (both plain C++, no CUDA/NCCL/MPI linkage -- cross-verified on both cloud sandbox and device), 4 ASCII diagrams (chapter-opening + one per section), zero diagram-rule violations on first check.
[stated] Planning note (2026-09-22, not yet placed in the chapter/appendix plan): user asked to integrate NVIDIA CUTLASS (the real production GEMM/tensor-op template library) at some point. Natural homes to decide between when we get there: a new Part 6 case-study chapter ("How CUTLASS Generates Production GEMM Code"), or folding it into Appendix C alongside LLVM/MLIR as a second "infrastructure you get for free" example. Not yet assigned a chapter number -- revisit before Part 4 (codegen) or Part 6 (case studies) so numbering doesn't need to be redone later.
[stated] Theme palette changed 2026-09-22: yellow/amber (was deep purple/purple).
Chapter 2 (Part 0, DONE, commit `c39c0bd`): "What a Compiler Pipeline Looks Like" -- completes Part 0's own build-up before Part 1 starts CUDA Hammer itself. Builds general compiler vocabulary (lex, parse, AST, lower, linear IR, pass, codegen) on a genuinely implemented tiny arithmetic-expression compiler, not tensors yet. 2.1/2.2 (File 003): real lexer + recursive-descent parser + AST + AST-to-three-address-code lowering + IR interpreter, cross-checked against an independent direct-AST evaluator across 5 expressions (precedence and left-associativity both genuinely exercised, e.g. `a - b - c`). 2.3 (File 004, reuses File 003's front end unchanged): a real constant-folding + constant-propagation pass operating purely on the IR, genuinely eliminating up to 100% of an expression's instructions (one test folds to zero instructions), correctness checked by interpreting before and after folding on every expression (all matched), 66.7% total instruction reduction across the 5 test expressions. 2 files, both plain C++ (no CUDA linkage), zero diagram-rule violations across 4 diagrams on first check. No external sources cited -- general compiler-construction vocabulary from first principles.
Chapter 3 (Part 0, DONE, commit `935b6a6`): "A Tour of Real ML Compilers: XLA, TVM, and Triton" -- closes Part 0. A citation-heavy survey chapter, not a code chapter (its own `code/` directory is intentionally empty, per Getting Started's own honesty discipline: claims about other projects are cited to their own real documentation, not measured by this book). Maps Chapter 2's vocabulary (IR, pass, lowering, codegen) onto three real production ML compilers: 3.1 XLA's HLO IR and its two-phase (target-independent then backend-specific) fusion passes, cited to openxla.org's own architecture/hlo_passes/gpu_architecture pages, including the direct quote "Fusion is XLA's single most important optimization"; 3.2 TVM's TensorIR and its explicit schedule primitives (`split`/`tile`/`reorder`/`bind`/`fuse`), cited to tvm.apache.org's own TensorIR and schedule-primitives docs, plus the DLight (rule-based) vs. MetaSchedule (search-based) distinction that foreshadows CUDA Hammer's own Part 5; 3.3 Triton's block-level "Blocked Program, Scalar Threads" model (inverted from CUDA's own "Scalar Program, Blocked Threads"), cited to triton-lang.org's own programming-guide introduction. 6 real sources cited, all fetched and quoted directly from each project's own current documentation this session. 4 ASCII diagrams, one stray `<-` caught and fixed by the automated diagram check before locking (replaced with `--`), zero violations after the fix.
[stated] Header bar text/icons changed to red 2026-09-23 (commit `9989464`), via `docs/stylesheets/extra.css` (`#c1121f`, wired in through `mkdocs.yml`'s `extra_css`) -- scoped to the header bar only, per the user's own explicit scope choice.
Chapter 4 (Part 1, DONE, commit pending): "Representing Tensors and Operations as a Graph" -- opens Part 1, CUDA Hammer's first real code chapter (no more parsing arithmetic text -- direct graph-construction API calls from here forward). 4.1/4.2 (File 005): real `Value`/`Node`/`Graph` classes; a genuinely shared 6-node diamond DAG built only through `addInput`/`addBinary`/`addUnary` (never parsed from text), with a computed non-leaf node (`t1`) proven to have exactly 2 real consumers, plus a deliberately unmemoized recursive `expandToTreeNodeCount()` genuinely computing that an equivalent unshared tree would need 10 nodes instead of 6 -- a real counted IR-size cost of not sharing, echoing Chapter 1's own memory-traffic argument. 4.3 (File 006, reuses File 005's classes unchanged): a real topological sort (Kahn's algorithm), cross-checked by an independent positional validator, confirmed valid on the normal DAG; the same checker correctly detects a cycle in a second graph deliberately hand-broken by reaching past the public API (`mutableNode()`) to mutate an existing node's inputs after construction -- proving the normal API's cycle-freedom-by-construction guarantee by breaking it on purpose. 2 files, both plain C++ (no CUDA linkage) -- cross-verified on cloud sandbox and device. 6 ASCII diagrams total (4 hand-drawn + 2 locked verbatim program-output blocks, the latter correctly exempt from the diagram-character rule since they contain real printf output using `<-`), zero violations on first check. No external sources cited -- CUDA Hammer's own graph design is original to this book.
NEXT: Chapter 5, "Building the Graph: CUDA Hammer's Frontend" (Part 1 continues).
