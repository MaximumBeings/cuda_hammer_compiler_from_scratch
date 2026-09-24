# Appendix B: Practice Quiz

Every question below is grounded in a real, specific fact this book already established somewhere in Chapters 1-34 or Appendix A -- a real measured number, a real quoted source, a real compiled/run result, or a real design decision this book's own code demonstrated rather than merely stated. None of these answers is a paraphrase of "what the chapter was about"; each one names the exact real fact, and the exact chapter and file where it was established, so this appendix can double as a fast way to relocate something you remember reading but can't quite place.

Click a question to reveal its answer.

## Part 0 -- Why Compile Tensor Programs At All (Chapters 1-3)

??? question "Q1. Chapter 1 quantified two real, independent costs of eager execution. What was the real memory-traffic formula, and what real NVIDIA-cited figure did the second cost use?"
    The memory-traffic model: `eager(N) = N * 2 * array_bytes` versus `fused = 2 * array_bytes`, checked for N=1..8 with the EAGER and FUSED outputs verified bit-identical. The second real cost used NVIDIA's own cited kernel-launch-overhead figure -- 9.6us per kernel, from NVIDIA's "Getting Started with CUDA Graphs" blog post -- applied via a real formula to the same chain lengths.

    *(Chapter 1, "The Cost of Eager Execution")*

??? question "Q2. Chapter 2 built a real constant-folding pass on a tiny arithmetic-expression IR. What real reduction did it achieve across its 5 test expressions, and what was the maximum reduction on any single one?"
    A 66.7% total instruction reduction across the 5 test expressions, with one expression folding to 100% -- zero instructions left at all. Correctness was checked by interpreting before and after folding on every expression, all matching.

    *(Chapter 2, File 004)*

??? question "Q3. What real, directly quoted claim did Chapter 3 cite from XLA's own documentation about fusion, and which two real strategies did it contrast for TVM's own scheduling?"
    "Fusion is XLA's single most important optimization," quoted directly from openxla.org. For TVM, it contrasted DLight (rule-based scheduling) against MetaSchedule (search-based scheduling) -- foreshadowing this book's own Part 5.

    *(Chapter 3, "A Tour of Real ML Compilers: XLA, TVM, and Triton")*

??? question "Q4. Chapter 3 described Triton's own programming model as 'inverted' relative to CUDA's. Inverted how?"
    Triton's own real model is "Blocked Program, Scalar Threads" -- the inverse of CUDA's own "Scalar Program, Blocked Threads" -- cited directly to triton-lang.org's own programming-guide introduction.

    *(Chapter 3, Section 3.3)*

## Part 1 -- CUDA Hammer's IR: A Minimal Tensor Graph From Scratch (Chapters 4-7)

??? question "Q1. Chapter 4's diamond graph shares one computed node (`t1`) between two consumers. How many nodes does the real shared graph use, versus an equivalent unshared tree, and what function computed the unshared figure?"
    6 nodes shared, versus 10 nodes for an equivalent unshared tree -- computed by a deliberately unmemoized recursive function, `expandToTreeNodeCount()`, echoing Chapter 1's own memory-traffic argument as a real, counted IR-size cost of not sharing.

    *(Chapter 4, File 005)*

??? question "Q2. What real algorithm does Chapter 4 use for topological sort, and how was a deliberately broken graph's cycle actually detected?"
    Kahn's algorithm, cross-checked by an independent positional validator. A cycle was deliberately introduced by reaching past the public API via `mutableNode()` to mutate an existing node's inputs after construction -- and the same checker correctly detected it, proving the normal API's cycle-freedom-by-construction guarantee by breaking it on purpose.

    *(Chapter 4, File 006)*

??? question "Q3. What is CUDA Hammer's own real frontend grammar (Chapter 5), and what was the strong correctness check used on the parser?"
    A minimal, flat text grammar -- `"name = op(arg, ...)"`, every right-hand side a call, no exceptions -- parsed straight into a Graph with no AST stage, the same real architecture choice LLVM's own IR text format uses. The strong check: parsing the Chapter 4 diamond program produced a graph that `graphsStructurallyEqual()` confirmed was node-for-node, edge-for-edge IDENTICAL to the graph Chapter 4 built by hand.

    *(Chapter 5, File 007)*

??? question "Q4. Where does Shape information live in CUDA Hammer's own IR, and what real external source did Chapter 6 cite for its broadcasting rule?"
    Shape lives as a separate, computed side table (`map<nodeId, Shape>`) -- never a field on `Node` itself, a deliberate pass/IR separation, not an oversight. The broadcasting rule is the real NumPy broadcasting rule, quoted directly from numpy.org's own documentation.

    *(Chapter 6, File 009)*

## Part 2 -- Basic Optimization Passes (Chapters 8-11)

??? question "Q1. What is `TransformPass`'s real type signature in Chapter 8, and what real design gap in `Graph` motivated that signature?"
    `TransformPass = std::function<Graph(const Graph&)>` -- a real design consequence of `Graph` having no node-removal API at all, stated explicitly as groundwork for Chapter 9's upcoming dead-code elimination.

    *(Chapter 8, File 013)*

??? question "Q2. In Chapter 9's own capstone (File 017), what real node-count sequence results from running constant folding then dead-code elimination on a 7-node graph, and what does running DCE ALONE on the unfolded graph prove?"
    7 (original) -> 7 (after folding -- unchanged, since folding only replaces computations, never removes nodes) -> 3 (after DCE). Running DCE alone on the unfolded original graph removes nothing (7->7) -- a real, executed proof that order matters, not an assertion.

    *(Chapter 9, File 017)*

??? question "Q3. What real, honestly conservative limitation does Chapter 10's `cseKey()` have with respect to commutativity?"
    `add(a,b)` and `add(b,a)` are mathematically identical but get DIFFERENT keys, since operand order is encoded and commutativity is never checked -- deliberately NOT merged. Staying conservative here avoids the much worse alternative: silently merging operands for an operation that ISN'T actually commutative, a real correctness bug.

    *(Chapter 10, File 019)*

??? question "Q4. What real hazard does Chapter 11's own Part C demonstrate between `algebraicSimplificationPass()` and Chapter 9's 'the last node is the output' convention?"
    When `algebraicSimplificationPass()` elides the literal LAST node a graph ever added, Chapter 9's own convention (which `deadCodeEliminationPass()` depends on directly) silently ends up pointing at the WRONG node afterward -- proven with real, minimal, executable code, framed as an open question carried forward with no fix offered, not silently patched.

    *(Chapter 11, File 023)*

## Part 3 -- Operator Fusion (Chapters 12-16)

??? question "Q1. What is Chapter 12's real closed-form arithmetic-intensity formula for an UNFUSED chain of K unary elementwise ops, and why is it independent of K?"
    `AI_unfused = 1 / (2 * bytesPerElement)` -- a constant independent of K, because both memory traffic and compute work scale by K in exact lockstep, so K cancels completely out of the ratio.

    *(Chapter 12, File 024)*

??? question "Q2. What real ridge point did Chapter 12 compute from NVIDIA's own cited A100 datasheet, and at exactly what K did a FUSED chain cross from memory-bound to compute-bound?"
    12.5402 FLOPs/byte, computed from the A100's real 19.5 TFLOPS FP32 and 1,555 GB/s bandwidth figures. The fused chain crosses into compute-bound at exactly K=101 (K=100 is confirmed not yet compute-bound).

    *(Chapter 12, File 025)*

??? question "Q3. What real new IR node kind did Chapter 13 introduce -- the first new kind since Chapter 4 -- and what real rule decides whether a value gets inlined into it?"
    `OpKind::FusedElementwise`. A single-consumer Add/Mul/ReLU gets inlined as an internal `FusedStep`; anything with MORE than one consumer stays external -- proven directly on Chapter 4's own diamond graph, where `t1` (2 consumers) stays materialized as its own node while the single-consumer chain around it fuses into one.

    *(Chapter 13, Files 027-028)*

??? question "Q4. What real new fusion rule did Chapter 14 add specifically for `Sum` nodes, and what real gap did it fix in Chapter 12's own `totalFlops()`?"
    A `Sum` node's own output is ALWAYS kept external, regardless of consumer count -- even with exactly one consumer, unlike every other op. It also fixed a real gap in `totalFlops()`: an N-element `Sum` needs N-1 real additions, not N, since Chapter 12's own convention had assumed output-element-count always equals operation-count (true for Add/Mul/ReLU, false for Sum).

    *(Chapter 14, Files 030 and 032)*

## Part 4 -- Code Generation (Chapters 17-20)

??? question "Q1. What real, hand-derived per-element results did Chapter 17's `evaluateArrays()` confirm at specific non-uniform positions in the diamond graph's own `out` node?"
    `out[0]=22`, `out[5]=182`, `out[11]=676` -- all hand-derived BEFORE running the code, then confirmed exactly, along with full per-element agreement across all 12 positions between the original and fused graphs.

    *(Chapter 17, File 039)*

??? question "Q2. Why does Chapter 18's CUDA reduction kernel need `atomicAdd()` instead of a plain `*out += step`, and is `atomicAdd()` the fastest real approach?"
    One thread per reduce element doing `*out += step` directly is a genuine data race -- a non-atomic read-modify-write on the same address with no ordering guarantee, a classic lost-update bug. `atomicAdd()` is stated explicitly as the CORRECT approach, not the fastest one: a real high-performance reduction would use shared-memory tree reduction or warp-shuffle instructions, named directly as an out-of-scope limitation.

    *(Chapter 18, File 043)*

??? question "Q3. What real argument-order difference between AVX2's and NEON's real FMA intrinsics does Chapter 19's `vecFma()` exist to hide?"
    AVX2's `_mm256_fmadd_ps(a,b,c)` computes `a*b+c` (accumulator last); NEON's `vfmaq_f32(c,a,b)` computes `c+a*b` (accumulator first) -- the same three real operands, genuinely different argument order between the two real ISAs, hidden behind CUDA Hammer's own single `vecFma()` call.

    *(Chapter 19, toolchain investigation and File 046)*

??? question "Q4. What real architecture fact did Chapter 20 confirm about the device via `uname -a`, and why did that make its JIT loader simpler than Chapter 19's own vectorized backend?"
    The device -- physically Apple Silicon hardware -- is confirmed a genuine aarch64 LINUX virtual machine (`Linux claude 6.8.0-138-generic ... aarch64 GNU/Linux`), not native macOS. Both machines produce the same real ELF 64-bit shared-object format, so `dlopen()`/`dlsym()`/`dlclose()`, `-shared -fPIC`, and `-ldl` need ZERO per-architecture special-casing -- a genuinely simpler mechanical story than Chapter 19's own real AVX2-vs-NEON intrinsic split.

    *(Chapter 20, File 048)*

## Part 5 -- Autotuning (Chapters 21-24)

??? question "Q1. What real IEEE-754 float32 non-associativity example did Chapter 21 measure, and what real discrepancy did it find between forward and reverse summation order?"
    Summing `16777216.0` (2^24, float32's own exactness limit) plus sixteen `1.0`'s: FORWARD order (big value first) silently absorbs all 16 increments and stays at `16777216.0`, while REVERSE order recovers the true `16777232.0` -- a real 16.0 discrepancy, directly explaining why every `evaluateArrays()`-vs-generated-code check since Chapter 17 uses a tolerance rather than exact equality.

    *(Chapter 21, File 052)*

??? question "Q2. How many real schedules did Chapter 21's `enumerateSchedules()` produce for the concrete `LoopNest[dim0:6, dim1:8]`, and how was that count derived?"
    128 real schedules: 4 tile candidates x 4 tile candidates x 2 loop orders x 4 unroll candidates -- confirmed to match `enumerateSchedules()`'s own actual output count exactly.

    *(Chapter 21, File 053)*

??? question "Q3. Chapter 22's loop-overhead cost model is monotonic by construction. What real, machine-specific turnover point did real measurement find that the model structurally cannot predict?"
    Real measurement found a genuine turnover point in chunk-size cost, at a DIFFERENT chunk size on each real machine: chunkSize=32 on the cloud sandbox's x86-64, chunkSize=8 on the device's aarch64 -- both real, honestly reported, machine-specific findings the monotonic model cannot see by construction.

    *(Chapter 22, File 054)*

??? question "Q4. In Chapter 23's real autotuner, what honest miss did BOTH machines share when checking the cost model's top-8 shortlist against real exhaustive measurement -- and did the hybrid approach still win on speed?"
    On both real machines, the true best schedule (found only by exhaustive real measurement) was NOT inside the cost model's own top-8 shortlist. Yet the hybrid approach (measuring only the top-8) still won on real wall-clock time by a real margin: 2.96-3.10x across 3 cloud runs, 4.90x on the device.

    *(Chapter 23, File 059)*

## Part 6 -- Case Studies: How CUDA Hammer Compares to Real Compilers (Chapters 25-30)

??? question "Q1. What real XLA fusion boundary did Chapter 25 find that is LOOSER than Chapter 13's own 'more than one consumer stays external' rule?"
    Real XLA fuses the WHOLE diamond graph -- including the shared `t1` node -- into ONE `kind=kLoop` fusion. Its real rule is "every consumer stays in the group," genuinely looser than Chapter 13's own rule that any node with more than one consumer must stay external.

    *(Chapter 25, "XLA: Fusion and Autotuning in a Real Production Compiler")*

??? question "Q2. What real, honest retitling did Chapter 26 have to make about TVM's own Python bindings, and why?"
    Chapter 3's and the prior TOC's own real binding names (`tvm.relay`, `tvm.auto_scheduler`, `tvm.autotvm`, `tvm.tir`) do NOT exist in the real, installed `apache-tvm==0.26.0` release used this session -- the real current bindings are `tvm.relax`, `tvm.s_tir` (plus `meta_schedule`), and `tvm.tirx` -- retitled honestly rather than left stated incorrectly.

    *(Chapter 26, "TVM's Real Modern Stack: Relax, TIR, and MetaSchedule")*

??? question "Q3. What real finding did Chapter 27 make about fusion in Triton that contrasts with every graph compiler studied earlier in Part 6?"
    Triton has no separate fusion PASS at all -- fusion is a PROGRAMMER'S language-level choice, the opposite shape from XLA's and Relax's automatic whole-graph fusion. Real compiled TTIR proved a hand-fused kernel uses 3 loads/1 store versus two hand-split kernels' 4 loads/2 stores -- 50% more real memory traffic from materializing the intermediate `t1`.

    *(Chapter 27, Files 069-070)*

??? question "Q4. Did Chapter 29 find that XLA, Relax, or Inductor could automatically DISCOVER Flash Attention's own online-softmax fusion boundary from ordinary primitives?"
    No -- none of the three real systems re-derives online-softmax from primitives; Flash Attention needed a genuinely new ALGORITHM (Milakov and Gimelshein's online-softmax, applied by Dao et al.'s Flash Attention), not a fusion boundary any of them could discover automatically. Triton (Chapter 27) is the one system where the programmer writes that kernel by hand -- exactly what File 076 did.

    *(Chapter 29, "Flash Attention as a Cross-System Fusion Case Study")*

## Part 7 -- CUDA Hammer in Practice: Real-World Domain Applications (Chapters 31-34)

??? question "Q1. What real staging technique did Section 31.2's 3x3 filter use, since CUDA Hammer's IR has no shift/gather op, and which real production library's own documentation was it cited to?"
    im2col-style pre-shifted inputs -- 9 real host-side neighbor shifts, zero-padded at the border, feeding an unrolled `Mul`/`Add` graph. Cited directly to NVIDIA's own real CUTLASS documentation ("constructs the convolution matrix explicitly via...im2col"), including its own stated real cost ("replicates each activation element by a factor equal the filter size").

    *(Section 31.2, File 082)*

??? question "Q2. Since CUDA Hammer's `Sum` has never had a per-row/batched reduction, what real workaround did Section 32.1 use for LayerNorm-style mean-centering, and what real boundary did it state directly rather than fake?"
    It reruns the existing two-pass Sum-then-Const calibration pattern ONCE PER TOKEN, in a host-side loop -- 4 independent small graphs, not one big one. It states directly that it computes `x - mean` only, never the variance/`sqrt`/divide half of real LayerNorm, since this IR has neither `sqrt` nor division, and nothing fakes a standard deviation with a hardcoded constant.

    *(Section 32.1, File 084)*

??? question "Q3. What real physical finding did Section 33.2's 5-step heat-stencil chain make about total heat conservation, and at exactly which step did real heat begin to leak?"
    Total heat stayed EXACTLY conserved through step 3 (sum=200.0000), then genuinely leaked starting at step 4 (sum=199.2188, matching a hand derivation exactly) as the disturbance reached the zero-padded boundary approximation -- a real physical consequence of that boundary technique, not an artifact to explain away.

    *(Section 33.2, File 088)*

??? question "Q4. What real structural finding did Section 34.2 make about representing a binomial option tree, in direct contrast to Sections 31.2's and 33.1's own staging techniques?"
    A tree's own parent-child relationship is IRREGULAR but completely, statically known at build time -- exactly what an ordinary CUDA Hammer `Value` reference already expresses, the SAME mechanism Chapter 4's own diamond graph used for one node feeding two consumers. Unlike Section 31.2/33.1, it needed NO im2col-style staging; unlike Section 33.2, it needed no host-orchestrated re-evaluation between levels -- the whole tree is one static graph, evaluated once.

    *(Section 34.2, File 091)*

## Appendix A -- Installation and Setup

??? question "Q1. What real minimum C++ standard did Appendix A.1's real 71-file sweep find CUDA Hammer's own source actually needs, and what two real constructs were responsible?"
    C++14 -- not C++17, despite `-std=c++17` being used in every compile command throughout this entire book. 58 of the book's 71 real `.cpp` files fail to compile under `-std=c++11`, splitting exactly into 54 files using `std::make_unique` and 4 files using digit-separator literals like `1'000'000`, both real C++14 features, with zero overlap and zero failures under `-std=c++14` or `-std=c++17`.

    *(Appendix A.1, File 093)*

??? question "Q2. What silent compile-time assumption has Chapter 19's own vectorized-CPU generator always made about ISA support, and how did Appendix A.2 check it directly instead of trusting it?"
    It assumes `__x86_64__` implies AVX2/FMA support and `__aarch64__` implies NEON support, using only the compile-time architecture macro with no direct hardware-flag check. Appendix A.2 checks this directly by reading each machine's own real, live `/proc/cpuinfo` "flags" line -- confirmed correct on both of this book's own machines.

    *(Appendix A.2, File 094)*

??? question "Q3. What real limitation does Appendix A.3's own CUDA toolkit smoke test carry, matching Chapter 18's own established discipline exactly?"
    It confirms `nvcc`'s real two-stage compile pipeline genuinely works, and honestly reports `cudaGetDeviceCount()=0` on a machine with no physical GPU -- never claiming a kernel's runtime result on hardware that isn't there, the same compile-but-don't-claim-a-run split Chapter 18 onward has carried since Chapter 18.

    *(Appendix A.3, File 095)*

??? question "Q4. What real, honestly-noted difference showed up in the `-std=c++11` compile-error TEXT between the cloud sandbox and the device in Appendix A.1, and what actually caused it?"
    The real compile-error text differs only in quote-mark style -- straight quotes from the cloud sandbox's g++13.3.0, curly quotes from the device's g++11.4.0 -- a real g++-version formatting difference, not a difference in root cause: both machines fail for the identical reason (`std::make_unique` and the digit-separator literal both need C++14).

    *(Appendix A.1 and A.4, Files 093 and 096)*
