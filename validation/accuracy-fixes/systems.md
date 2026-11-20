# Systems accuracy fixes: Chapters17–24 and AppendicesA/C/D/E/F

## Confirmed issues resolved

1. **FMA/data-flow legality is not bitwise numerical equivalence.** Chapter19 now explicitly permits contraction and vector-reduction reassociation, distinguishes the single-consumer invariant from floating-point rounding, and contains an actual counterexample: separate multiply/add0 versus FMA-2^-46. Source046 runs that regression. Reused FMA generators047/050 disclose the same numerical policy. Fixed the false claim that the NEON horizontal-sum intrinsic necessarily lowers to one instruction.
2. **NaNs passed approximate comparisons.** Changed finite-fixture comparisons in040/042/045–050 to require absolute error<=tolerance, thereby rejecting NaNs. tests/codegen_numerics.cpp verifies matching numbers pass while NaNs and mismatches fail. This is a finite-fixture audit contract, not a generic IEEE-special-value equivalence implementation.
3. **CUDA drivers were designed to pass only without a GPU.** Chapter18 examples042–044 now compare real device output against the interpreter. No-device is an explicit incomplete validation(skip77), never a numeric pass. Generated CUDA_CHECK stops at the first failure rather than continuing with uninitialized pointers. Shell failures are checked before executing generated artifacts. Host-only generator compilation passes; generated CUDA compilation/runtime remains pending a real CUDA host.
4. **Hardcoded runtime compilation architecture.** Added HAMMER_CUDA_ARCH override to042–044 so both outer and runtime-generated child kernels can target the same GPU. tools/validate_gpu.py --arch sm_86 --sanitize targets A10, captures device/compiler/runtime logs, rejects skips, and runs memcheck/racecheck/synccheck across child processes. NVIDIA lists A10 as compute capability8.6. Historical defaultsm70 remains configurable; current toolkits may not support every historical target.
5. **Language linkage/JIT claims.** Chapter20 now states C language linkage applies to function types as well as names; STL-bearing signatures still require a compatible C++ ABI. In-process JIT is not synonymous with self-hosting, and libclang's parsing/analysis API is not an ORC replacement for JIT execution.
6. **Architecture-only tuning-cache key was overstated.** Chapter24 now calls it a teaching key, not complete machine identity; production keys need CPU/features/compiler flags/generator revision and workload numerical/layout contracts. Preserved the actual cache demonstration, removed duplicated misplaced24.2/.3 prose before24.1 code. Trusted/local text-file scope is explicit.
7. **Linux feature file absent was called hardware mismatch.** AppendixA094/096 distinguishes /proc/cpuinfo unavailable from features absent on native macOS. Corrected advice suggesting an SSE backend existed; Chapter19's shape fallback is not runtime CPU dispatch. Added explicit AVX2/FMA prerequisite.
8. **Memory/cache and profiling overclaims.** Removed the assertion that64MB cannot fit any real cache. AppendixE distinguishes instrumented/simulated counts from physical PMU counters and qualifies determinism and causal claims. AppendixD states AXPY's independence requires dependence-safe aliasing; C array parameters do not supply no-alias guarantees.
9. **LLVM IR description.** Removed claim that compare/select is the only way to encode ReLU; branches/phi and suitable intrinsics are alternatives. Embedded/downloadable sources synchronized.

## Actual validation and sample results

- All28 assigned plain C++ examples compiled with Apple Clang C++17/O2/Wall/Wextra and executed successfully on native macOS arm64. Per-source stdout/stderr and exit status are retained alongside this note; timings are local observations, not cross-machine performance guarantees. The11 materially edited host examples were rerun successfully.
- New numeric comparison regression passes; FMA regression prints separate=0x0p+0 and FMA=-0x1p-46. New readable sample included in Chapter19. AppendixA includes actual unavailable-feature-file output.
- Chapter18 outer generators042–044 compile as ordinary host C++ without requiring CUDA headers; this validates generator syntax only, not generated CUDA kernels. Host compile logs are distinct from GPU logs.
- LLVM097 and098 compiled through Apple clang and each executed with result28.000000. Stencil101 executed; AXPY102 compiled. This does not rerun opt/Polly/MLIR transformations.
- Assigned Python105/107 syntax checks pass; TVM and PyTorch unavailable. mlir-opt/opt/Valgrind/perf/nvcc/ComputeSanitizer unavailable. No fresh profiling, TVM, MLIR, Polly or GPU result claimed. Historical transcripts were preserved as historical examples rather than relabeled as newly measured output.
- tools/verify_book.py checks source listings/section structure/links/sample-output structure. See book-systems-check.txt.

## Primary references

- C++ working draft language linkage: https://eel.is/c++draft/dcl.link
- LLVM ORC design: https://llvm.org/docs/ORCv2.html
- Clang libclang scope: https://clang.llvm.org/docs/LibClang.html
- LLVM IR reference: https://llvm.org/docs/LangRef.html
- NVIDIA floating-point guidance: https://docs.nvidia.com/cuda/floating-point/index.html
- NVIDIA GPU capabilities(A10=8.6): https://developer.nvidia.com/cuda/gpus
- Cachegrind manual: https://valgrind.org/docs/manual/cg-manual.html (manual fetch returned503 during audit; no new numeric claims inferred from the inaccessible page)

No commits made. No instance was provisioned by this agent; the later user-provided A100 validation is recorded below.

## GPU follow-up completed

A real NVIDIA A100-PCIE-40GB became available after the initial local audit. CUDA12.8 and ComputeSanitizer2025.1 are installed; targetsm80, not A10/sm86. All four generated-program numeric comparisons pass on the A100 (042 vector,043 atomic reduction,044 uncapped and capped graphs). All three drivers exited0; all12 direct memcheck/racecheck/synccheck runs exited0 and reported zero errors or hazards. Text logs and metadata were downloaded to `validation/accuracy-fixes/gpu`; final runner-summary.txt reports PASS. The original local-unavailable statements above describe that earlier platform, not this subsequent Linux GPU run.

The first wrapper-only sanitizer attempt did not instrument child GPU calls and was rejected, not counted as a pass; logs are retained in gpu/superseded-wrapper-attempt. The runner now directly instruments all four generated CUDA executables and requires an explicit zero-error summary. AppleDouble metadata files from the macOS transfer caused a separate false aggregate compile failure; source discovery now excludes dotfiles. Genuine CUDA sources compiled and executed correctly.

These are small-fixture correctness and sanitizer checks, not performance benchmarks or a proof for every input shape. No A10 execution is claimed. No GPU work remains for this agent and all required remote logs have been saved locally.
