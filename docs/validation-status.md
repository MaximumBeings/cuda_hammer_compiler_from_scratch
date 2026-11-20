# Validation Status

The November 20, 2026 independent re-audit found 12 further defects after the first publication, including a constant-folding heap overflow and incorrect generated constants. All 12 have been corrected. [The second-pass correction record](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/main/SECOND_PASS_CORRECTIONS.md) links implementation changes, actual results and remaining limits. Passing ordinary exit-code checks was insufficient; current checks include sanitizer and deliberate-failure regressions.

## Current local correction-pass checks

- **78/78 standalone host C++ drivers pass** native macOS/Apple Clang builds with AddressSanitizer, UndefinedBehaviorSanitizer and float-cast-overflow instrumentation. Generated child programs are exercised but are not automatically sanitizer-instrumented by that driver sweep.
- **38/38 foundations examples** also pass an independent sanitized run, including the formerly failing complete constant-folding example. **21 focused foundations regression suites** pass.
- Four targeted systems regressions also pass with ASan/UBSan. JIT regressions reject diagnostic and silent compiler failures with an existing library, preserve the prior artifact, and promote only a successful staged build. Successful compiler warnings remain diagnostics rather than false failures.
- Float-literal and comparison-output tests preserve nine finite bit patterns, including subnormal/negative zero. The formerly incorrect generated calculation now returns **10 for `1e8 * 1e-7`**. CUDA-source text is checked locally; device code was not freshly compiled or executed in this pass.
- Fifteen local NumPy/AST acceptance/calibration checks pass; application safety and actual-body-budget regressions pass across standalone copies. Adapters test failure wiring, not unavailable framework compiler internals.
- **17 negative coverage cases pass**: host, GPU, foundations and publication validators reject missing/empty manifests, absent inputs and incomplete expected coverage.
- All **108 actual source files** match their embedded listings. All **159 SVG diagrams** retain the shared deep-blue 760-unit canvas and accessible labels. Strict MkDocs and local publication checks pass. The previous displayed count of 109 included a cache directory, not an additional source file.

Current evidence is under [second-pass fixes](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation/second-pass-fixes). Native sample stdout was refreshed where code changed. Timings, random-library details and compiler IR need not be byte-identical across platforms.

## Earlier Linux/A100 baseline — not rerun for the changed CUDA generators

The earlier `69a3b19` validation used an **NVIDIA A100-PCIE-40GB**, driver 580.105.08 and CUDA toolkit 12.8. The runs below were actual executions, but they preceded the latest literal, JIT and fusion-budget corrections. **The GPU stayed off during the correction pass; the modified CUDA generators have no fresh device run.** The unchanged Triton kernels retain their matching source-hash evidence for the stated cases.

| Check | Result | Scope |
| --- | --- | --- |
| GCC/Linux host lessons | **78/78 passed** | All actual standalone C++ sources; macOS archive metadata is excluded as non-source. |
| Foundations regressions on Linux | **5/5 passed** | GNU g++ 13.3 with undefined-behavior and float-cast sanitizers. |
| Generated CUDA programs | **3 drivers, 4 numerical cases passed** | Elementwise, reduction, and capped/uncapped full-graph variants compared with their interpreter. |
| Direct CUDA diagnostics | **12 clean runs** | Memcheck, racecheck and synccheck on each of the four generated executables. Earlier uninstrumented wrapper attempts are explicitly superseded. |
| Triton attention and int8 dot | **12 numerical checks passed** | Six attention cases including incomplete KV tiles; six random/extreme int8 cases against independent references. |
| Triton memory check | **0 errors; exit 0** | All 12 cases rerun under Compute Sanitizer memcheck. This is not a claim of separate Triton racecheck/synccheck runs. |
| Framework CPU probes | **10 passed** | JAX/PyTorch/NumPy examples on the installed versions. TVM is not installed. |
| Fresh Triton AOT examples | **2 passed** | Attention and quantized-dot compiler dumps; separate from their numerical GPU tests. |

The attention comparison's largest observed absolute error was approximately **2.38e-7** against a float64 reference; its stated test tolerance was 3e-5. No device timing or speedup is inferred from these correctness tests.

The instance had Python 3.12.3, PyTorch 2.7.0, Triton 3.3.0, NumPy 1.26.4 and JAX 0.6.0. Historical book transcripts may target different versions. Optional introspection was guarded where the installed PyTorch lacked a newer API; no claim of an exhaustive framework-version compatibility sweep is made.

Actual commands, device metadata, numerical output, sanitizer summaries and source hashes are retained in [GPU code-generation logs](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation/accuracy-fixes/gpu), [A100 case-study logs](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation/accuracy-fixes/a100-case-studies), and [Linux host records](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation/linux-host). These historical results supersede the original no-device diagnostics for their exercised revision and cases; they do not validate later CUDA-generator changes.

## What passing checks do not establish

This is a small compiler built for instruction. Passing examples does not establish a hardened parser, a complete IR verifier, full IEEE-preserving optimization, arbitrary dependence-safe loop fusion, production tuning-cache identity, framework-wide compatibility or a measured GPU performance advantage.

Historical XLA/TVM/Triton/PyTorch, profiling and no-device transcripts remain historical where not freshly rerun. A CPU reference supplies expected values; it cannot substitute for running the generated kernel. Any unavailable or skipped validation is explicitly distinguished from a pass.
