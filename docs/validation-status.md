# Validation Status

The November 20, 2026 audit corrected code defects and narrowed claims that exceeded the teaching implementations. [The correction record](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/main/TECHNICAL_ACCURACY_AUDIT.md) links the findings, primary references and reproducible checks.

## Host and documentation checks

- All **78 standalone host C++ lessons** compile and execute on native macOS/Apple Clang/arm64. This includes generated CPU child programs; it does not execute CUDA kernels.
- All **five focused foundations sanitizer suites** pass, including floating-point rewrite policy, parser/printer precision, shapes, integer overflow and tiling boundaries.
- Numerical comparison regressions reject NaN and distinguish FMA rounding from separate multiply/add.
- NumPy attention and quantization checks pass. LLVM examples compile and execute. TVM, MLIR/Polly, CUTLASS and optional profiling-tool examples have separately stated dependency and version limits; this audit does not claim fresh execution of those unavailable stacks.
- All **109 source listings** match their embedded code. Local links, numbered sections and sample-output structure are checked. All **159 SVG diagrams** have accessible labels, share the deep-blue 760-unit canvas and resolve in the strict MkDocs build.

Fresh native-host logs and framework/device records are available in the repository's [validation directory](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation). Sample stdout may differ in machine timings, random-library details and compiler IR across platforms; byte-identical timing is not a requirement or a meaningful promise.

## GPU and Linux validation

The connected instance was an **NVIDIA A100-PCIE-40GB**, with driver 580.105.08 and CUDA toolkit 12.8. These are actual executions, not offline-compile or CPU-reference substitutes.

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

Actual commands, device metadata, numerical output, sanitizer summaries and source hashes are retained in [GPU code-generation logs](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation/accuracy-fixes/gpu), [A100 case-study logs](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation/accuracy-fixes/a100-case-studies), and [Linux host records](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/tree/main/validation/linux-host). These results supersede the original no-device diagnostics for the exercised cases.

## What passing checks do not establish

This is a small compiler built for instruction. Passing examples does not establish a hardened parser, a complete IR verifier, full IEEE-preserving optimization, arbitrary dependence-safe loop fusion, production tuning-cache identity, framework-wide compatibility or a measured GPU performance advantage.

Historical XLA/TVM/Triton/PyTorch, profiling and no-device transcripts remain historical where not freshly rerun. A CPU reference supplies expected values; it cannot substitute for running the generated kernel. Any unavailable or skipped validation is explicitly distinguished from a pass.
