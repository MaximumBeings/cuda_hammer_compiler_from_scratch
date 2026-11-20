# CUDA Hammer technical accuracy audit and correction record

Reviewed on November 20, 2026 against source snapshot `839638b`. The review covers all 34 chapters, six appendices, standalone lesson sources, embedded listings, sample-output structure, site configuration and diagrams. Three reviewers examined foundations, systems/code generation, and external compiler/application case studies; integration added cross-book structural checks and full host execution.

This is an implementation review with focused regressions, not a universal proof, production compiler certification or complete performance benchmark campaign. A corrected teaching scope does not implement the production capability that the original text overstated.

The machine-readable [findings ledger](validation/findings.json) records 25 grouped review items; several cover multiple copied implementations.

## Second independent review and correction pass

The independent review of published `69a3b19` found 12 further defects, including a real AddressSanitizer failure in the introductory constant-folder and silent generated-literal/JIT failures. These were missed by the original ordinary-exit-code tests. All 12 are now addressed in [the second-pass correction record](SECOND_PASS_CORRECTIONS.md), including a fresh 78-driver host sanitizer sweep and targeted failure-mode regressions. Earlier Linux/A100 results below remain historical; the updated CUDA generators were not freshly executed on a GPU.

## Original detailed findings and corrections

- [Foundations: Chapters 1–16](validation/accuracy-fixes/foundations.md): floating-point rewrite policy, finite FP32 parser/printer round trips, shape/product validation, overflow-safe tiling, timing provenance, roofline interpretation and fusion legality.
- [Systems: Chapters 17–24 and toolchain appendices](validation/accuracy-fixes/systems.md): FMA semantics, nonfinite comparisons, genuine CUDA execution checks, checked child processes, architecture selection, JIT/linkage claims, tuning-cache scope, profiling terminology and source-listing consistency.
- [Compiler case studies and applications](validation/accuracy-fixes/case-studies.md): masked attention tails, affine calibration and accumulator width, framework/API claims, scalar capture, meaningful numerical checks and external-versus-generated kernel conclusions.

These reports distinguish implementation fixes from narrowed contracts and list primary sources. Their earlier statements that GPU validation was pending are superseded only by the actual device-run record below, not by successful host compilation.

## Publication and reproducibility changes

The homepage no longer claims that every tensor framework and hand-written CUDA kernel executes one operation at a time or synchronizes after each launch. The Part 6 roadmap now names the Relax/TIR/MetaSchedule material actually covered, and Part 7 is included.

Added 159 responsive SVGs: a whole-book map, 34 chapter openers, 102 chapter concept diagrams, six appendix openers and 16 numbered appendix concept diagrams. They use the shared deep-blue style and a 760-unit canvas, accessible labels and full-size links. Diagrams describe control/data flow and boundaries; their cards and arrows are not hardware measurements.

`tools/verify_book.py` checks numbered-section uniqueness, local Markdown links, source-listing correspondence, nontrivial sample outputs and SVG structure/built assets. All 108 downloadable source listings are checked against embedded code. Historical diagnostic/IR transcripts are kept distinct from fresh execution; a retained historical excerpt is not relabeled as a new toolchain result.

`tools/validate_host.py` compiles every standalone C++ lesson in an isolated temporary directory, executes it and records stdout/stderr, return codes, host/compiler metadata and duration. The script returns failure if any lesson fails. The Markdown illustrates actual calculations, graph states, shapes and before/after results rather than only pass counts.

## Current validation evidence

- [Native host execution](validation/host/results.json): all 78 host C++ lessons compile and execute on macOS/Apple Clang/arm64.
- [Focused foundations regressions](validation/accuracy-fixes/foundations-regressions.txt): five sanitizer suites covering arithmetic policy, serialization, shape/product checks and tiling boundaries.
- [Code-generation numeric regressions](validation/accuracy-fixes/codegen-numerics.txt): FMA rounding counterexample and rejection of NaN in finite-fixture comparisons.
- NumPy attention and affine quantization checks, Python syntax checks, LLVM execution and individual reviewer logs are under [accuracy-fix validation](validation/accuracy-fixes).
- [GPU runner](tools/validate_gpu.py) targets the actual NVIDIA architecture, checks generated program results against the interpreter, rejects no-device skips, and runs Compute Sanitizer across child processes.
- [Triton execution runner](validation/accuracy-fixes/gpu_case_validation.py) covers incomplete KV tiles and int8 accumulation against independent references. A missing device is a failure, not a pass.

All 78 actual host lessons also pass under GCC/Linux; the original archive contained non-source AppleDouble entries, whose diagnostic failures are preserved separately in the raw manifest. The validator now excludes dotfiles. Three CUDA drivers/four generated cases pass, all 12 direct memcheck/racecheck/synccheck runs are clean, and all 12 Triton numerical cases pass both normally and under memcheck. Ten framework CPU probes and two fresh AOT examples also pass on the recorded installed versions.

The GPU/GCC evidence and external-package compatibility limits are summarized on the [public validation page](https://maximumbeings.github.io/cuda_hammer_compiler_from_scratch/validation-status/). Device details and executed commands are retained with the results. No GPU speedup is inferred from a host byte-count model.

## Remaining boundaries

The IR, cache formats and parsers are educational, trusted-input implementations. Strict algebraic rewriting is the default; relaxed arithmetic must be requested explicitly, and SIMD FMA/reassociation has a separate numerical contract. Equal loop extents alone do not establish dependence-safe fusion.

Attention examples have explicit query/tile/dimension contracts; they are not a complete batched, causal, arbitrary-tail transformer backend. The public C++ JIT interface assumes compatible C++ ABI details. Full production cache keys, dependence analysis, arbitrary malformed-input hardening and exhaustive IEEE equivalence are not claimed.

External frameworks evolve. The book's historical framework captures target their stated versions; currently installed versions and fresh execution coverage must be read from the logs. CPU compiler observations do not prove a GPU backend's universal capabilities, and compiler acceptance does not prove kernel execution correctness.
