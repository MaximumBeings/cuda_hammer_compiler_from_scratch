# Second-pass corrections and validation

November 20, 2026. All 12 findings from the independent review of `69a3b19` have been corrected. The original review and failure reproductions remain historical evidence; current acceptance tests exercise the repaired implementations. This is a correction record, not a formal certification of every possible graph or backend.

| Finding | Correction | Current evidence |
| --- | --- | --- |
| R2-01 | Temporary storage is keyed by actual SSA id, with checked lookup; folding can retain sparse ids safely. | Formerly failing complete 004 example and all 38 foundations examples pass ASan/UBSan. |
| R2-02 | Shared classic-locale hex-float emission preserves finite float bits; comparison stdout uses round-trip precision and explicit finite parsing. | Nine bit-pattern cases, subnormal/negative zero, locale variation, and generated `1e8 * 1e-7 = 10` pass. CUDA text is checked locally; no fresh device run. |
| R2-03 | Subprocess wait status is checked. Compilation uses unique staged source/output artifacts; failed builds throw before loading/timing/caching. Successful artifacts are promoted only after success. | Diagnostic and silent-exit-1 failures are rejected with a preexisting library; successful warnings are accepted; quoted paths and staged cleanup pass. |
| R2-04 | Fusion budgets count complete inlineable bodies, including the final operation, rather than longest-path depth; invalid budgets are rejected. | Branching trees obey budgets 1/2/3/4/8. The policy is propagated to later standalone copies. |
| R2-05 | Loop, logical-count and byte arithmetic uses checked integer additions/products; no overflowing double-to-integer byte conversion. | Overflow counterexamples throw under sanitizers; source copies are checked across the book. |
| R2-06 | Integer literal accumulation and arithmetic diagnose overflow, invalid operators and division errors. | Literal, add/multiply and division boundary regressions pass. |
| R2-07 | Leaf shapes and shape products are validated at inference entrypoints and in copied implementations. | All eleven foundations inference copies and all twelve application copies reject invalid shapes/counts; later backend copies are covered by propagation tests. |
| R2-08 | XLA logs are described as context, not proof of cache-key equality or physical-device identity. Actual key/version behavior and user-managed persistence compatibility are distinguished. | Main prose, source conclusion and local sample output corrected with primary references. |
| R2-09 | Framework numeric acceptance now raises on mismatch with explicit tolerances; explanatory prints remain. | Fifteen NumPy/AST guard/calibration regressions pass. Adapter tests establish fail-fast wiring, not fresh execution of unavailable framework stacks. |
| R2-10 | Stencil/CRR explanation now distinguishes array-level staging from statically expanded scalar graphs. | Source, Chapter 34 and quiz explanation corrected; twelve application drivers execute successfully. |
| R2-11 | Host, GPU, foundations and publication checks require explicit nonempty expected manifests and complete discovery. | Seventeen negative coverage cases pass. Manifest requires 78 host lessons, 3 CUDA drivers, 43 pages and 108 actual source files. |
| R2-12 | All-zero calibration chooses an in-range zero point for accepted custom subranges. | Positive-only, negative-only and default signed-int8 zero-data cases pass. |

## Current integrated validation

`python3 tools/validate_host.py --jobs 3 --sanitize --log-dir validation/second-pass-fixes/host-sanitized` compiles and runs all **78 host drivers** with AddressSanitizer, UndefinedBehaviorSanitizer and float-cast-overflow checks. All 78 pass. Generated child programs are exercised by their drivers but are not automatically sanitizer-instrumented by this runner; no claim of child-kernel sanitizer coverage follows from this host sweep.

The foundations owner independently ran all **38 original examples** with sanitizers and **21 focused regression suites**. Four targeted systems regressions also pass with ASan/UBSan, covering failed JIT builds, exact literals, propagated fusion bounds and CUDA-source generation. Application/shape/budget checks and framework-guard adapter tests are recorded alongside the relevant report.

- [Foundations corrections and commands](validation/second-pass-fixes/foundations.md)
- [Systems corrections and commands](validation/second-pass-fixes/systems.md)
- [Case-study corrections and commands](validation/second-pass-fixes/cases.md)
- [Validation coverage corrections](validation/second-pass-fixes/coverage.md)
- [Integrated host sanitizer manifest](validation/second-pass-fixes/host-sanitized/results.json)
- [Original reviewed findings and dispositions](audit/second-pass-2026-11-20/findings.json)

The current publication checks cover 43 pages, 108 actual downloadable source files and all 159 diagrams. The former count of 109 included a cache directory in the collection; that directory was skipped for source comparison. Current manifests count actual files and reject incomplete input sets.

## GPU and external-tool boundaries

The GPU remained off during this correction pass. The CUDA generators changed, so their earlier A100 outputs and device sanitizer logs are explicitly **previous-revision evidence**, not a fresh validation of these generator changes. The local CUDA regression checks emitted literal/diagnostic text and host process-status behavior; it does not compile or execute device code.

The Triton attention/dot kernels are unchanged by this correction pass; their prior A100 numerical/memcheck evidence remains applicable to the recorded source hashes and concrete cases. It is not an all-shape guarantee or a new GPU execution.

No fresh PyTorch/JAX/TVM/MLIR/Polly/CUTLASS compatibility sweep, Linux rerun or GPU performance claim is made. The new framework acceptance regressions use explicit local adapters where those dependencies are unavailable. Historical transcripts retain their original scope. Corrected sample stdout is from the stated current native macOS runs.
