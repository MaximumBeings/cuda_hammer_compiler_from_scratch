# Systems second-pass fix mapping

## S2-SYS01: process status and fresh artifacts — fixed

All copied Part4/5/7 and AppendixF subprocess helpers now inspect pclose/WIFEXITED/WEXITSTATUS and throw on failure rather than inferring success from empty output. Source-write failures are checked. CUDA diagnostics permit exit77 only at explicit generated-program execution sites, never at compiler calls.

The Chapter20 JIT helper and Part5/Appendix108 copies compile private staged source and output files, check compiler exit status, then rename a successful candidate to the final shared-library path. Failed candidates cannot proceed into loading/timing/cache insertion. A previous successful artifact is left intact, but not accepted as the failed new candidate. Quoted paths are escaped. Compiler warnings remain available as text and do not turn a successful process into failure. Staged files are removed on success or exception.

The same helper implementation is copied deliberately to preserve each downloadable lesson as a self-contained single-file program; scalar/vector JIT and autotuner paths use this implementation consistently.

## S2-SYS02: finite float literals and stdout — fixed

All literal-producing scalar/SIMD/CUDA paths in040–047 and083/086/089/092 now use the same finite floatLiteral implementation: classic-locale hexadecimal source literal plus f suffix. It preserves the exact binary float, including negative zero and subnormals; nonfinite constants are explicitly rejected. Fixed-six-decimal std::to_string is no longer used to serialize floats.

Generated comparison stdout uses nine significant decimal digits. Explicit C-locale strtof_l parsing avoids libc++ float-stream underflow truncation and std::stof ERANGE exceptions for representable subnormals, while rejecting malformed/nonfinite output and true underflow-to-zero. The host test checks both source-literal and returned-text bit patterns for nine finite values (signed zeros,1e-7,precision-sensitive neighbors,normal limits and smallest subnormal). The original failing graph now returns10 for1e8*1e-7 and retains the small constant in comparison output.

## Foundation fix propagation

Propagated checked count products, validated leaf/broadcast shapes and true fused-body step-budget implementation into later Part4/5 copies and Appendix106/108. Case-study agent propagated Part7 foundation contracts. A balanced-tree regression in the actual Chapter17.3 backend confirms every fused body obeys cap3 and preserves result36; invalid leaf dimensions and count overflow are rejected before allocation/codegen.

## Validation actually performed

- All28 assigned plain C++ host lessons compiled/executed successfully on native macOS arm64 with Apple Clang,C++17,O2,Wall,Wextra. Per-example logs and fresh stdout are in this directory; systems-host-results.json records exits. Timings are local observations, not performance guarantees. The final explicit-C-locale/ERANGE path is covered by targeted tests; root performs a final current-source global sanitized sweep.
- systems_jit_failure_regression.cpp passes real diagnostic compilation failure, silent compiler exit1, previous-artifact preservation, successful-warning promotion, quote-containing paths and staged-file cleanup.
- systems_float_literal_regression.cpp passes all9 exact finite literal bit patterns, all9 comparison-stdout bit patterns, and the amplified small-constant graph result10.
- systems_cuda_text_regression.cpp passes host-only actual CUDA-emitter checks for exact small constant,9-digit stdout, fail-fast diagnostics and compiler77 rejection. No generated CUDA device code compilation/execution is claimed.
- systems_fusion_propagation_regression.cpp passes cap/shape/count propagation tests.
- All three .cu generator drivers pass fresh host-only clang++ -x c++ -fsyntax-only checks. Commands/exits are recorded in their host-syntax logs.
- Chapter17/20 include concrete fresh regression output; first fresh-local host stdout captures in Parts4/5 have been refreshed and labeled. Historical cloud/device captures are distinguished. Code blocks match downloadable sources and159diagrams are retained; systems-book-structure.txt contains the final checker result.

## GPU and platform limits

No GPU connection, remote command, provisioning or GPU execution occurred during these fixes. The A100 four-fixture numeric/12sanitizer transcript is explicitly previous69a3b19 baseline evidence. Current literal/process/fusion-cap changes have host and host-only CUDA-text coverage, not a fresh CUDA runtime/sanitizer result. Local nvcc/ComputeSanitizer unavailable. The GPU remains off as requested. Existing unchanged Triton evidence is managed separately by the case-study/root reports.

No commit made by this agent. Root owns integration/publishing and the empty-input validator coverage fix.
