# Follow-up review — November 20, 2026

Reviewed baseline: `a01589b`. This follows, and does not replace, the earlier technical review and the 12 second-pass corrections.

## Findings fixed

1. Getting Started omitted a clone/build path, fresh host-tool installation, driver/toolkit separation, environment setup and optional framework boundaries. It now gives Ubuntu and macOS host paths, a concrete Ubuntu x86_64 CUDA 12.8 recipe, actual GPU architecture selections and first-failure troubleshooting. Installation references are primary NVIDIA/project documentation. Package installation was reviewed against those references, not executed with administrator privileges on a fresh GPU machine.
2. Appendix A's historical 71-file C++14 sweep could be mistaken for the current 78-driver collection's requirement. The page now labels that sweep historical and directs current readers to C++17 and the maintained setup guide.
3. Six chapter transitions contained production editing instructions. These have been removed. Chapter 17's absolute “never executed” CUDA claim now points to the A100 baseline and the pending fresh-generator run.
4. Three shared-value diagrams used `c` without depicting its input. The generator and all three SVGs/embeds now show it. The overview and a representative shared-value diagram were rasterized and inspected for readable, unclipped labels; all 159 SVGs pass the existing structural checks. Syntax-highlighted code now soft-wraps without changing downloadable source bytes; fixed-column text diagrams retain their geometry.
5. The GPU runner wrote into the archived A100 directory, risking evidence replacement. It now uses a fresh dated directory by default, accepts `--log-dir`, rejects nonempty destinations and records source hashes, commands, UTC timestamps and exits. A failed/skipped driver cannot trigger sanitizer checks on old fixed-path child binaries. Tests cover deliberate driver failures, missing diagnostic summaries and existing-evidence preservation. The lesson drivers still use fixed `/tmp/hammer_ch18_*` paths; the setup guide requires running one GPU validation at a time on a host.
6. No automatic verification workflow protected publishing. A new workflow runs publication/coverage checks, 78 sanitized host drivers and focused regressions on Linux and macOS before deploying `main`. It has no CUDA runner and makes no GPU-validation claim.
7. The source-level publication checker did not check built HTML fragments. A separate strict checker now rejects missing assets/pages/anchors, duplicate IDs, escaping relative paths and absent/empty sites, with negative regressions. No prior acceptance criteria were relaxed.

## Checks actually run locally

- Strict MkDocs build, 43 content pages / 108 downloadable source embeds / 159 SVGs.
- All built relative links and anchors across 44 HTML pages, including 404.
- All 78 standalone host drivers with AddressSanitizer, UndefinedBehaviorSanitizer and float-cast-overflow instrumentation.
- All 21 foundations regression cases.
- All 15 NumPy/AST case-study acceptance/calibration checks; these do not execute unavailable frameworks.
- All 12 application safety drivers, including four actual fusion-budget checks.
- Four systems C++ regression programs plus the numerical comparison regression, each with sanitizers.
- All 17 existing negative coverage cases, plus five publication/GPU-runner test methods with deliberate negative cases.

Commands, fresh stdout and source/evidence hashes are under `validation/final-followup-2026-11-20/`. Generated child programs are not automatically sanitizer-instrumented by the host driver sweep. No source algorithm in the 108 downloadable lessons was changed by this follow-up.

## Remaining limits

No GPU was started or accessed. The second-pass CUDA generator changes still need actual device compilation/execution and diagnostics. Earlier A100 device records remain historical for that earlier revision. No new TVM, MLIR, Polly, CUTLASS or full framework compatibility sweep was performed. Historical captures are preserved rather than rewritten as current results. Passing this finite set of lessons and counterexamples does not establish correctness for arbitrary graphs or all target/toolchain combinations.
