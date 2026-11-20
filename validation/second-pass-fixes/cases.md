# Second-pass case-study resolutions

No GPU connection or provisioning was used. These changes do not modify the
Triton kernels (076/079) or their previously captured A100 evidence.

## SC-1 — XLA log context is not a cache-reuse guarantee

Replaced the main Chapter 25 explanation and File 065's source/output conclusion.
Log fields are diagnostic context, not key equality or physical-device identity.
The explanation now links the version-specific actual key (device-model string,
canonical HLO, cache version) and OpenXLA's user-managed persisted-cache version
separation. The dataclass illustration remains explicitly placeholder data.
File 065 was run locally and its current labeled stdout refreshed.

Primary sources:
- https://github.com/openxla/xla/blob/main/xla/service/gpu/autotuning/autotune_cache_key.h
- https://github.com/openxla/xla/blob/main/xla/service/gpu/autotuning/autotune_cache_key.cc
- https://openxla.org/xla/persisted_autotuning

## SC-2 — Numerical checks now fail rather than only print False

Added actual acceptance assertions in Files 066/067/068 (TVM/NumPy result
checks), 072/073/074 (Inductor), 077 (three CPU attention comparisons) and 080
(three quantized comparisons). All assertions state atol, rtol and equal_nan=False.
TVM output is checked finite before numerical acceptance. File 068 now verifies
its compiled result before timing it. Existing bit-exact diagnostic prints in
066/067 remain informative; 067 accepts the disclosed one-ULP difference with
atol=1e-6/rtol=1e-5, whereas 066's elementwise add requires exact equality.
Torch checks retain the prior comparison tolerances explicitly; integer
comparisons use atol=rtol=0.

`tests/cases_regressions.py` executes each canonical assertion AST. For Torch
it supplies a NumPy assertion adapter, testing argument/tolerance wiring and
nonzero deliberate-mismatch exit, not Torch internals or compiler execution.
The TVM result-check AST uses the actual NumPy assertion implementation.
All 12 guard checks and three calibration cases pass (15 checks total).
These frameworks are not installed locally and were not rerun in this pass;
previous A100-host CPU logs remain historical observations before these guards.
No new TVM/Torch stdout was fabricated.

## SC-3 — Static connectivity versus representation granularity

Chapter 34, source comments/prints in 091/092 and Appendix B now explain that
both fixed-grid stencils and the CRR lattice have statically known connectivity.
Earlier examples choose array-valued nodes with host shifts; the CRR example
expands fixed-depth scalar nodes. A stencil can also be expanded into scalar
nodes, trading staging for graph size. Removed the false runtime-known/irregular
contrast and updated current native-host output and group-count discussion.

## SC-4 — Valid all-zero calibration in custom signed-int8 subranges

File 078 clamps the all-zero fallback's integer zero point into [qmin,qmax].
Default, positive-only [1,127] and negative-only [-128,-1] subranges now all
return valid parameters and round-trip zeros exactly. Regressions are in the
example itself and `tests/cases_regressions.py`. File 078 was run locally with
NumPy 2.5.5; current output is in `078-current.log` and the chapter.

## Propagated standalone-copy safety fixes

All 12 Part7 copies now validate positive declared leaf dimensions and checked
shape products. Broadcast results are validated; incompatible dimensions are
rejected. Their numElements helpers use checked multiplication. Part7's local
LoopNest structs have no totalIterations method and there are no logical-byte
estimators requiring the corresponding change.

The four fusion copies (083/086/089/092) now use the foundation's actual
body-step budget, summing inlineable subtrees and counting the final operation,
including Sum, instead of using longest-path depth. Invalid budgets are rejected.
The American example now has 72 total nodes: 38 fused elementwise groups and
14 plain compute nodes, all 52 dispatched on the vector path; prose/output was
updated from the former 55-node/35-group claim.

The systems agent's Part7 finite-float literal, checked subprocess status,
round-trip-safe comparison output and finite C-locale parsing changes were
preserved. All current Part7 source listings were synchronized after those edits.

## Actual verification and artifacts

- `cases-regressions.log`: 15 NumPy/AST numerical acceptance/calibration checks.
- `part7-safety.log`: all 12 standalone copies pass zero/negative leaf shape,
  shape/count overflow, valid scalar extent and incompatible broadcast checks;
  four copies also pass branching-body budget and invalid-budget regressions.
- All 12 application drivers 081–092 compiled with clang++ C++17/Wall/Wextra/O2
  and ran with exit 0 on Apple Silicon/macOS. Final runs include the systems
  agent's updated float parser; full logs and separate stdout are alongside
  this report. Generated NEON child programs were actually compiled and run.
- All Part6 Python sources and the new host regressions pass syntax compilation.
- Chapter 25/30 local NumPy/std-library outputs and all Part7 application outputs
  were refreshed and labeled as one current local host run, replacing paired
  historical captures instead of claiming two fresh platforms.
- Part6/7 canonical source listings match current sources. The whole-book
  checker is run again by the integrating agent after other agents finish their
  separate Part0–5/appendix edits. Existing diagram figures were preserved.
- No new x86 runtime, framework compiler execution, GPU run or GPU performance
  claim is made. No dependency installation was needed for this pass; NumPy
  runs used the existing temporary `/tmp/cuda-hammer-case-venv` environment.
