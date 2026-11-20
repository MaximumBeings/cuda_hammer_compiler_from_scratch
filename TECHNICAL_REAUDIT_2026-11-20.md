# CUDA Hammer: independent second technical audit

**November 20, 2026 — reviewed published commit `69a3b19`.** This review found **12 confirmed findings: three P1, eight P2 and one P3**. That reviewed edition did not merit a technical all-clear. The earlier audit missed these defects; successful ordinary executions did not establish correctness.

The read-only review initially left the book unchanged. The subsequent authorized correction pass addresses all 12 findings; see [current fixes and validation](SECOND_PASS_CORRECTIONS.md). Original failure descriptions and reproductions below refer to the reviewed snapshot, not the repaired code. The GPU was not used for this second review or correction pass.

## Highest-priority findings

1. **Memory safety:** the unmodified constant-folding example removes `t0` but retains `t1`, then allocates storage for only one temporary. Its first sample writes beyond that allocation. AddressSanitizer confirms a heap-buffer overflow.
2. **Silent numerical miscompilation:** float literal formatting emits `1e-7` as `0.000000f`. A legal finite graph whose interpreter returns **10** produces generated code returning **0**. This affects literal-producing scalar, SIMD and CUDA paths.
3. **JIT/tuning integrity:** compiler exit status is discarded or ignored. A failed candidate build can load the previous fixed-name shared library and attribute its timing/output to code that never compiled. A silent exit-1 compiler is even reported as successful.

## Finding index

| ID | Priority | Finding | Primary location |
| --- | --- | --- | --- |
| R2-01 | P1 | Constant folding writes outside temporary storage | [004_constant_folding_pass.cpp:153](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part0/code/004_constant_folding_pass.cpp#L153) |
| R2-02 | P1 | Generated float literals silently lose valid values | [041_lowering_a_whole_fused_graph.cpp:500](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part4/code/041_lowering_a_whole_fused_graph.cpp#L500) |
| R2-03 | P1 | Failed JIT compilation can reuse stale code | [057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp:76](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part5/code/057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp#L76) |
| R2-04 | P2 | Fusion cap limits depth rather than emitted body size | [037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp:344](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part3/code/037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp#L344) |
| R2-05 | P2 | Loop and logical byte arithmetic can overflow | [034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp:42](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part3/code/034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp#L42) |
| R2-06 | P2 | Integer expression parsing and arithmetic are unchecked | [003_parsing_lowering_and_ir.cpp:48](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part0/code/003_parsing_lowering_and_ir.cpp#L48) |
| R2-07 | P2 | Later shape inference copies omit dimension validation | [033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds.cpp:187](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part3/code/033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds.cpp#L187) |
| R2-08 | P2 | XLA logging fields are mistaken for cache reuse guarantees | [25-xla-fusion-and-autotuning-in-a-real-production-compiler.md:1110](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part6/25-xla-fusion-and-autotuning-in-a-real-production-compiler.md#L1110) |
| R2-09 | P2 | Framework numerical checks print False without failing | [072_torch_compiles_real_two_stage_pipeline_dynamo_capture_then_inductor_codegen.py:111](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part6/code/072_torch_compiles_real_two_stage_pipeline_dynamo_capture_then_inductor_codegen.py#L111) |
| R2-10 | P2 | Stencil connectivity explanation confuses graph granularity with runtime knowledge | [091_the_whole_binomial_tree_as_one_static_graph_european_put.cpp:261](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part7/code/091_the_whole_binomial_tree_as_one_static_graph_european_put.cpp#L261) |
| R2-11 | P2 | Validation entrypoints report success with no inputs | [validate_gpu.py:12](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/tools/validate_gpu.py#L12) |
| R2-12 | P3 | All-zero calibration returns invalid parameters for custom subranges | [078_affine_quantization_math_and_why_codegen_needs_a_wider_accumulator.py:54](https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch/blob/69a3b19/docs/part6/code/078_affine_quantization_math_and_why_codegen_needs_a_wider_accumulator.py#L54) |

## Evidence and coverage

- [Foundations review](audit/second-pass-2026-11-20/foundations-review.md): Chapters 1–16, source copies and numerical/shape/fusion contracts. A fresh **ASan/UBSan campaign ran all 38 original examples: 37 pass, example 004 fails**. Targeted reproductions confirm the remaining findings.
- [Systems review](audit/second-pass-2026-11-20/systems-review.md): Chapters 17–24, toolchain appendices A/C/D/E/F, generated literal round trips, compiler-failure recovery and saved device evidence. Two independent compile/run reproductions confirm the P1 findings.
- [Case-study review](audit/second-pass-2026-11-20/cases-review.md): Chapters 25–34 and Appendix B, current primary compiler documentation, local application/NumPy runs and saved GPU source hashes. All 12 application examples still exit successfully, and the saved 12-case A100/memcheck evidence remains consistent within its disclosed contracts.
- [Validator coverage reproduction](audit/second-pass-2026-11-20/root/validation-tools.md): copied validation scripts in empty temporary fixtures return success with no sources, pages or manifest. GPU discovery was stubbed solely for this coverage experiment; it was not presented as GPU execution.
- [Machine-readable findings](audit/second-pass-2026-11-20/findings.json). Source snippets, reproduction scripts and actual stdout/sanitizer diagnostics are saved beside each review. Some standalone repros include the original workspace source by absolute path; use the listed checkout path or adjust it when relocating.

## Interpretation and limits

The prior **78/78 ordinary host exit-code results** remain observations of those unsanitized runs. They do not contradict the newly reproduced undefined behavior: a program can print the expected result and exit 0 after an out-of-bounds access. Likewise, the recorded A100 fixture comparisons and sanitizer results remain valid for those particular generated programs, not every legal graph or constant.

This is a broad implementation and fact-checking review, not a formal proof or exhaustive fuzzing campaign. Explicitly disclosed limitations were not counted as new defects, including restricted broadcasting, relaxed SIMD arithmetic, trusted educational caches, one-tile noncausal attention, and unavailable optional toolchain reruns. The low-priority custom calibration edge does not invalidate the default signed-int8 demonstration.

## Recommended correction order

Fix the three P1 issues first, sharing corrected temporary indexing, process-status handling and round-trip literal formatting across every copy. Add regressions that fail on the current code. Then propagate shape/counting checks, correct the fusion policy or its stated guarantee, make numerical checks fail on mismatch, require nonempty expected validation manifests, and repair the factual explanations. Rerun the sanitized host suite and affected generated-code tests before publishing another accuracy-clearance claim.
