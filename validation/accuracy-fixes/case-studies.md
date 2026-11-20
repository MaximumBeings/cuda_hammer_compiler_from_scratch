# Case-study technical audit: Chapters 25–34 and Appendix B

Reviewed canonical source and embedded Markdown for XLA, TVM, Triton,
TorchDynamo/Inductor, attention/quantization and the four application domains.
Changes are educational correctness fixes, not production certification.

## Confirmed findings and corrections

1. **Triton attention KV tails and arithmetic precision.** File 076 loaded a
   final incomplete key/value tile without masking; its compile-only 48/16
   shape concealed the out-of-bounds path. Added masked loads and -infinity
   padded scores. Both float32 dot operations now request IEEE precision
   rather than depending on NVIDIA's default TF32. Explicit contract: one
   program owns exactly BLOCK_M query rows; positive KV length, power-of-two
   dimensions; scaled noncausal attention. This is not arbitrary query-tail,
   causal, batched or production attention. Source and embedded block agree.
   **Real A100 execution now passes** six attention cases, including masked tails,
   also under Compute Sanitizer memcheck; historical inline IR remains labeled
   as predating the correction. Fresh compiler output is saved separately.

2. **Quantization calibration and accumulation.** File 078 used observed
   min/max without including zero, so one-sided ranges could saturate after
   clamping the zero point; constant zero caused division by zero. Calibration
   now includes zero, has an all-zero fallback, and rejects empty/nonfinite
   data. Quantization validates positive finite scale and signed-int8 bounds.
   Half-step error is explicitly limited to unclipped values. The purported
   int32 NumPy sum now actually uses `sum(dtype=np.int32)` and is checked
   against an int64 reference. Added positive/negative/constant/zero and bad
   input regressions. File 079 explicitly constrains positive power-of-two
   N to <=65536 so worst-case signed-int8 sums fit int32. It uses zero-point
   zero; asymmetric quantization requires additional terms not implemented.

3. **No invented numerical GPU validation.** Chapter 29 previously claimed
   three independent implementations agreed, despite Triton only being
   compiled. Corrected source, prose and Appendix B: NumPy numerical checks,
   recorded PyTorch CPU comparisons, and offline Triton compilation are
   separate evidence. Changed IR was not silently presented as rerun evidence.

4. **Overgeneralized compiler limits.** One CPU SDPA/_int_mm dispatch does
   not prove XLA, Relax or Inductor can never generate/fuse specialized
   attention or quantization kernels. Narrowed Files 077/080 and chapter/quiz
   conclusions; omitted explicitly identified obsolete universal conclusions
   from historical transcript excerpts. Added FlexAttention and XLA codegen
   references. AutotuningLog fields are identified as a contextual illustration,
   not proof of XLA's complete cache-key algorithm.

5. **Scalar graph capture.** File 073 and Chapter 28 incorrectly described
   Tensor.item as universally untraceable. The example now explicitly sets
   capture_scalar_outputs=False before capture and describes that configuration;
   supported scalar capture can be enabled, while data-dependent control flow
   may still break graphs. No new torch execution is claimed locally.

6. **Quantization migration/API stability.** Two underscore-prefixed aten
   primitives are low-level examples, not the complete supported public
   replacement for deprecated quantized dtypes. Chapter 30 and File 080 now
   distinguish these internals from the plain-dtype/TorchAO migration direction.

7. **Host numerical validation rejects nonfinite output.** Array comparisons
   in Files 082/083/086 previously accepted NaN because `fabs(NaN)>tol` is false.
   They now reject nonfinite inputs and include NaN/infinity regressions.
   File 089's per-element stencil comparisons also explicitly reject nonfinite
   values. Actual generated NEON child programs were compiled/executed on this
   Apple Silicon host. Scalar `< tolerance` checks elsewhere already reject NaN.

8. **Attention comparison tolerances and chapter structure.** File 075 now
   uses rtol=0 for the claimed absolute tolerances and fails on numerical
   mismatch, rather than only printing a boolean. Corrected bit-exact language.
   Chapter 30's misplaced Triton section now precedes File 079 and a distinct
   30.3 introduces the PyTorch capstone before File 080. Appendix B's 9.6us
   launch-cost answer is explicitly one measured setup, not a hardware constant.

## Actual validation

- All 12 standalone C++ application drivers (081–092) compiled with
  `clang++ -std=c++17 -Wall -Wextra -O2` and ran with exit 0. Individual logs
  are alongside this report. Changed 082/083/086/089 were rerun after fixes.
- Files 075 and 078 ran successfully with NumPy 2.5.5 in temporary environment
  `/tmp/cuda-hammer-case-venv`. Logs: `075-numpy.log`, `078-quantization.log`.
  File 075's three block-size checks pass; File 078's calibration regressions
  pass and its random round-trip max error is below the stated half-step bound.
- All Part 6 Python sources and the GPU runner pass Python syntax compilation.
- A remote NVIDIA A100-PCIE-40GB run used Python 3.12.3, PyTorch 2.7.0,
  Triton 3.3.0, NumPy 1.26.4 and JAX 0.6.0. Actual metadata, all 10 CPU probe
  transcripts (063/064/065/072/073/074/075/077/078/080), fresh AOT output
  (076/079), GPU checks and sanitizer output are in `a100-case-studies/`.
  All final probes exited 0. This is not a sweep of every framework release.
- The GPU runner passed all 12 real device checks: six scaled/noncausal
  attention cases and six int8 dots. Maximum observed attention absolute
  error was 2.384185791015625e-7 against independent float64 references,
  below the declared 3e-5 tolerances. The same runner passed under
  Compute Sanitizer memcheck with `ERROR SUMMARY: 0 errors` and exit 0.
  Source SHA-256 hashes match the downloaded manifest.
- Two compatibility/diagnostic issues were discovered and corrected during
  remote testing: optional `list_flash_attention_impls` introspection is now
  guarded for older torch; File 079 no longer assumes a specific SSA name for
  its IR excerpt. Its operation counter excludes `tt.reduce.return` from the
  count of actual `tt.reduce` operations. CPU-path explanations no longer
  equate CUDA configuration flags with execution device or assume no GPU.
- TVM was not installed remotely and its examples were not rerun. No GPU timing,
  performance claim, physical Jetson test or complete compatibility sweep is
  claimed. Successful offline compilation remains distinct from GPU execution.

## Reproducing the completed GPU validation

These checks were completed on an A100 (sm_80). A single NVIDIA A10 (sm_86),
A100 (sm_80) or newer compatible NVIDIA GPU is
sufficient for these small correctness tests; no large VRAM model is needed.
Use a compatible CUDA-enabled PyTorch plus Triton installation on Linux. The
runner uses PyTorch for allocation and independent float64 references; it does
not need JAX or TVM. NumPy is needed only to run the separate host examples.
Record package versions, device and driver before interpreting results.

From the repository root:

```bash
python validation/accuracy-fixes/gpu_case_validation.py
compute-sanitizer --tool memcheck --error-exitcode 1 python validation/accuracy-fixes/gpu_case_validation.py
```

This executes six attention cases `(query rows, KV rows, D, KV tile)`:
`(16,1,16,16)`, `(16,17,16,16)`, `(16,31,16,16)`, `(16,48,16,16)`,
`(32,49,32,32)`, `(16,65,64,32)`, using float32 kernels against float64
scaled-softmax references (atol=rtol=3e-5), plus random and extreme -128 int8
inputs for dot lengths 32, 64 and 128. Each launch synchronizes and asserts
correct output. The runner fails without a real visible CUDA device.

Files 069/070/071/076/079 can separately be run for fresh AOT compiler dumps.
Their hardcoded sm_80 target is an offline target; launch-time Triton JIT uses
the actual device. Those offline scripts alone are not runtime tests.

## Primary sources checked

- [XLA GPU architecture](https://openxla.org/xla/gpu_architecture): native and
  Triton emitters plus library selection; CPU observations do not define GPU behavior.
- [TVM MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html):
  current s_tir namespace, actual hardware measurements and Relax integration.
  The book's explicit version-specific namespace discussion is not itself an error.
- [Triton dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html):
  float32 dot precision modes and default TF32.
- [FlexAttention](https://pytorch.org/blog/flexattention/): compiler-generated
  specialized attention, contradicting the universal external-kernel conclusion.
- [PyTorch quantized-dtype deprecation](https://github.com/pytorch/pytorch/issues/184982)
  and [migration guidance](https://docs.pytorch.org/docs/2.14/quantization.html):
  plain dtypes/explicit metadata and TorchAO, rather than two private ops as a full API.
- [MinMaxObserver](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html)
  and [observer source](https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/observer.py):
  affine quantization and degenerate-range handling.

The stencil's demonstrated r=0.25 lies in the 1-D FTCS stability interval,
its zero-padding permits heat loss, and the fixed-depth CRR examples use
positive volatility and an admissible risk-neutral probability. These narrow
examples do not certify arbitrary boundary conditions or finance inputs.
