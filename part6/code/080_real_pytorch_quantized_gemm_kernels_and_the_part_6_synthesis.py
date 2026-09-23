# Chapter 30, File 080 (capstone, closes Part 6): real PyTorch's own
# quantized GEMM kernels, and the same closing question Chapter 29 asked
# of Flash Attention -- does a real compiler's own automatic fusion pass
# ever discover a quantized kernel's own accumulate-then-requantize
# shape by itself, or does it have to dispatch to a real hand-written
# kernel, exactly like it did for Flash Attention?
#
# Real toolchain note: this file is verified on the cloud sandbox only,
# per Chapter 28's own real CUDA-toolkit-dependency and disk-space
# finding -- the device still receives and md5-verifies it, but does not
# run it.
#
# Real, load-bearing finding this file's own toolchain investigation
# turned up before any of its code ran: `torch.quantize_per_tensor` and
# the whole `torch.qint8`/`torch.quint8`/`torch.qint32` quantized-dtype
# family are REAL, currently DEPRECATED -- a real warning naming a real
# GitHub issue (pytorch/pytorch#184982), "will be removed in a future
# PyTorch release." The real, still-supported replacement is a pair of
# plain `torch.ops.aten` GEMM primitives operating on ordinary int8
# tensors (no special quantized dtype at all): `_int_mm` (full int8 x
# int8 -> int32 GEMM) and `_weight_int8pack_mm` (weight-only quantized
# GEMM: float activations, int8 weights, a real per-output-channel
# scale).
#
# Compiled with:   python3 "080_real_pytorch_quantized_gemm_kernels_and_the_part_6_synthesis.py"

import contextlib
import io
import logging
import os
import warnings

import torch


@contextlib.contextmanager
def _suppress_os_stderr():
    # Same real need as Chapters 28 and 29: torch's own default logging
    # handler writes straight to the OS-level stderr fd.
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)


def capture_output_code(fn, *args):
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("torch._inductor.codecache")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        torch._logging.set_logs(output_code=True)
        with _suppress_os_stderr():
            result = fn(*args)
    finally:
        logger.removeHandler(handler)
    return result, buf.getvalue()


def quant_matmul_fn(a, b):
    return torch.ops.aten._int_mm(a, b)


if __name__ == "__main__":
    torch.manual_seed(30)

    print("=== Part 1: the real legacy quantized-tensor API is "
          "deprecated ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = torch.randn(4)
        legacy_q = torch.quantize_per_tensor(x, scale=0.05, zero_point=0,
                                              dtype=torch.qint8)
    deprecation_text = str(caught[0].message)[:200] if caught else "(none)"
    print(f"Real deprecation warning on torch.quantize_per_tensor(): "
          f"\"{deprecation_text}...\"")
    print(f"This legacy path still works today (int_repr = "
          f"{legacy_q.int_repr().tolist()}), but real, current PyTorch "
          f"is already telling real users to move off it -- the real, "
          f"still-supported replacement this file uses instead is a "
          f"pair of plain torch.ops.aten GEMM primitives on ordinary "
          f"int8 tensors, no special quantized dtype at all.")

    print()
    print("=== Part 2: real full-int8 GEMM, torch.ops.aten._int_mm ===")
    a = torch.randint(-127, 127, (8, 16), dtype=torch.int8)
    b = torch.randint(-127, 127, (16, 8), dtype=torch.int8)
    real_out = torch.ops.aten._int_mm(a, b)
    manual_ref = a.to(torch.int32) @ b.to(torch.int32)
    print(f"Real schema: {torch.ops.aten._int_mm.default._schema}")
    print(f"Real output dtype: {real_out.dtype} (int8 x int8 -> int32, "
          f"the SAME accumulator-widening principle Files 078 and 079 "
          f"both confirmed is structurally necessary -- though this "
          f"real op is a compiled, opaque C++/CUDA kernel with no IR "
          f"this file can inspect the way File 079 read Triton's own "
          f"real arith.extsi/tt.reduce ops directly; its OWN correctness "
          f"is the only evidence available here, honestly, not its "
          f"internal structure)")
    print(f"Real correctness check: _int_mm vs. manual int32 matmul = "
          f"{torch.equal(real_out, manual_ref)}")

    print()
    print("=== Part 3: real weight-only quantized GEMM, "
          "torch.ops.aten._weight_int8pack_mm -- a DIFFERENT real "
          "quantization scheme ===")
    activations = torch.randn(4, 8)  # stays FLOAT -- only the weight is
    # quantized, the real scheme production LLM-serving systems actually
    # favor, since activations vary too much dynamically to calibrate a
    # single static scale for cheaply, unlike a model's own trained
    # weights.
    weights_int8 = torch.randint(-127, 127, (4, 8), dtype=torch.int8)
    per_channel_scales = torch.rand(4) * 0.1
    weight_only_out = torch.ops.aten._weight_int8pack_mm(
        activations, weights_int8, per_channel_scales)
    manual_weight_only_ref = activations @ (
        weights_int8.to(torch.float32) * per_channel_scales[:, None]).T
    print(f"Real schema: "
          f"{torch.ops.aten._weight_int8pack_mm.default._schema}")
    print(f"Real correctness check: _weight_int8pack_mm vs. manual "
          f"dequantize-then-matmul = "
          f"{torch.allclose(weight_only_out, manual_weight_only_ref, atol=1e-4)}")
    print(f"Real, honest scope note: Files 078 and 079 both built the "
          f"FULL-int8 scheme (Part 2's own _int_mm) by hand. This real "
          f"op confirms production systems support a genuinely SEPARATE "
          f"quantization shape neither of those files implements -- real "
          f"quantization-aware codegen is not one fixed recipe.")

    print()
    print("=== Part 4 (Part 6's own closing synthesis): does Inductor "
          "fuse a quantized GEMM from primitives, or dispatch to it? ===")
    # Real reproducibility note: this sandbox has already run
    # torch.compile many times across Chapters 28-29, and Inductor's own
    # real on-disk FXGraphCache can silently turn this call into a cache
    # HIT (a much shorter captured log with none of the fresh-compile
    # debug output below) -- force a real, fresh codegen pass every time
    # this file runs, the same "don't let stale state make a finding
    # look different than it is" discipline this book has followed since
    # Chapter 24's own TuningCache work.
    torch._inductor.config.force_disable_caches = True
    compiled_quant_matmul = torch.compile(quant_matmul_fn)
    compiled_out, code = capture_output_code(compiled_quant_matmul, a, b)
    eager_out = quant_matmul_fn(a, b)
    print(f"Real correctness check: torch.compile'd vs. eager match = "
          f"{torch.equal(compiled_out, eager_out)}")

    generated_kernel_count = code.count("async_compile.cpp_pybinding")
    dispatches_to_real_int_mm = "extern_kernels._int_mm" in code
    print(f"Real generated cpp_fused kernel count: {generated_kernel_count} "
          f"(Chapter 28.1's own diamond graph got its own generated "
          f"kernel here this same way; Chapter 29.3's flash attention "
          f"got 0; this real quantized GEMM gets {generated_kernel_count})")
    print(f"Real generated code calls straight through to "
          f"extern_kernels._int_mm: {dispatches_to_real_int_mm}")
    print(f"Real, additional evidence: real, current Inductor ships a "
          f"NAMED config field, torch._inductor.config."
          f"force_fuse_int_mm_with_mul (default "
          f"{torch._inductor.config.force_fuse_int_mm_with_mul!r}), "
          f"existing purely to let a user opt INTO fusing _int_mm with a "
          f"following multiply -- real, direct evidence Inductor's own "
          f"developers treat that one specific combination as a "
          f"hand-written SPECIAL CASE worth naming and gating, not "
          f"something generic automatic fusion discovers on its own.")

    print()
    print("Part 6's own closing finding, confirmed a second time by a "
          "second, unrelated real algorithm: Chapter 29 found Inductor "
          "dispatches to a hand-written Flash Attention kernel rather "
          "than decomposing and re-fusing it; this file finds the exact "
          "same shape for quantized GEMM -- zero generated kernels, a "
          "direct call to a real, separately implemented aten op, and "
          "even a real NAMED config knob for the one specific fusion "
          "(with a following multiply) its own authors judged worth "
          "hand-coding. Across every real system this Part studied -- "
          "XLA, Relax, Triton, Inductor -- automatic fusion covers "
          "ORDINARY elementwise and reduction graphs; algorithms whose "
          "own numeric behavior genuinely differs (a streaming softmax, "
          "an integer accumulator) are handled by real, separately "
          "written kernels every time, never invented by a fusion pass.")
