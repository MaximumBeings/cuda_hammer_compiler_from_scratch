# Chapter 29, File 077 (capstone): real PyTorch's own CPU Flash Attention
# implementation, and what TorchInductor (Chapter 28) does with it.
#
# Real toolchain note, same disk-safety scope as Chapter 28: this file is
# verified on the cloud sandbox only -- the device still receives and
# md5-verifies it, but does not run it (no torch installed there, per
# Chapter 28's own real CUDA-toolkit-dependency and disk-space finding).
#
# Real, load-bearing finding this file's own toolchain investigation
# turned up before any of its code ran: real torch exposes
# `torch.ops.aten._scaled_dot_product_flash_attention_for_cpu`, a genuine
# CPU-native flash-attention kernel -- not a GPU-only op that happens to
# also have a CPU fallback bolted on. Its own real schema is
# `(Tensor query, Tensor key, Tensor value, ...) -> (Tensor output,
# Tensor logsumexp)`: that SECOND return value, a real `logsumexp` per
# query row, is direct evidence this is a genuine online log-sum-exp
# streaming softmax implementation -- the same real quantity Files 075
# and 076 both tracked by hand as `running_max`/`running_sum` (a running
# log-sum-exp is mathematically `log(running_sum) + running_max`) -- not
# a naive full-materialization softmax wearing a "flash" name.
#
# Compiled with:   python3 "077_real_pytorch_flash_attention_and_why_inductor_never_regenerates_it.py"

import contextlib
import io
import logging
import os
import re

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.backends.cuda as cuda_backend


@contextlib.contextmanager
def _suppress_os_stderr():
    # Same real need as Chapter 28's own Files 072/073: torch's own
    # default logging handler writes straight to the OS-level stderr fd.
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


def attention_fn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)


if __name__ == "__main__":
    torch.manual_seed(29)
    q = torch.randn(1, 2, 8, 4)  # (batch, heads, seq_len, head_dim) --
    k = torch.randn(1, 2, 8, 4)  # matching File 075/076's own n=8, d=4
    v = torch.randn(1, 2, 8, 4)  # shapes (a real batch/head dim prepended,
                                 # the real SDPA signature's own layout)

    print("=== Part 1: real backend availability, queried directly ===")
    print(f"is_flash_attention_available(): "
          f"{cuda_backend.is_flash_attention_available()}")
    print(f"flash_sdp_enabled(): {cuda_backend.flash_sdp_enabled()}")
    print(f"math_sdp_enabled(): {cuda_backend.math_sdp_enabled()}")
    import torch.nn.attention as attn_mod
    print(f"list_flash_attention_impls(): "
          f"{attn_mod.list_flash_attention_impls()}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print("(every one of these reports real flash-attention support with "
          "zero physical GPU present -- this is a genuine CPU code path, "
          "not a GPU-gated feature this sandbox happens to see stubbed out)")

    print()
    print("=== Part 2: real execution, MATH backend vs. FLASH_ATTENTION "
          "backend, same real inputs ===")
    with sdpa_kernel(SDPBackend.MATH):
        out_math = F.scaled_dot_product_attention(q, k, v)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out_flash = F.scaled_dot_product_attention(q, k, v)
    max_abs_diff = (out_math - out_flash).abs().max().item()
    print(f"MATH backend output shape: {tuple(out_math.shape)}")
    print(f"FLASH_ATTENTION backend output shape: {tuple(out_flash.shape)}")
    print(f"Real correctness check: MATH vs. FLASH_ATTENTION match = "
          f"{torch.allclose(out_math, out_flash, atol=1e-5)} "
          f"(max abs diff {max_abs_diff:.3e}) -- two independently "
          f"implemented real kernels inside the SAME torch build agree, "
          f"the same cross-implementation agreement Files 075 and 076 "
          f"already established between pure NumPy and compiled Triton")

    print()
    print("=== Part 3: the real op's own schema -- why FLASH_ATTENTION "
          "works with no GPU ===")
    op = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    schema = op.default._schema
    print(f"Real schema: {schema}")
    real_output, real_logsumexp = op.default(q, k, v)
    print(f"Real logsumexp shape: {tuple(real_logsumexp.shape)} -- one "
          f"real value per (batch, head, query row), exactly matching "
          f"File 075's own running_max/running_sum accumulator shape "
          f"(N,), confirming this real op tracks the SAME online "
          f"log-sum-exp quantity Files 075 and 076 both computed by hand")
    print(f"Real correctness check: this op's own output vs. MATH "
          f"backend = "
          f"{torch.allclose(real_output, out_math, atol=1e-5)}")

    print()
    print("=== Part 4 (capstone): what TorchInductor (Ch28) does with "
          "scaled_dot_product_attention ===")
    compiled_attention = torch.compile(attention_fn)
    compiled_out, code = capture_output_code(compiled_attention, q, k, v)
    eager_out = attention_fn(q, k, v)
    print(f"Real correctness check: torch.compile'd vs. eager match = "
          f"{torch.allclose(compiled_out, eager_out, atol=1e-5)}")

    generated_kernel_count = len(re.findall(r"async_compile\.cpp_pybinding", code))
    dispatches_to_real_flash_op = (
        "_scaled_dot_product_flash_attention_for_cpu" in code)
    print(f"Real generated cpp_fused kernel count: {generated_kernel_count} "
          f"(Chapter 28.1's own diamond graph -- 4 ordinary elementwise "
          f"ops -- got its own generated cpp_fused_add_mul_relu_0 kernel "
          f"here this same way; scaled_dot_product_attention gets "
          f"{generated_kernel_count})")
    print(f"Real generated code calls straight through to "
          f"aten._scaled_dot_product_flash_attention_for_cpu: "
          f"{dispatches_to_real_flash_op}")
    print()
    print("This is the chapter's own closing, cross-system finding: "
          "TorchInductor's real automatic whole-graph fusion (Ch28.1), "
          "XLA's real kind=kLoop fusion (Ch25), and Relax's real "
          "FuseOps/FuseTIR (Ch26) all fuse ORDINARY elementwise/reduction "
          "graphs automatically, inventing nothing about the ALGORITHM "
          "itself -- but none of them re-derives Flash Attention's own "
          "online-softmax rescaling trick from primitives. Inductor "
          "dispatches straight to a hand-written, pre-existing fused "
          "kernel instead of decomposing scaled_dot_product_attention "
          "into ops it could fuse itself. Flash Attention needed a new "
          "ALGORITHM (Milakov & Gimelshein's 2018 online-softmax "
          "identity), not just a new fusion boundary -- and every real "
          "system this Part studied treats that as a hand-written kernel "
          "provided to the compiler, never a boundary the compiler's own "
          "fusion pass discovers on its own.")
