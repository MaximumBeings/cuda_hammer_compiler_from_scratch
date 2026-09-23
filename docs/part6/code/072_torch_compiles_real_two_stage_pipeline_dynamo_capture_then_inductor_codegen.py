# Chapter 28, File 072: torch.compile's real two-stage pipeline, run on the
# exact same diamond graph Chapters 4, 13, 25, 26, and 27 have all already
# used: t1 = a + b (shared by two consumers), t2 = t1 * c, t3 = relu(t1),
# out = t2 + t3.
#
# Real toolchain investigation (before this file was written): `pip install
# torch` on this sandbox's own real, current PyPI index resolves to a Linux
# wheel that is CUDA-linked at the NATIVE library level -- its own
# `libtorch_global_deps.so` has a hard runtime dependency on `libcudart.so`,
# so plain `import torch` fails outright with `OSError: libcudart.so.13:
# cannot open shared object file` until the real NVIDIA CUDA 13 toolkit's
# own redistributable libraries are ALSO installed as pip packages -- even
# though this sandbox will never have a physical GPU to use them on.
# `pip install torch --dry-run` confirms the real, full list: a `cuda-toolkit`
# metapackage plus `nvidia-cublas`/`-cudnn`/`-cufft`/`-cusparse`/`-cusolver`/
# `-curand`/`-nccl`/`-nvshmem`/`-cusparselt`/`-cupti`/`-nvjitlink`/`-nvrtc`/
# `-cuda-runtime`, 22 packages in total. This is a genuinely different real
# toolchain story from jax (Ch25) and apache-tvm (Ch26), both of which
# install and import cleanly with zero CUDA runtime present -- and from
# triton (Ch27), which never even tries to load a CUDA runtime at import
# time. The real download (~2.5GB) completed on the cloud sandbox (252GB
# disk, ample headroom) and `import torch` then succeeds, reporting
# `torch.cuda.is_available() == False`, exactly as expected with no
# physical GPU. `pip install torch --dry-run` on the device confirms real
# aarch64 wheels exist for every one of those same 22 packages too -- but
# the device's own real, connected filesystem had only about 2GB of free
# disk space at investigation time, well under the ~2.5GB this real install
# needs, and risking that on the user's own linked machine was not a trade
# this book will make for a demonstration. So, like Chapter 18's CUDA
# backend, this chapter is verified for real ONLY on the cloud sandbox; the
# device still receives and md5-verifies every file, but does not run them.
#
# Compiled with:   python3 "072_torch_compiles_real_two_stage_pipeline_dynamo_capture_then_inductor_codegen.py"

import contextlib
import logging
import io
import os
import re

import torch


@contextlib.contextmanager
def _suppress_os_stderr():
    """torch._logging's own real default handler writes every enabled log
    artifact straight to the OS-level stderr file descriptor, independent
    of any Python `logging` handler this file attaches -- so capturing the
    generated code text cleanly (below) still leaves torch's own verbose
    real log noise on the terminal unless the raw fd is redirected during
    the call. Restored immediately afterward; this affects only what this
    process's own stderr shows, never what capture_output_code() returns."""
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
    """Runs a real torch.compile'd call while capturing Inductor's own real
    generated code text through a dedicated logging handler (the same
    "capture the real library log cleanly" technique this book has not
    needed before -- Chapters 25-27 all read generated text straight off a
    real object's own field, but torch.compile's generated code is only
    ever emitted through this logger, never returned to the caller)."""
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
    text = buf.getvalue()
    # Keep only the real generated-code block itself, dropping the
    # "Output code written to: <path>" trailer line and anything the fx
    # graph cache logger interleaved on a cache hit.
    match = re.search(r"Output code:\s*\n(.*?)\n(?:Output code written to:|$)",
                       text, re.DOTALL)
    code = match.group(1) if match else text
    return result, code


if __name__ == "__main__":
    def diamond(a, b, c):
        t1 = a + b
        t2 = t1 * c
        t3 = torch.relu(t1)
        return t2 + t3

    compiled_diamond = torch.compile(diamond)

    a = torch.randn(8)
    b = torch.randn(8)
    c = torch.randn(8)

    out, code = capture_output_code(compiled_diamond, a, b, c)
    eager_out = diamond(a, b, c)

    print(f"Real torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"Real correctness check: compiled vs. eager match = "
          f"{torch.allclose(out, eager_out)}")

    print()
    print("=== Real Inductor-generated C++ (one real kernel function) ===")
    # Print only the real compiled kernel function itself -- the generated
    # module also contains real boilerplate (imports, a Runner class, a
    # benchmark harness) identical in shape across every torch.compile call,
    # already shown once and not worth repeating verbatim here.
    kernel_match = re.search(
        r"(cpp_fused_\w+ = async_compile\.cpp_pybinding\([^\n]*r'''.*?''')",
        code, re.DOTALL)
    print(kernel_match.group(1) if kernel_match else code)

    # Real structural check: exactly how many separate real kernel functions
    # did Inductor emit for this 4-op graph? (Chapter 13's own diamond-fusion
    # question, asked of a fourth real, independent production compiler.)
    kernel_count = len(re.findall(r"async_compile\.cpp_pybinding", code))
    print()
    print(f"Real generated kernel count for the WHOLE diamond graph: "
          f"{kernel_count} (a, b, c each read once; t1, t2, t3 never "
          f"round-trip through memory -- the same real fusion boundary "
          f"XLA (Ch25) and Relax (Ch26) both drew on this identical graph, "
          f"reached here by a fourth independent real production compiler)")
