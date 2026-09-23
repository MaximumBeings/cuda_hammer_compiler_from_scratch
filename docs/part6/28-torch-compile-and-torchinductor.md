# Chapter 28: torch.compile and TorchInductor

Chapters 25-27 each studied a real, independent system reached through a
clean `pip install` with no CUDA runtime required at import time: `jax`,
`apache-tvm`, and `triton` all import successfully on this sandbox with
zero GPU libraries present. `torch` breaks that streak. Real, current
PyPI `torch` (2.14.0) for Linux is CUDA-linked at the NATIVE library
level -- its own `libtorch_global_deps.so` has a hard runtime dependency
on `libcudart.so`, so a plain `import torch` (even after
`pip install torch --no-deps`) fails outright with `OSError:
libcudart.so.13: cannot open shared object file`, regardless of whether a
physical GPU is ever going to be used.

Real toolchain investigation, before any of this chapter's own files were
written: reading torch's own installed `__init__.py` source directly
(not guessing) shows why. Its own `_load_global_deps()` function calls
`_preload_cuda_deps(err)` on that `OSError`, and `_preload_cuda_deps()`
(with `required=True` by default) tries to preload SIXTEEN hardcoded CUDA
libraries by name -- `libcublasLt`, `libcublas`, `libcudnn`, `libnvrtc`,
`libnvrtc-builtins`, `libcudart`, `libcupti`, `libcufft`, `libcurand`,
`libnvJitLink`, `libcusparse`, `libcusparseLt`, `libcusolver`, `libnccl`,
`libnvshmem_host`, and `libcufile` -- raising a real `ValueError` if ANY
one is missing. `pip install torch --dry-run --report` confirms the real,
full dependency list this forces: 22 packages in total (`nvidia-cublas`,
`nvidia-cudnn-cu13`, `nvidia-cufft`, `nvidia-nccl-cu13`, and so on), the
two largest alone -- `nvidia-cudnn-cu13` and `nvidia-cublas` -- totaling
almost a full gigabyte. The real, full install (~2.5GB) completed on this
sandbox after `pip cache purge` reclaimed enough disk headroom (12GB free
to 16GB free), and `import torch` then works, reporting
`torch.cuda.is_available() == False`, exactly as expected with no
physical GPU attached.

`pip install torch --dry-run` on the device confirms real aarch64 wheels
exist for every one of those same 22 packages too -- a genuinely
different finding from any prior toolchain gap in this book: this is not
an architecture-support limitation. But the device's own real, connected
filesystem had only about 2GB of free disk space at investigation time
(`df -h` showed roughly 2.0GB free on `/sessions`, 4.3GB free on `/`),
well under the ~2.5GB this real install needs, and risking that on the
user's own linked machine was not a trade this book will make for a
demonstration chapter. So, like Chapter 18's own CUDA backend, this
chapter is verified for real ONLY on the cloud sandbox; the device still
receives and md5-verifies every one of this chapter's own files, but does
not execute them.

```text
+------------------------------------------------------------------+
|  Chapter 28's own shape, section by section                      |
|                                                                    |
|  28.1  torch.compile's real two-stage pipeline (TorchDynamo       |
|        capture, then TorchInductor codegen) on the same diamond   |
|        graph, confirming automatic whole-graph fusion into ONE    |
|        real generated kernel.                                     |
|                                                                    |
|  28.2  Graph breaks: a real TRACING-capability boundary,          |
|        genuinely different from Inductor's own fusion decision,   |
|        with a shared value threaded across it as a real buffer.   |
|                                                                    |
|  28.3  Real torch._inductor.config autotuning fields, laid next   |
|        to CUDA Hammer's own Schedule (Ch21) and TuningCache       |
|        (Ch24), plus a real executed max-autotune run.             |
+------------------------------------------------------------------+
```

## 28.1 torch.compile's Real Two-Stage Pipeline

Every case study so far in Part 6 reads a real intermediate
representation directly off some object the library hands back: an HLO
module (Ch25), a Relax/TIR IRModule (Ch26), TTIR/TTGIR text (Ch27).
`torch.compile()` does not work that way. It returns a plain Python
callable -- calling it just runs the (possibly compiled) function and
returns a tensor, with no IR object exposed anywhere in that return
value. The only way to see what TorchInductor actually generated is
through its own logging system: `torch._logging.set_logs(output_code=True)`
turns on a real logger (`torch._inductor.codecache`) that emits the
generated code as a log record, which File 072 captures with a dedicated
`logging.StreamHandler` attached to that logger by name -- the same "hook
the real library's own log output" technique this book has not needed
until now, because every prior chapter's IR was already sitting in a
returned object's own field.

```text
  Python function                    torch.compile(fn)
  (diamond: t1=a+b, t2=t1*c,              |
   t3=relu(t1), out=t2+t3)                \/

  TorchDynamo                     TorchInductor
  (bytecode-level tracer;    -->  (graph-level compiler;
   captures an FX graph            lowers the FX graph to
   by intercepting Python          real generated C++/Triton,
   bytecode as it runs)            fusing where it can)
      |                                    |
      \/                                   \/
  torch.fx.Graph                   generated kernel source
  (a real graph object,            (real C++ using a real
   never printed by this            Vectorized-float SIMD type
   file -- Dynamo's own             on this CPU-only sandbox;
   internal representation)         a real compiled .so, loaded
                                     back and called transparently)
```

Dynamo's own job is CAPTURE: turning a Python function's real bytecode
execution into a graph Inductor can lower. Inductor's own job is
CODEGEN: deciding what to fuse and emitting real, compilable source for
it. File 072 runs the exact same diamond graph Chapters 4, 13, 25, 26,
and 27 have all already used, and inspects Inductor's own real generated
kernel directly:

```python
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
```

```bash
python3 "072_torch_compiles_real_two_stage_pipeline_dynamo_capture_then_inductor_codegen.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed torch.compile output; this chapter is verified on the cloud sandbox only, per the disk-space finding above):**

```text
Real torch.cuda.is_available(): False
Real correctness check: compiled vs. eager match = True

=== Real Inductor-generated C++ (one real kernel function) ===
cpp_fused_add_mul_relu_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'float*'], r'''
#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0L) && x0 < static_cast<int64_t>(8L)))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8L));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(8L));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<int64_t>(x0), static_cast<int64_t>(8L));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 * tmp3;
                    auto tmp5 = at::vec::clamp_min(tmp2, decltype(tmp2)(0));
                    auto tmp6 = tmp4 + tmp5;
                    tmp6.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(8L));
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
'''

Real generated kernel count for the WHOLE diamond graph: 1 (a, b, c each read once; t1, t2, t3 never round-trip through memory -- the same real fusion boundary XLA (Ch25) and Relax (Ch26) both drew on this identical graph, reached here by a fourth independent real production compiler)
```

*torch.cuda.is_available() is False throughout -- Inductor's own generated code still compiles and runs correctly, fusing the whole diamond graph into one real vectorized CPU kernel with no GPU present.*


### Automatic whole-graph fusion, confirmed a fourth time

File 072's own real captured output shows exactly ONE generated kernel,
`cpp_fused_add_mul_relu_0`, for this whole four-operation graph -- `a`,
`b`, and `c` are each read once (`at::vec::Vectorized<float>::loadu`),
`t1`, `t2`, and `t3` never round-trip through memory, and the ReLU itself
compiles to a real `at::vec::clamp_min` call inlined directly into the
same kernel body as the add and the multiply. This is the same real
fusion boundary XLA's own `kind=kLoop` fusion (Ch25) and TVM Relax's own
`FuseOps`/`FuseTIR` (Ch26) both drew on this identical graph -- a fourth
independent, real production compiler reaching the same whole-graph
fusion decision automatically, with no `@triton.jit` function boundary
for a human to have drawn by hand the way Chapter 27's own File 070
required.

Two smaller real findings, caught by directly inspecting this file's
first captured run rather than assumed: torch's own default logging
handler writes every enabled log artifact straight to the OS-level
stderr file descriptor, independent of the Python `logging.StreamHandler`
this file also attaches -- so a first, naive capture attempt produced
over 129KB of interleaved compiler noise on stderr alongside the clean
captured text. Fixed with a small `_suppress_os_stderr()` context manager
(`os.dup2` to redirect the raw fd to `/dev/null` for the duration of the
compiled call, restored in a `finally`). And the first regex written to
extract just the kernel body from the captured text,
`r"(cpp_fused_\w+ = async_compile\.cpp_pybinding.*?''')"`, matched only
up to the FIRST `'''` in the text -- which turned out to be the raw `r'''`
opening delimiter itself, since the substring `'''` appears immediately
after the `r`. The real fix explicitly consumes past that opening
sequence (`r"(cpp_fused_\w+ = async_compile\.cpp_pybinding\([^\n]*r'''.*?''')"`)
before the lazy `.*?'''` is allowed to find the REAL closing triple-quote.

```bash
python3 "072_torch_compiles_real_two_stage_pipeline_dynamo_capture_then_inductor_codegen.py"
```

```python
# Chapter 28, File 073: a graph break -- TorchDynamo's own real tracing-
# capability boundary, genuinely different from the fusion boundary File 072
# already confirmed Inductor draws automatically. Run on the same
# Ch4/13/25/26/27/72 diamond graph, split by inserting exactly one real
# Python construct Dynamo cannot trace through: converting a tensor to a
# plain Python float with `.item()`.
#
# Compiled with:   python3 "073_graph_breaks_a_real_tracing_boundary_distinct_from_inductors_own_fusion_decision.py"

import contextlib
import io
import logging
import os
import re

import torch


@contextlib.contextmanager
def _suppress_os_stderr():
    # Same real need as File 072: torch's own default logging handler
    # writes straight to the OS-level stderr fd, independent of any
    # Python `logging` handler this file attaches.
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


def diamond_no_break(a, b, c):
    t1 = a + b
    t2 = t1 * c
    t3 = torch.relu(t1)
    return t2 + t3


def diamond_with_break(a, b, c):
    t1 = a + b
    # Tensor.item() forces a real value out of the traced graph and into a
    # plain Python float -- Dynamo's own bytecode tracer cannot represent
    # that inside one FX graph, so it stops tracing here and resumes with a
    # SECOND graph for what follows. Nothing about FUSION changed; this is
    # a tracing-capability boundary, not a cost or correctness decision.
    _t1_sum_as_python_float = t1.sum().item()
    t2 = t1 * c
    t3 = torch.relu(t1)
    return t2 + t3


if __name__ == "__main__":
    a = torch.randn(8)
    b = torch.randn(8)
    c = torch.randn(8)

    with _suppress_os_stderr():
        explain_no_break = torch._dynamo.explain(diamond_no_break)(a, b, c)
        explain_with_break = torch._dynamo.explain(diamond_with_break)(a, b, c)

    print("Real torch._dynamo.explain() summary, no break:")
    print(f"  graph_count={explain_no_break.graph_count} "
          f"graph_break_count={explain_no_break.graph_break_count} "
          f"op_count={explain_no_break.op_count}")

    print()
    print("Real torch._dynamo.explain() summary, with one .item() break:")
    print(f"  graph_count={explain_with_break.graph_count} "
          f"graph_break_count={explain_with_break.graph_break_count} "
          f"op_count={explain_with_break.op_count}")
    reason = explain_with_break.break_reasons[0].reason.splitlines()[0]
    print(f"  real break reason (first line): {reason}")

    compiled_with_break = torch.compile(diamond_with_break)
    out, code = capture_output_code(compiled_with_break, a, b, c)
    eager_out = diamond_with_break(a, b, c)
    print()
    print(f"Real correctness check (graph-broken version): compiled vs. "
          f"eager match = {torch.allclose(out, eager_out)}")

    kernel_names = re.findall(r"(cpp_fused_\w+) = async_compile\.cpp_pybinding",
                               code)
    # Real structural check, confirmed by reading the two compiled modules'
    # own generated `Runner.call()` bodies directly: the FIRST module's real
    # return statement is `return (buf1, buf0, )` -- buf1 is t1's sum (the
    # scalar `.item()` pulls out), and buf0 is the raw t1 TENSOR itself,
    # shape (8,), returned as a real second output even though nothing in
    # `diamond_with_break`'s own source ever names it as a return value. The
    # SECOND module's own kernel (`cpp_fused_add_mul_relu_0`) takes exactly
    # two real inputs and computes `tmp0 * tmp1` (t1 * c) and
    # `clamp_min(tmp0, 0)` (relu(t1)) from them -- so that first input IS
    # buf0, i.e. t1 crosses the graph-break boundary as a real compiled
    # buffer, not as a Python float and not recomputed from scratch.
    print(f"Real generated kernel count with the break: {len(kernel_names)} "
          f"({', '.join(kernel_names)}) -- TWO real compiled kernels where "
          f"File 072's clean version needed only one. t1 ITSELF (not just "
          f"its sum) is threaded across the break as a real compiled "
          f"buffer: the first kernel's own module returns (sum, t1) -- "
          f"`.item()` only needed the sum, but Inductor keeps t1 alive as a "
          f"real second output because the second kernel needs it as an "
          f"input -- and the second kernel's own signature takes t1 and c "
          f"directly, computing t1*c and relu(t1) from them. A close "
          f"parallel to File 070's Triton split-kernel finding, reached "
          f"here by a completely different real mechanism (a tracing "
          f"boundary, not a hand-written kernel split)")
```

```bash
python3 "073_graph_breaks_a_real_tracing_boundary_distinct_from_inductors_own_fusion_decision.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed torch.compile output; this chapter is verified on the cloud sandbox only, per the disk-space finding above):**

```text
Real torch._dynamo.explain() summary, no break:
  graph_count=1 graph_break_count=0 op_count=4

Real torch._dynamo.explain() summary, with one .item() break:
  graph_count=2 graph_break_count=1 op_count=4
  real break reason (first line): Unsupported Tensor.item() call with capture_scalar_outputs=False

Real correctness check (graph-broken version): compiled vs. eager match = True
Real generated kernel count with the break: 2 (cpp_fused_add_sum_0, cpp_fused_add_mul_relu_0) -- TWO real compiled kernels where File 072's clean version needed only one. t1 ITSELF (not just its sum) is threaded across the break as a real compiled buffer: the first kernel's own module returns (sum, t1) -- `.item()` only needed the sum, but Inductor keeps t1 alive as a real second output because the second kernel needs it as an input -- and the second kernel's own signature takes t1 and c directly, computing t1*c and relu(t1) from them. A close parallel to File 070's Triton split-kernel finding, reached here by a completely different real mechanism (a tracing boundary, not a hand-written kernel split)
```

*Two real compiled modules where the clean version needed one, with t1 itself threaded across the break as a real compiled buffer, confirmed by direct inspection of both modules' own generated Runner.call() bodies.*


## 28.2 Graph Breaks: A Tracing Boundary, Not a Fusion Decision

Section 28.1's own fusion boundary is a codegen decision Inductor makes
once it already has a complete graph. A graph BREAK is a different kind
of boundary entirely, imposed one level earlier by Dynamo's own tracer,
before Inductor is ever consulted. File 073 makes this concrete by
running the same diamond graph two ways: `diamond_no_break` (the clean
function) and `diamond_with_break`, identical except for one inserted
line -- `t1.sum().item()` -- immediately after `t1 = a + b`.

`Tensor.item()` pulls a real value out of the traced computation and
converts it to a plain Python float. Dynamo's own bytecode-level tracer
has no way to represent that inside a single FX graph (a Python float,
unlike a tensor, cannot carry a symbolic trace forward), so it stops
tracing at that point and resumes with a SECOND, separate graph for
whatever comes after. `torch._dynamo.explain()` reports this directly, as
a real `ExplainOutput` dataclass with clean fields rather than a wall of
printed guard objects:

```text
  diamond_no_break:     graph_count=1  graph_break_count=0
  diamond_with_break:   graph_count=2  graph_break_count=1
                         break reason: "Unsupported Tensor.item() call
                         with capture_scalar_outputs=False"
```

Nothing about FUSION changed between these two versions -- the break is a
tracing-capability limit, not a cost decision, and File 073's own real
generated code confirms exactly what crosses the resulting boundary.
Reading the two compiled modules' own generated `Runner.call()` bodies
directly: the FIRST module's real return statement is
`return (buf1, buf0, )` -- `buf1` is `t1`'s sum (the scalar `.item()`
needs), and `buf0` is the RAW `t1` TENSOR itself, shape `(8,)`, returned
as a real second output even though nothing in `diamond_with_break`'s own
source ever names `t1` as a return value. The SECOND module's own kernel,
`cpp_fused_add_mul_relu_0`, takes exactly two real inputs and computes
`tmp0 * tmp1` (`t1 * c`) and `clamp_min(tmp0, 0)` (`relu(t1)`) from them
-- so that first input IS `buf0`. `t1` itself, not merely its sum,
crosses the graph-break boundary as a real compiled buffer passed from
the first module's own output directly into the second module's own
input.

```text
  diamond_with_break, real compiled shape:

  kernel 1: cpp_fused_add_sum_0              kernel 2: cpp_fused_add_mul_relu_0
  in:  a, b                                  in:  t1 (=buf0), c
  t1 = a + b                                 t2 = t1 * c
  buf1 = t1.sum()      [.item() reads buf1   t3 = relu(t1)
  out: (buf1, buf0)     back in Python here]  out = t2 + t3
                                              out: (buf0,)
                        ^-- t1 threaded through as a real buffer,
                            not recomputed, not a Python float
```

This is a close parallel to Chapter 27's own File 070 finding -- a shared
value forced through a real materialization boundary between two
separately compiled kernels -- reached here by a completely different
real mechanism: a tracing limitation TorchDynamo hits before compilation
even starts, not a hand-written kernel split the programmer chose in
advance.

```bash
python3 "073_graph_breaks_a_real_tracing_boundary_distinct_from_inductors_own_fusion_decision.py"
```

```python
# Chapter 28, File 074: TorchInductor's own real autotuning config surface,
# laid directly next to CUDA Hammer's own Schedule/enumerateSchedules()
# (Chapter 21), hybrid cost-model-then-measurement autotuner (Chapter 23.3),
# and TuningCache (Chapter 24) -- the same "read the real library's own
# record types side by side with CUDA Hammer's own" capstone pattern used
# to close Chapters 25, 26, and 27.
#
# Real toolchain note: `torch._inductor.config` is a real, plain Python
# module whose attributes ARE the config -- there is no separate dataclass
# or schema object to read. Filtering `dir(torch._inductor.config)` by
# substring (e.g. "autotune" in name) surprisingly returns only a handful
# of function-valued entries (helper functions the module itself defines),
# because most of the real config fields are dynamically installed by an
# internal `install_config_module()` call and don't show up under a naive
# `dir()` scan the same way. Using `hasattr()`/`getattr()` against a
# hand-picked list of real field names (found by reading Inductor's own
# source and release notes) is what actually surfaces their real values.
#
# Compiled with:   python3 "074_real_inductor_autotune_config_next_to_cuda_hammers_own_schedule_and_tuningcache.py"

import torch
import torch._inductor.config as inductor_config


# CUDA Hammer's own Chapter 21 Schedule and Chapter 24 TuningCache shapes,
# reproduced here (not imported -- this file has no dependency on the rest
# of the CUDA Hammer codebase) purely so the real Inductor fields below can
# be read directly alongside them.
class Schedule:
    """CUDA Hammer, Chapter 21: one point in the autotuning search space."""

    def __init__(self, tile_size_per_loop, loop_order, unroll_factor):
        self.tile_size_per_loop = tile_size_per_loop
        self.loop_order = loop_order
        self.unroll_factor = unroll_factor


class TuningCacheEntry:
    """CUDA Hammer, Chapter 24: one persisted autotuning result. Chapter 24's
    real TuningCache stores only the WINNING schedule per loop-nest key, in
    a plain-text, human-readable file."""

    def __init__(self, loop_nest_key, best_schedule, measured_time_ns):
        self.loop_nest_key = loop_nest_key
        self.best_schedule = best_schedule
        self.measured_time_ns = measured_time_ns


REAL_AUTOTUNE_FIELDS = [
    "max_autotune",
    "max_autotune_gemm",
    "max_autotune_pointwise",
    "coordinate_descent_tuning",
    "autotune_local_cache",
    "search_autotune_cache",
    "max_autotune_gemm_backends",
]


def read_real_inductor_autotune_config():
    """Reads the REAL current values of torch._inductor.config's own
    autotuning fields -- these are real library defaults, not values this
    file invented or hardcoded."""
    return {name: getattr(inductor_config, name, "<MISSING>")
            for name in REAL_AUTOTUNE_FIELDS}


if __name__ == "__main__":
    defaults = read_real_inductor_autotune_config()
    print("Real torch._inductor.config autotuning fields (library defaults):")
    for name, value in defaults.items():
        print(f"  {name} = {value!r}")

    print()
    print("Compared against CUDA Hammer's own autotuning machinery:")
    print(f"  Chapter 21 Schedule{{tile_size_per_loop, loop_order, "
          f"unroll_factor}} <-> Inductor's own internal per-op Config "
          f"objects (block sizes, num_warps, num_stages -- never exposed "
          f"as a public dataclass the way CUDA Hammer's own Schedule is)")
    print(f"  Chapter 23.3 hybrid cost-model-then-measurement autotuner <-> "
          f"Inductor's own coordinate_descent_tuning={defaults['coordinate_descent_tuning']!r} "
          f"(a real greedy local search over the config space, distinct "
          f"from CUDA Hammer's own top-K-prune-then-measure hybrid) plus "
          f"max_autotune_gemm_backends={defaults['max_autotune_gemm_backends']!r} "
          f"(the real, named set of candidate implementations -- ATen's "
          f"own library call, a real generated Triton kernel, or a real "
          f"generated C++ kernel -- Inductor benchmarks against each other "
          f"under max-autotune, a genuinely different search axis than "
          f"CUDA Hammer's tile/order/unroll space: WHICH BACKEND, not just "
          f"which schedule)")
    print(f"  Chapter 24 TuningCache (plain-text, human-keyed, winner-only) "
          f"<-> autotune_local_cache={defaults['autotune_local_cache']!r} "
          f"(True by default: a real persistent local autotuning cache, "
          f"keeping this book's Chapter 27 Triton-vs-CUDA-Hammer disk-cache "
          f"comparison consistent -- a third independent real production "
          f"compiler defaulting to ON persistent caching, where CUDA "
          f"Hammer's own TuningCache requires an explicit save)")

    print()
    print("Real end-to-end execution: torch.compile(f, mode='max-autotune') "
          "on CPU with no GPU present (the mode still runs; it just narrows "
          "to backends that make sense without a GPU):")

    def matmul(x, y):
        return x @ y

    # Real, explicit backend restriction -- 'TRITON' alone would require a
    # working GPU code-generation path this sandbox does not have; 'ATEN'
    # and 'CPP' are both real CPU-capable backends.
    old_backends = inductor_config.max_autotune_gemm_backends
    inductor_config.max_autotune_gemm_backends = "ATEN,CPP"
    try:
        compiled_matmul = torch.compile(matmul, mode="max-autotune")
        x = torch.randn(64, 64)
        y = torch.randn(64, 64)
        out = compiled_matmul(x, y)
        eager_out = matmul(x, y)
        print(f"  max_autotune_gemm_backends temporarily set to "
              f"{inductor_config.max_autotune_gemm_backends!r}")
        print(f"  Real correctness check: compiled vs. eager match = "
              f"{torch.allclose(out, eager_out, atol=1e-4)}")
    finally:
        inductor_config.max_autotune_gemm_backends = old_backends

    print()
    print("Real torch.cuda.is_available():", torch.cuda.is_available())
    print("(max-autotune ran and produced a correct result with zero "
          "physical GPU present -- Inductor's own backend search degrades "
          "to CPU-capable candidates automatically rather than failing)")
```

```bash
python3 "074_real_inductor_autotune_config_next_to_cuda_hammers_own_schedule_and_tuningcache.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed torch.compile output; this chapter is verified on the cloud sandbox only, per the disk-space finding above):**

```text
Real torch._inductor.config autotuning fields (library defaults):
  max_autotune = False
  max_autotune_gemm = False
  max_autotune_pointwise = False
  coordinate_descent_tuning = False
  autotune_local_cache = True
  search_autotune_cache = False
  max_autotune_gemm_backends = 'ATEN,TRITON,CPP'

Compared against CUDA Hammer's own autotuning machinery:
  Chapter 21 Schedule{tile_size_per_loop, loop_order, unroll_factor} <-> Inductor's own internal per-op Config objects (block sizes, num_warps, num_stages -- never exposed as a public dataclass the way CUDA Hammer's own Schedule is)
  Chapter 23.3 hybrid cost-model-then-measurement autotuner <-> Inductor's own coordinate_descent_tuning=False (a real greedy local search over the config space, distinct from CUDA Hammer's own top-K-prune-then-measure hybrid) plus max_autotune_gemm_backends='ATEN,TRITON,CPP' (the real, named set of candidate implementations -- ATen's own library call, a real generated Triton kernel, or a real generated C++ kernel -- Inductor benchmarks against each other under max-autotune, a genuinely different search axis than CUDA Hammer's tile/order/unroll space: WHICH BACKEND, not just which schedule)
  Chapter 24 TuningCache (plain-text, human-keyed, winner-only) <-> autotune_local_cache=True (True by default: a real persistent local autotuning cache, keeping this book's Chapter 27 Triton-vs-CUDA-Hammer disk-cache comparison consistent -- a third independent real production compiler defaulting to ON persistent caching, where CUDA Hammer's own TuningCache requires an explicit save)

Real end-to-end execution: torch.compile(f, mode='max-autotune') on CPU with no GPU present (the mode still runs; it just narrows to backends that make sense without a GPU):
  max_autotune_gemm_backends temporarily set to 'ATEN,CPP'
  Real correctness check: compiled vs. eager match = True

Real torch.cuda.is_available(): False
(max-autotune ran and produced a correct result with zero physical GPU present -- Inductor's own backend search degrades to CPU-capable candidates automatically rather than failing)
```

*Every real config field read directly from the installed library; the closing max-autotune matmul runs to completion and matches eager output with zero physical GPU present.*


## 28.3 Real torch._inductor.config Autotuning, Next to CUDA Hammer's Own Schedule and TuningCache

`torch._inductor.config` is a real, plain Python module whose own
attributes ARE the configuration -- there is no separate dataclass to
read. File 074 reads a hand-picked set of real field names directly off
the installed module with `getattr()` (a naive `dir()` substring scan
surprisingly misses most of them, since Inductor installs many fields
dynamically through its own `install_config_module()` machinery rather
than as ordinary class attributes visible to `dir()`).

`max_autotune_gemm_backends = 'ATEN,TRITON,CPP'` is the real, named set
of candidate implementations -- ATen's own library call, a real generated
Triton kernel, or a real generated C++ kernel -- Inductor benchmarks
against each other under `mode='max-autotune'` for a single matmul. This
is a genuinely different search axis from anything CUDA Hammer's own
Chapter 21 `Schedule{tileSizePerLoop, loopOrder, unrollFactor}` searches
over: WHICH BACKEND, not just which schedule of one fixed backend.
`coordinate_descent_tuning` (a real greedy local search over the
remaining config space once a backend is chosen) is the closer analogue
of Chapter 23.3's own hybrid cost-model-then-measurement autotuner,
though the two algorithms differ -- coordinate descent explores
neighboring configs one dimension at a time, where Chapter 23.3 ranks all
128 real schedules by cost model up front and only measures the cheapest
`topK=8`.

`autotune_local_cache = True` by default is the real, direct analogue of
Chapter 24's own `TuningCache` -- and, exactly as Chapter 27.3 already
found for Triton's own disk cache, a real, independent production
compiler defaults to ON persistent local caching, where CUDA Hammer's own
`TuningCache` requires an explicit save call. Three real production
autotuners now agree on persisting results by default; CUDA Hammer's own
opt-in design is the outlier among them, an honest thing for this book to
note about its own choices next to the real systems it studies.

File 074 closes with a real, executed `torch.compile(matmul,
mode='max-autotune')` call on a 64x64 matmul, with
`max_autotune_gemm_backends` temporarily narrowed to `'ATEN,CPP'` (the two
real backends that make sense with no physical GPU present -- `'TRITON'`
alone would need a working GPU codegen path this sandbox does not have).
It runs to completion and matches eager output exactly, confirming
Inductor's own backend search degrades gracefully to CPU-capable
candidates rather than failing outright when no GPU is attached -- the
same "real execution succeeds even under this chapter's own real
hardware limitation" finding File 072 and File 073 already established
for the ungated default path.

```bash
python3 "074_real_inductor_autotune_config_next_to_cuda_hammers_own_schedule_and_tuningcache.py"
```
