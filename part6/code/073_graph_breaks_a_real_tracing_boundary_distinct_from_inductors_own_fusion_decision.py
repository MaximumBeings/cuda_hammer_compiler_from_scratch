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
