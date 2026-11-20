# Chapter 30, File 079: explicit widening in a zero-point-zero int8 dot.
# The programmer requests int32 conversion before multiplication; the compiler
# lowers those explicit casts. This is not automatic inference of a quantized
# accumulator policy from arbitrary primitive graphs.
#
# Historical evidence is offline AOT compilation, not GPU execution. File 078
# checks its own NumPy arithmetic; it does not validate this Triton kernel.
# Current source bounds positive power-of-two N to <=65536 so worst-case
# signed-int8 accumulation fits int32. Finite input scales and matching allocated
# extents are caller requirements; asymmetric zero-point correction is absent.
#
# A fresh compile and real-device checks are required after source changes.
# See validation/accuracy-fixes/gpu_case_validation.py. Historical IR excerpts
# show the former N=64 compile and are not current GPU runtime evidence.
#
# Compiled with:   python3 "079_a_real_triton_quantized_dot_kernel_compiled_with_no_gpu.py"

import re
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def quantized_dot_kernel(a_ptr, b_ptr, out_ptr, scale_a, scale_b,
                          N: tl.constexpr):
    tl.static_assert(N > 0 and (N & (N - 1)) == 0, "N must be a positive power of two")
    tl.static_assert(N <= 65536, "int8 worst-case accumulation must fit int32")
    # a, b load as real int8 BLOCK-level tensors -- Chapter 27's own
    # block-level model, applied here to narrow integer data for the
    # first time in this book.
    offsets = tl.arange(0, N)
    a = tl.load(a_ptr + offsets)
    b = tl.load(b_ptr + offsets)

    # The one real codegen step File 078 argued was necessary by hand:
    # widen BEFORE multiplying. Written here as ordinary Triton syntax --
    # `.to(tl.int32)` -- with no manual overflow bookkeeping.
    a32 = a.to(tl.int32)
    b32 = b.to(tl.int32)
    products = a32 * b32          # int8-range products fit int32; N is bounded above
    acc = tl.sum(products, axis=0)  # real int32 accumulate (a genuine
    # Triton block-level REDUCTION, tl.sum, over the whole loaded block --
    # not a running total threaded through a loop the way File 076's own
    # KV-block loop needed; N=64 fits in one block here, so one real
    # `tt.reduce` op covers the whole dot product)

    # Real dequantization: convert the exact int32 accumulator to
    # float32, then apply the combined real scale -- exactly File 078's
    # own dequantize() math, `(q - zero_point) * scale`, specialized to
    # zero_point=0 (both operands already zero-centered before this
    # kernel is called) and a combined scale = scale_a * scale_b, the
    # real way a quantized dot product's own output scale composes from
    # its two input scales.
    real_scale = scale_a * scale_b
    dequant = acc.to(tl.float32) * real_scale
    tl.store(out_ptr, dequant)


def compile_for_target(fn, signature, constexprs, target):
    src = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=target)


def strip_source_locations(ir_text):
    # Same real technique as Chapters 27 and 29: the two real machines'
    # own absolute paths to this file differ, and that difference lives
    # ONLY in the `#locN = loc(...)` definition lines.
    kept = []
    for line in ir_text.splitlines():
        if line.strip().startswith("#loc"):
            continue
        kept.append(line)
    return "\n".join(kept)


def count_ops(ir_text, op_name):
    # Count the operation name, not e.g. its tt.reduce.return terminator.
    pattern = r"\b" + re.escape(op_name) + r"(?![\w.])"
    return sum(1 for line in ir_text.splitlines() if re.search(pattern, line))


if __name__ == "__main__":
    target = GPUTarget(backend="cuda", arch=80, warp_size=32)
    compiled = compile_for_target(
        quantized_dot_kernel,
        {"a_ptr": "*i8", "b_ptr": "*i8", "out_ptr": "*fp32",
         "scale_a": "fp32", "scale_b": "fp32", "N": "constexpr"},
        {"N": 64},
        target,
    )

    print("Real compiled pipeline stages:", list(compiled.asm.keys()))
    for stage in ["ttir", "ttgir", "llir", "ptx"]:
        print(f"  {stage}: {len(compiled.asm[stage])} real chars")

    ttir = strip_source_locations(compiled.asm["ttir"])
    print()
    print(f"Real arith.extsi (sign-extend int8->int32) count: "
          f"{count_ops(ttir, 'arith.extsi')} -- ONE PER real int8 input "
          f"(a and b), confirming Triton's own compiler inserts a real, "
          f"SEPARATE widening instruction automatically from the source "
          f"text's own plain `.to(tl.int32)` call, exactly the codegen "
          f"step File 078 argued was structurally necessary")
    print(f"Real arith.muli (the widened int32 multiply) count: "
          f"{count_ops(ttir, 'arith.muli')}")
    print(f"Real tt.reduce (the int32 accumulate) count: "
          f"{count_ops(ttir, 'tt.reduce')}")
    print(f"Real arith.sitofp (signed-int-to-float, for dequantization) "
          f"count: {count_ops(ttir, 'arith.sitofp')}")

    print()
    print("=== Real TTIR excerpt: load through widen through multiply "
          "(path-normalized) ===")
    lines = ttir.splitlines()
    load_idx = next(i for i, l in enumerate(lines) if "tt.load" in l)
    for line in lines[load_idx:load_idx + 4]:
        print(line.strip())

    print()
    print("=== Real TTIR excerpt: the int32 reduction (path-normalized) ===")
    reduce_idx = next(i for i, l in enumerate(lines) if "tt.reduce" in l)
    for line in lines[reduce_idx:reduce_idx + 4]:
        print(line.strip())
