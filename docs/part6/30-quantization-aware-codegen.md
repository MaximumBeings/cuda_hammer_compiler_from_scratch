# Chapter 30: Quantization-Aware Codegen

Every op this book's own IR has ever generated code for, since Chapter
4's own `Value`, has been float-in/float-out, with no change of bit
width between an operation's inputs and its output. Quantization breaks
that assumption directly: a quantized op reads narrow integers (commonly
8-bit) and, if its own generated code is not careful about WHICH type it
accumulates in, the real arithmetic silently overflows. This closing
Part 6 chapter studies that one codegen requirement -- an explicit
accumulator-width change -- across the same real systems this Part has
already installed: pure NumPy (no new toolchain), Triton (Chapter 27's
own `target=` AOT-compile technique), and real PyTorch (Chapter 28's own
installed `torch`).

```text
+------------------------------------------------------------------+
|  Chapter 30's own shape, section by section                       |
|                                                                    |
|  30.1  The quantization math itself, in pure NumPy -- affine      |
|        scale+zero-point quantization, and a real, directly        |
|        observed overflow demonstrating why an int8 accumulator    |
|        is not just imprecise but actually WRONG.                  |
|                                                                    |
|  30.2  The same accumulate-then-requantize shape as one real      |
|        Triton kernel, compiled with no GPU -- inspecting the      |
|        real compiler-inserted widening instruction directly.      |
|                                                                    |
|  30.3  Real PyTorch's own quantized GEMM kernels, and Part 6's    |
|        own closing synthesis: does automatic fusion ever          |
|        discover this shape, or does it dispatch to a real,        |
|        hand-written kernel every time?                            |
+------------------------------------------------------------------+
```

## 30.1 Affine Quantization, and Why Codegen Needs a Wider Accumulator

Affine (linear) quantization maps a real-valued array onto narrow
integers with two real numbers: a `scale` (the real value one integer
step represents) and a `zero_point` (which integer represents 0.0).
Quantizing rounds and clips; dequantizing recovers an approximation,
bounded by `scale/2` in the worst case -- this is ordinary, bounded
imprecision, the same kind of rounding error Chapter 21's own float32
non-associativity finding already introduced this book to. The real
danger is not imprecision; it's what happens during the ARITHMETIC a
quantized op has to do before it can even produce a dequantizable
result:

```text
  Quantized dot product of two real int8 arrays, a and b:

  wrong (int8 accumulator, same width in and out -- every OTHER op
  this book has ever generated code for uses this shape):

    acc: int8 = 0
    for i in range(N):
        acc = acc + (a[i] * b[i])   -- int8 * int8 OVERFLOWS SILENTLY,
                                        and the += ALSO overflows

  right (int32 accumulator -- WIDER than either input's own type):

    acc: int32 = 0
    for i in range(N):
        acc = acc + (int32(a[i]) * int32(b[i]))   -- both the multiply
                                                       AND the running
                                                       sum stay exact
```

File 078 makes the "wrong" version concrete rather than hypothetical:
real random int8 data, multiplied and accumulated at int8 width using
real NumPy arithmetic, silently wraps to a value nowhere close to the
true sum -- with no exception and, at the array level, no warning
either.

```python
# Chapter 30, File 078 (closes Part 6): affine (scale + zero-point)
# quantization's own real math, and the real reason quantization-aware
# CODEGEN is a genuinely different problem from anything Part 4's own
# CUDA Hammer codegen (Ch17-20) ever had to solve -- every earlier CUDA
# Hammer op was float-in/float-out, with no change of bit width between
# an operation's inputs and its output. A quantized op is not: it reads
# narrow (commonly 8-bit) integers, and if its own generated code
# accumulates their products in that SAME narrow width, the real
# arithmetic silently overflows.
#
# No new install for this file -- pure NumPy, already on both of this
# book's own machines, the same "reuse what Part 6 already installed"
# discipline Chapter 29 established.
#
# Compiled with:   python3 "078_affine_quantization_math_and_why_codegen_needs_a_wider_accumulator.py"

import warnings

import numpy as np


def quantize(x, scale, zero_point, qmin=-128, qmax=127):
    """Real affine (linear) quantization: map a real-valued array to
    narrow integers, given a real scale (float, quantization step size)
    and zero_point (integer, which quantized value represents 0.0)."""
    q = np.round(x / scale) + zero_point
    q = np.clip(q, qmin, qmax)
    return q.astype(np.int8)


def dequantize(q, scale, zero_point):
    """The inverse: recover an approximate real value from a quantized
    integer. Exact only when the original real value landed exactly on
    a representable quantization step -- otherwise this is where the
    real, bounded rounding error (at most scale/2) comes from."""
    return (q.astype(np.float64) - zero_point) * scale


def choose_scale_zero_point(x, qmin=-128, qmax=127):
    """A real, minimal calibration routine: pick scale/zero_point so the
    real observed [min(x), max(x)] range maps exactly onto
    [qmin, qmax] -- the same affine-mapping idea every real quantization
    library (including PyTorch's own, inspected in File 080) calibrates
    from real observed activation/weight ranges."""
    x_min, x_max = float(x.min()), float(x.max())
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - round(x_min / scale)
    zero_point = int(np.clip(zero_point, qmin, qmax))
    return scale, zero_point


if __name__ == "__main__":
    rng = np.random.default_rng(seed=30)

    print("=== Part 1: real round-trip error, calibrated from real data ===")
    x = rng.normal(loc=0.0, scale=2.0, size=1000).astype(np.float64)
    scale, zero_point = choose_scale_zero_point(x)
    q = quantize(x, scale, zero_point)
    x_recovered = dequantize(q, scale, zero_point)

    abs_error = np.abs(x - x_recovered)
    print(f"Calibrated from 1000 real samples: scale={scale:.6f}, "
          f"zero_point={zero_point}")
    print(f"Real max abs round-trip error: {abs_error.max():.6f} "
          f"(theoretical bound scale/2 = {scale / 2:.6f})")
    print(f"Real mean abs round-trip error: {abs_error.mean():.6f}")
    print(f"Quantized int8 range actually used: "
          f"[{q.min()}, {q.max()}] out of the legal [-128, 127]")

    print()
    print("=== Part 2: the real reason quantization-aware codegen is a "
          "genuinely new problem -- accumulator width ===")
    a = rng.integers(-127, 127, size=64, dtype=np.int8)
    b = rng.integers(-127, 127, size=64, dtype=np.int8)

    # The WRONG way -- exactly what naively reusing Chapter 17's own
    # emitSteps() scalar accumulation pattern (a same-width running sum)
    # would generate for a quantized op, with NO type change from input
    # to accumulator. Computed at the ARRAY level deliberately: real
    # numpy's own int8 array*array multiply wraps SILENTLY (no
    # exception, no warning -- confirmed separately below, where the
    # SAME overflow on a single scalar pair DOES raise a real
    # RuntimeWarning, an honest, real inconsistency in numpy's own
    # overflow reporting between vectorized and scalar integer ops, not
    # a hypothetical worth glossing over).
    wrong_products = a * b  # real numpy int8 * int8, silently wrapped
    wrong_sum = np.int8(wrong_products.astype(np.int32).sum())  # also
    # re-truncated to int8, matching what an int8 ACCUMULATOR would
    # really hold after summing already-wrapped int8 products

    # The RIGHT way -- CAST TO int32 BEFORE multiplying, exactly the one
    # real codegen change this chapter's own Files 079/080 both confirm
    # real systems make: widen the accumulator's own TYPE, not just its
    # value.
    correct_products = a.astype(np.int32) * b.astype(np.int32)
    correct_sum = correct_products.sum()

    overflowed_terms = int(np.sum(
        wrong_products.astype(np.int32) != correct_products))

    print(f"64 real int8 x int8 products, accumulated the WRONG way "
          f"(int8 accumulator, matching every OTHER op this book has "
          f"ever generated code for -- same width in, same width out):")
    print(f"  {overflowed_terms} of 64 individual products already "
          f"overflow int8 on their own (magnitude > 127) before any "
          f"summation even happens")
    print(f"  final int8-accumulated sum: {int(wrong_sum)} (silently "
          f"wrapped, no exception, no warning at the array level -- "
          f"real numpy {np.__version__} behavior)")
    print(f"  final real (int32) sum:     {int(correct_sum)}")
    print(f"  WRONG by: {int(correct_sum) - int(wrong_sum)}")

    print()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scalar_result = a[0] * b[0]  # the SAME real overflow, but on a
        # single scalar element instead of a whole array
        warning_text = str(caught[0].message) if caught else "(none)"
    print(f"Real aside: the identical overflow on a single SCALAR "
          f"int8 pair ({int(a[0])} * {int(b[0])} -> wrapped to "
          f"{int(scalar_result)}) DOES raise a real numpy RuntimeWarning "
          f"(\"{warning_text}\"), even though the array-level version "
          f"just computed above raises nothing -- numpy's own overflow "
          f"reporting is genuinely inconsistent between vectorized and "
          f"scalar integer arithmetic, an honest quirk worth knowing "
          f"before trusting numpy's silence as proof of correctness.")
    print()
    print("This is the one real, structural codegen change quantization "
          "forces: Part 4's own emitSteps()/generateCudaElementwiseKernel() "
          "never had to distinguish an operation's OWN type from its "
          "ACCUMULATOR's type, because every prior CUDA Hammer op used "
          "the same float type for both. A real quantized-matmul kernel "
          "generator has to emit code where the loop's own running sum "
          "is declared int32 even though every value it reads is int8 -- "
          "a genuinely new codegen requirement, not just a new op kind.")
```

```bash
python3 "078_affine_quantization_math_and_why_codegen_needs_a_wider_accumulator.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
=== Part 1: real round-trip error, calibrated from real data ===
Calibrated from 1000 real samples: scale=0.061880, zero_point=23
Real max abs round-trip error: 0.030918 (theoretical bound scale/2 = 0.030940)
Real mean abs round-trip error: 0.015629
Quantized int8 range actually used: [-128, 127] out of the legal [-128, 127]

=== Part 2: the real reason quantization-aware codegen is a genuinely new problem -- accumulator width ===
64 real int8 x int8 products, accumulated the WRONG way (int8 accumulator, matching every OTHER op this book has ever generated code for -- same width in, same width out):
  62 of 64 individual products already overflow int8 on their own (magnitude > 127) before any summation even happens
  final int8-accumulated sum: -17 (silently wrapped, no exception, no warning at the array level -- real numpy 2.4.4 behavior)
  final real (int32) sum:     -17169
  WRONG by: -17152

Real aside: the identical overflow on a single SCALAR int8 pair (53 * 75 -> wrapped to -121) DOES raise a real numpy RuntimeWarning ("overflow encountered in scalar multiply"), even though the array-level version just computed above raises nothing -- numpy's own overflow reporting is genuinely inconsistent between vectorized and scalar integer arithmetic, an honest quirk worth knowing before trusting numpy's silence as proof of correctness.

This is the one real, structural codegen change quantization forces: Part 4's own emitSteps()/generateCudaElementwiseKernel() never had to distinguish an operation's OWN type from its ACCUMULATOR's type, because every prior CUDA Hammer op used the same float type for both. A real quantized-matmul kernel generator has to emit code where the loop's own running sum is declared int32 even though every value it reads is int8 -- a genuinely new codegen requirement, not just a new op kind.
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
=== Part 1: real round-trip error, calibrated from real data ===
Calibrated from 1000 real samples: scale=0.061880, zero_point=23
Real max abs round-trip error: 0.030918 (theoretical bound scale/2 = 0.030940)
Real mean abs round-trip error: 0.015629
Quantized int8 range actually used: [-128, 127] out of the legal [-128, 127]

=== Part 2: the real reason quantization-aware codegen is a genuinely new problem -- accumulator width ===
64 real int8 x int8 products, accumulated the WRONG way (int8 accumulator, matching every OTHER op this book has ever generated code for -- same width in, same width out):
  62 of 64 individual products already overflow int8 on their own (magnitude > 127) before any summation even happens
  final int8-accumulated sum: -17 (silently wrapped, no exception, no warning at the array level -- real numpy 2.2.6 behavior)
  final real (int32) sum:     -17169
  WRONG by: -17152

Real aside: the identical overflow on a single SCALAR int8 pair (53 * 75 -> wrapped to -121) DOES raise a real numpy RuntimeWarning ("overflow encountered in scalar multiply"), even though the array-level version just computed above raises nothing -- numpy's own overflow reporting is genuinely inconsistent between vectorized and scalar integer arithmetic, an honest quirk worth knowing before trusting numpy's silence as proof of correctness.

This is the one real, structural codegen change quantization forces: Part 4's own emitSteps()/generateCudaElementwiseKernel() never had to distinguish an operation's OWN type from its ACCUMULATOR's type, because every prior CUDA Hammer op used the same float type for both. A real quantized-matmul kernel generator has to emit code where the loop's own running sum is declared int32 even though every value it reads is int8 -- a genuinely new codegen requirement, not just a new op kind.
```

*Identical apart from the printed NumPy version string itself (2.4.4 cloud vs. 2.2.6 device) -- every real number, including the overflow demonstration, agrees exactly.*


### A real, directly observed overflow -- and a real inconsistency in how NumPy reports it

File 078's own real, printed output shows 62 of 64 real random int8
products already exceeding int8's own representable range before any
summation even happens, and the resulting int8-width running sum lands
at `-17` against a true value of `-17169` -- wrong by 17,152, silently,
with real NumPy `2.4.4` raising nothing at the array level. A real,
honest aside this file's own investigation turned up along the way: the
IDENTICAL overflow on a single SCALAR int8 pair (rather than a whole
array) DOES raise a real `RuntimeWarning` from the same NumPy build --
numpy's own overflow reporting is genuinely inconsistent between
vectorized and scalar integer arithmetic, worth knowing before trusting
its silence as proof of correctness.

This is the one real, structural codegen change quantization forces on
this book's own accumulated Part 4 machinery: Chapter 17's own
`emitSteps()` and Chapter 18's own `generateCudaElementwiseKernel()`
never had to distinguish an operation's own type from its
ACCUMULATOR's type, because every prior CUDA Hammer op used the same
float type for both, throughout Parts 1-5. A real quantized-GEMM kernel
generator has to emit code where the loop's own running sum is declared
at a WIDER type than every value it reads -- a genuinely new codegen
requirement, confirmed directly against real arithmetic, not asserted
from theory.

```bash
python3 "078_affine_quantization_math_and_why_codegen_needs_a_wider_accumulator.py"
```

```python
# Chapter 30, File 079: File 078's own hand-argued claim -- that a
# quantized dot product needs its accumulator's own TYPE widened, not
# just a wider value -- confirmed as real, automatically-inserted
# compiler behavior, by writing the same computation as one real Triton
# kernel and reading what its own compiler actually emits.
#
# Real toolchain note, same honest limitation as every Triton file since
# Chapter 27: no physical GPU here, so this file is real AOT COMPILATION
# only, never execution -- File 078's own pure-NumPy version already
# proved the arithmetic correct; this file proves the SAME accumulator-
# widening step is something Triton's own compiler inserts automatically
# from ordinary Python type-conversion syntax (`.to(tl.int32)`), not
# something the block-level programmer has to hand-encode as a separate
# instruction.
#
# Compiled with:   python3 "079_a_real_triton_quantized_dot_kernel_compiled_with_no_gpu.py"

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def quantized_dot_kernel(a_ptr, b_ptr, out_ptr, scale_a, scale_b,
                          N: tl.constexpr):
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
    products = a32 * b32          # real int32 * int32, no overflow risk
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
    return sum(1 for line in ir_text.splitlines() if op_name in line)


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
    load_idx = next(i for i, l in enumerate(lines) if "b_3 = tt.load" in l)
    for line in lines[load_idx:load_idx + 4]:
        print(line.strip())

    print()
    print("=== Real TTIR excerpt: the int32 reduction (path-normalized) ===")
    reduce_idx = next(i for i, l in enumerate(lines) if "tt.reduce" in l)
    for line in lines[reduce_idx:reduce_idx + 4]:
        print(line.strip())
```

```bash
python3 "079_a_real_triton_quantized_dot_kernel_compiled_with_no_gpu.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
Real compiled pipeline stages: ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
  ttir: 4052 real chars
  ttgir: 4428 real chars
  llir: 5596 real chars
  ptx: 12145 real chars

Real arith.extsi (sign-extend int8->int32) count: 2 -- ONE PER real int8 input (a and b), confirming Triton's own compiler inserts a real, SEPARATE widening instruction automatically from the source text's own plain `.to(tl.int32)` call, exactly the codegen step File 078 argued was structurally necessary
Real arith.muli (the widened int32 multiply) count: 1
Real tt.reduce (the int32 accumulate) count: 2
Real arith.sitofp (signed-int-to-float, for dequantization) count: 1

=== Real TTIR excerpt: load through widen through multiply (path-normalized) ===
%b_3 = tt.load %b_2 : tensor<64x!tt.ptr<i8>> loc(#loc25)
%a32 = arith.extsi %a_1 : tensor<64xi8> to tensor<64xi32> loc(#loc26)
%b32 = arith.extsi %b_3 : tensor<64xi8> to tensor<64xi32> loc(#loc27)
%products = arith.muli %a32, %b32 : tensor<64xi32> loc(#loc28)

=== Real TTIR excerpt: the int32 reduction (path-normalized) ===
%acc = "tt.reduce"(%products) <{axis = 0 : i32}> ({
^bb0(%acc_5: i32 loc(callsite(#loc11 at #loc29)), %acc_6: i32 loc(callsite(#loc11 at #loc29))):
%acc_7 = arith.addi %acc_5, %acc_6 : i32 loc(#loc34)
tt.reduce.return %acc_7 : i32 loc(#loc32)
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
Real compiled pipeline stages: ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
  ttir: 4733 real chars
  ttgir: 5109 real chars
  llir: 5678 real chars
  ptx: 12592 real chars

Real arith.extsi (sign-extend int8->int32) count: 2 -- ONE PER real int8 input (a and b), confirming Triton's own compiler inserts a real, SEPARATE widening instruction automatically from the source text's own plain `.to(tl.int32)` call, exactly the codegen step File 078 argued was structurally necessary
Real arith.muli (the widened int32 multiply) count: 1
Real tt.reduce (the int32 accumulate) count: 2
Real arith.sitofp (signed-int-to-float, for dequantization) count: 1

=== Real TTIR excerpt: load through widen through multiply (path-normalized) ===
%b_3 = tt.load %b_2 : tensor<64x!tt.ptr<i8>> loc(#loc25)
%a32 = arith.extsi %a_1 : tensor<64xi8> to tensor<64xi32> loc(#loc26)
%b32 = arith.extsi %b_3 : tensor<64xi8> to tensor<64xi32> loc(#loc27)
%products = arith.muli %a32, %b32 : tensor<64xi32> loc(#loc28)

=== Real TTIR excerpt: the int32 reduction (path-normalized) ===
%acc = "tt.reduce"(%products) <{axis = 0 : i32}> ({
^bb0(%acc_5: i32 loc(callsite(#loc11 at #loc29)), %acc_6: i32 loc(callsite(#loc11 at #loc29))):
%acc_7 = arith.addi %acc_5, %acc_6 : i32 loc(#loc34)
tt.reduce.return %acc_7 : i32 loc(#loc32)
```

*Every real op count and IR excerpt agrees between the two machines -- only the raw ttir/ttgir/llir/ptx character COUNTS differ, from the same embedded absolute-path-length artifact Chapter 27 already traced and explained.*


## 30.2 The Same Shape as One Real Triton Kernel

File 078 argued, by hand, that a quantized dot product's own generated
code has to widen before multiplying. File 079 writes that same
computation as one real `@triton.jit` kernel -- ordinary Triton syntax,
`.to(tl.int32)`, with no manual overflow bookkeeping -- and compiles it
through the same explicit `target=GPUTarget(...)` technique Chapter 27
established (no physical GPU needed, real AOT compilation only).

```text
  quantized_dot_kernel (one @triton.jit function):

  load a, b            (tt.load, real int8 block-level tensors)
  a32 = a.to(int32)     -- Triton's own compiler must insert a REAL,
  b32 = b.to(int32)        SEPARATE widening instruction here
  products = a32 * b32  -- now a real int32 multiply, no overflow risk
  acc = sum(products)   -- a real int32 block-level REDUCTION
  dequant = float(acc) * (scale_a * scale_b)
  store dequant
```

File 079's own real, printed TTIR confirms the claim directly, not by
assumption: `arith.extsi` (a real MLIR sign-extend instruction) appears
exactly TWICE -- once per real int8 input -- immediately before the real
`arith.muli` that multiplies the now-widened values, and a real
`tt.reduce` block performs the int32 accumulation with `arith.addi`
inside it. Triton's own compiler inserts this widening step
automatically from nothing more than the source text's own plain
`.to(tl.int32)` call; the block-level PROGRAMMER never has to write a
manual overflow check the way File 078's own hand-argued case implied
one might need to. Real, verified structural finding (the same
comparison every Triton file in this book has made): every real IR
excerpt and op count is byte-identical between the cloud sandbox and the
device, with only the raw ttir/ttgir/llir/ptx character COUNTS differing
from the same embedded absolute-path-length artifact Chapter 27 already
traced.

```bash
python3 "079_a_real_triton_quantized_dot_kernel_compiled_with_no_gpu.py"
```

```python
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
```

```bash
python3 "080_real_pytorch_quantized_gemm_kernels_and_the_part_6_synthesis.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed output; this file is verified on the cloud sandbox only, per Chapter 28's own disk-space finding):**

```text
=== Part 1: the real legacy quantized-tensor API is deprecated ===
Real deprecation warning on torch.quantize_per_tensor(): "torch.quantize_per_tensor, torch.quantize_per_channel and other quantized tensor creation functions that produce tensors with dtype torch.quint8, torch.qint8, and torch.qint32 are deprecated and will ..."
This legacy path still works today (int_repr = [9, 33, 10, -5]), but real, current PyTorch is already telling real users to move off it -- the real, still-supported replacement this file uses instead is a pair of plain torch.ops.aten GEMM primitives on ordinary int8 tensors, no special quantized dtype at all.

=== Part 2: real full-int8 GEMM, torch.ops.aten._int_mm ===
Real schema: aten::_int_mm(Tensor self, Tensor mat2) -> Tensor
Real output dtype: torch.int32 (int8 x int8 -> int32, the SAME accumulator-widening principle Files 078 and 079 both confirmed is structurally necessary -- though this real op is a compiled, opaque C++/CUDA kernel with no IR this file can inspect the way File 079 read Triton's own real arith.extsi/tt.reduce ops directly; its OWN correctness is the only evidence available here, honestly, not its internal structure)
Real correctness check: _int_mm vs. manual int32 matmul = True

=== Part 3: real weight-only quantized GEMM, torch.ops.aten._weight_int8pack_mm -- a DIFFERENT real quantization scheme ===
Real schema: aten::_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales) -> Tensor
Real correctness check: _weight_int8pack_mm vs. manual dequantize-then-matmul = True
Real, honest scope note: Files 078 and 079 both built the FULL-int8 scheme (Part 2's own _int_mm) by hand. This real op confirms production systems support a genuinely SEPARATE quantization shape neither of those files implements -- real quantization-aware codegen is not one fixed recipe.

=== Part 4 (Part 6's own closing synthesis): does Inductor fuse a quantized GEMM from primitives, or dispatch to it? ===
Real correctness check: torch.compile'd vs. eager match = True
Real generated cpp_fused kernel count: 0 (Chapter 28.1's own diamond graph got its own generated kernel here this same way; Chapter 29.3's flash attention got 0; this real quantized GEMM gets 0)
Real generated code calls straight through to extern_kernels._int_mm: True
Real, additional evidence: real, current Inductor ships a NAMED config field, torch._inductor.config.force_fuse_int_mm_with_mul (default False), existing purely to let a user opt INTO fusing _int_mm with a following multiply -- real, direct evidence Inductor's own developers treat that one specific combination as a hand-written SPECIAL CASE worth naming and gating, not something generic automatic fusion discovers on its own.

Part 6's own closing finding, confirmed a second time by a second, unrelated real algorithm: Chapter 29 found Inductor dispatches to a hand-written Flash Attention kernel rather than decomposing and re-fusing it; this file finds the exact same shape for quantized GEMM -- zero generated kernels, a direct call to a real, separately implemented aten op, and even a real NAMED config knob for the one specific fusion (with a following multiply) its own authors judged worth hand-coding. Across every real system this Part studied -- XLA, Relax, Triton, Inductor -- automatic fusion covers ORDINARY elementwise and reduction graphs; algorithms whose own numeric behavior genuinely differs (a streaming softmax, an integer accumulator) are handled by real, separately written kernels every time, never invented by a fusion pass.
```

*The real deprecation finding, both real GEMM kernels' own real correctness checks, and the real zero-generated-kernel Inductor finding all confirmed in one real, live run.*
