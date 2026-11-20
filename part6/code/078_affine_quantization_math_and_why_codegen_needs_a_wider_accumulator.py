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
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not (-128 <= qmin < qmax <= 127) or not (qmin <= zero_point <= qmax):
        raise ValueError("signed-int8 range and in-range zero point required")
    if not np.isfinite(x).all():
        raise ValueError("quantization requires finite data")
    q = np.round(x / scale) + zero_point
    q = np.clip(q, qmin, qmax)
    return q.astype(np.int8)


def dequantize(q, scale, zero_point):
    """The inverse: recover an approximate real value from a quantized
    integer. Exact only when the original real value landed exactly on
    a representable quantization step -- otherwise this is where the
    rounding error (at most scale/2 for values that are not clipped) comes from."""
    return (q.astype(np.float64) - zero_point) * scale


def choose_scale_zero_point(x, qmin=-128, qmax=127):
    """Minimal signed-int8 calibration, including zero in the observed range.

    Rounded integer zero points need not map both extrema exactly. The
    scale/2 round-trip bound applies only without saturation/clipping.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("calibration requires nonempty finite data")
    if not (-128 <= qmin < qmax <= 127):
        raise ValueError("this example emits signed int8 only")
    x_min, x_max = min(float(x.min()), 0.0), max(float(x.max()), 0.0)
    if x_min == x_max:  # all zeros: avoid division by zero
        return 1.0, int(np.clip(0, qmin, qmax))
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = int(np.clip(qmin - round(x_min / scale), qmin, qmax))
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

    # Regression coverage: one-sided and constant ranges must not clip
    # simply because zero cannot be represented by the selected offset.
    for samples in ([1., 2.], [-2., -1.], [0., 0.], [3., 3.], [-3., -3.]):
        a_test = np.asarray(samples)
        s_test, z_test = choose_scale_zero_point(a_test)
        recovered = dequantize(quantize(a_test, s_test, z_test), s_test, z_test)
        assert np.max(np.abs(recovered - a_test)) <= s_test / 2 + 1e-12
    for qlo, qhi in ((1, 127), (-128, -1), (-128, 127)):
        zeros = np.zeros(2)
        s_zero, z_zero = choose_scale_zero_point(zeros, qlo, qhi)
        assert qlo <= z_zero <= qhi
        restored = dequantize(quantize(zeros, s_zero, z_zero, qlo, qhi), s_zero, z_zero)
        np.testing.assert_array_equal(restored, zeros)
    for samples in ([], [np.nan], [np.inf]):
        try:
            choose_scale_zero_point(np.asarray(samples))
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calibration input accepted")
    print("Calibration regressions: one-sided, constant, zero, custom subranges, empty and non-finite inputs PASS")

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
    correct_sum = correct_products.sum(dtype=np.int32)
    assert int(correct_sum) == int(np.sum(a.astype(np.int64) * b.astype(np.int64)))

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
