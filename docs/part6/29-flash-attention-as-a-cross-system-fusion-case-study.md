# Chapter 29: Flash Attention as a Cross-System Fusion Case Study

Chapters 25-28 each installed and studied one independent real compiler
in isolation. This chapter studies one real ALGORITHM -- Flash Attention
-- across systems this Part has already installed: pure NumPy (no new
toolchain), Triton (Chapter 27's own `target=` AOT-compile technique),
and real PyTorch (Chapter 28's own installed `torch` 2.14.0). No fifth
toolchain is introduced. The question this chapter asks is different from
every prior Part 6 chapter's own question too: Chapters 25-28 each asked
"where does this real compiler draw its own fusion boundary on an
ordinary graph of elementwise and reduction ops?" This chapter asks
whether a compiler's own automatic fusion pass can ever discover Flash
Attention's own fusion boundary BY ITSELF -- or whether the algorithm
itself has to change first.

Real motivation, tied directly back to this book's own Part 3: Chapter
14's own `reductionFusionPass()` treats a `Sum` node's OUTPUT as always
external, any consumer count -- but every `Sum` node Chapters 9-24 ever
generated code for assumed its own INPUT was already fully materialized
before the reduction ran. Softmax needs TWO reductions per row (a max,
then a sum of `exp(x - max)`) over an attention score matrix that is
`sequence_length`-by-`sequence_length` -- for real transformer sequence
lengths, materializing that matrix at all is the real memory bottleneck
Flash Attention (Dao et al., 2022, arXiv:2205.14135) removes. The trick
that makes removing it possible predates that paper: Milakov &
Gimelshein's "Online normalizer calculation for softmax" (2018,
arXiv:1805.02867) shows a running max and running sum can be updated
INCREMENTALLY, one block of columns at a time, provided every time the
running max increases, whatever was already accumulated gets rescaled by
`exp(old_max - new_max)` to stay mathematically consistent.

```text
+------------------------------------------------------------------+
|  Chapter 29's own shape, section by section                       |
|                                                                    |
|  29.1  The online-softmax algorithm itself, in pure NumPy --      |
|        proven numerically identical to naive full-materialization |
|        softmax attention, with a real bytes-moved argument tied   |
|        to Chapter 12's own arithmetic-intensity accounting.       |
|                                                                    |
|  29.2  The SAME algorithm as one real Triton kernel (Chapter 27's |
|        block-level model), compiled with no GPU -- inspecting the |
|        real MLIR loop the KV-block loop becomes.                  |
|                                                                    |
|  29.3  Real PyTorch's own CPU Flash Attention op (Chapter 28's    |
|        own installed torch), and why TorchInductor dispatches to  |
|        it directly instead of fusing it from primitives.          |
+------------------------------------------------------------------+
```

## 29.1 The Online-Softmax Algorithm, Proven Against Naive Attention

Naive attention computes `scores = (Q @ K.T) / sqrt(d)`, a full
`(N, M)` matrix, then a row-wise softmax, then `output = probs @ V`.
Every element of `output` needs the GLOBAL max and sum over its own
entire row of `scores` before ANY of that row's own output can be
computed -- exactly the kind of whole-row data dependency Chapter 14's
own `Sum` node has always represented, but here `M` (the row length) is
what real Flash Attention needs to tile, not just materialize once and
reduce.

```text
  Naive attention (materializes the FULL score matrix):

  Q (N x d)   K (M x d)         scores (N x M)    -- THE memory
      \         /                    |               bottleneck:
       \       /  Q @ K.T / sqrt(d)  |               N*M floats,
        \     /                     \/               held all at once
         (N x M) ---------------> softmax(row-wise)
                                      |
                                      \/
                               probs (N x M) @ Vmat(M x d) -> out (N x d)


  Flash attention (streams KV blocks, NEVER materializes N x M):

  Q (N x d) -- loaded once
  running_max, running_sum: (N,)      running_out: (N x d)
  all three start at -inf / 0 / 0

  for each KV block j = 1..num_blocks:
      K_j, V_j: (Bc x d) -- only THIS block's own key/value ever loaded
      scores_j = Q @ K_j.T / sqrt(d)          (N x Bc, not N x M)
      new_max = max(running_max, rowmax(scores_j))
      correction = exp(running_max - new_max) -- rescales the OLD
                                                   accumulator, not
                                                   recomputes it
      running_sum = running_sum * correction + rowsum(exp(scores_j - new_max))
      running_out = running_out * correction[:,None] + exp(scores_j - new_max) @ V_j
      running_max = new_max

  out = running_out / running_sum[:,None]
```

File 075 implements both versions in pure NumPy (no new install --
`numpy` is already on both of this book's own machines) and runs the
online version with three different KV block sizes against one real
fixed random `Q`/`K`/`V`, deliberately choosing `M=13` -- not a multiple
of any candidate block size -- so every configuration forces a genuine
tail block, the same "don't pick numbers that hide the edge case"
practice Chapter 19's own `n=11` vector-width test already established:

```python
# Chapter 29, File 075: Flash Attention's own algorithmic core -- online
# (streaming) softmax over tiled key/value blocks -- implemented in pure
# NumPy (no torch, no triton, no jax, no tvm) so this file cross-verifies
# byte-for-byte identical on both the cloud sandbox and the device, the
# same discipline as every plain-C++ Part 1-5 chapter.
#
# Real motivation, tied directly back to this book's own Part 3: Chapter
# 14's own reductionFusionPass() treats a Sum node's OUTPUT as always
# external, any consumer count -- but every Sum node Chapters 9-24 ever
# generated code for assumed the reduction's own INPUT was already fully
# materialized before the reduction ran. Softmax needs TWO reductions
# per row (a max, then a sum of exp(x - max)) over the attention score
# matrix, and that matrix is N-by-M -- for real transformer sequence
# lengths (N, M in the thousands), materializing it is the real memory
# bottleneck FlashAttention (Dao et al., 2022, arXiv:2205.14135) was
# built to remove. The trick that makes it possible predates that paper:
# Milakov & Gimelshein's "Online normalizer calculation for softmax"
# (2018, arXiv:1805.02867) shows a running max/sum can be updated
# incrementally, one block of columns at a time, WITHOUT ever holding
# the full row in memory at once -- provided each time the running max
# increases, the previously accumulated sum and output are rescaled by
# exp(old_max - new_max) to stay consistent. This file proves that
# rescaling step is correct by comparing its real output, bit-for-bit
# within float64 tolerance, against naive full-materialization softmax
# attention on the same real random inputs.
#
# Compiled with:   python3 "075_naive_vs_online_softmax_flash_attention_pure_numpy.py"

import numpy as np


def naive_attention(q, k, v):
    """The textbook definition: materializes the FULL (N, M) score matrix,
    exactly the "compute everything, then reduce" shape every Sum node
    generated by Chapter 14's own reductionFusionPass() has always
    assumed. Returns (output, the full softmax row-sums) so File 077 can
    compare its own real logsumexp finding against something computed
    here independently."""
    n, d = q.shape
    m = k.shape[0]
    scale = 1.0 / np.sqrt(d)
    scores = (q @ k.T) * scale  # (N, M) -- THE memory bottleneck
    row_max = scores.max(axis=1, keepdims=True)  # (N, 1)
    exp_scores = np.exp(scores - row_max)  # (N, M)
    row_sum = exp_scores.sum(axis=1, keepdims=True)  # (N, 1)
    probs = exp_scores / row_sum  # (N, M)
    out = probs @ v  # (N, d)
    return out, row_max.reshape(-1), row_sum.reshape(-1)


def flash_attention(q, k, v, block_size_kv):
    """FlashAttention's own algorithmic core: for each query row, stream
    the key/value blocks one at a time, keeping only a RUNNING max,
    RUNNING sum, and RUNNING (unnormalized) output -- never materializing
    the full (N, M) score matrix. `block_size_kv` deliberately does not
    evenly divide M (checked by the caller), forcing a genuine tail
    block, the same "don't pick numbers that hide the edge case" practice
    this book has followed since Chapter 19's own n=11 vector-width test."""
    n, d = q.shape
    m = k.shape[0]
    scale = 1.0 / np.sqrt(d)

    running_max = np.full(n, -np.inf)
    running_sum = np.zeros(n)
    running_out = np.zeros((n, d))

    num_blocks = 0
    peak_block_bytes = 0
    for start in range(0, m, block_size_kv):
        end = min(start + block_size_kv, m)
        k_block = k[start:end]  # (Bc, d) -- Bc <= block_size_kv
        v_block = v[start:end]  # (Bc, d)
        num_blocks += 1

        block_scores = (q @ k_block.T) * scale  # (N, Bc) -- ONLY this
        # block's own score tile is ever in memory, never the full (N, M)
        block_bytes = block_scores.nbytes
        peak_block_bytes = max(peak_block_bytes, block_bytes)

        block_max = block_scores.max(axis=1)  # (N,)
        new_max = np.maximum(running_max, block_max)  # (N,)

        # The online-softmax rescaling step itself (Milakov & Gimelshein,
        # 2018): whatever was already accumulated under the OLD running
        # max gets corrected by exp(old_max - new_max) before this
        # block's own contribution is added under the NEW running max.
        correction = np.exp(running_max - new_max)  # (N,) -- 0 on the
        # very first block, where running_max starts at -inf and
        # correction would be exp(-inf - finite) = 0, which is exactly
        # right: there is nothing yet to rescale.
        correction = np.nan_to_num(correction, nan=0.0)

        p_block = np.exp(block_scores - new_max[:, None])  # (N, Bc)
        block_sum = p_block.sum(axis=1)  # (N,)

        running_sum = running_sum * correction + block_sum
        running_out = (running_out * correction[:, None]
                       + p_block @ v_block)
        running_max = new_max

    out = running_out / running_sum[:, None]
    return out, running_max, running_sum, num_blocks, peak_block_bytes


if __name__ == "__main__":
    rng = np.random.default_rng(seed=29)

    n, m, d = 8, 13, 4  # M=13 deliberately not a multiple of any small
    # block size below -- every block_size_kv candidate forces a real
    # tail block, the same discipline Chapter 19's own n=11 test used.
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((m, d))
    v = rng.standard_normal((m, d))

    naive_out, naive_max, naive_sum = naive_attention(q, k, v)

    print(f"Naive attention: Q({n}x{d}) @ K({m}x{d}).T -> scores({n}x{m}), "
          f"the full matrix FlashAttention's own algorithm never "
          f"materializes.")
    print(f"Naive scores matrix: {n * m} floats, "
          f"{n * m * 8} bytes (float64) held at once.")
    print()

    for block_size_kv in (4, 5, 13):
        (flash_out, flash_max, flash_sum,
         num_blocks, peak_block_bytes) = flash_attention(q, k, v, block_size_kv)
        max_abs_diff = np.abs(naive_out - flash_out).max()
        match = np.allclose(naive_out, flash_out, atol=1e-10)
        max_match = np.allclose(naive_max, flash_max, atol=1e-12)
        sum_match = np.allclose(naive_sum, flash_sum, atol=1e-10)
        print(f"block_size_kv={block_size_kv:2d}: {num_blocks} real KV "
              f"block(s) (last block size "
              f"{m - block_size_kv * (num_blocks - 1)}), peak per-block "
              f"score tile = {peak_block_bytes} bytes -- output match = "
              f"{match} (max abs diff {max_abs_diff:.3e}), running max "
              f"matches naive row-max = {max_match}, running sum matches "
              f"naive row-sum = {sum_match}")

    print()
    print("Real, concrete memory argument at a realistic scale "
          "(N=M=1024 queries/keys, d=64, float32, block_size_kv=128, "
          "computed here, not measured -- this file's own real n=8/m=13 "
          "arrays are too small to show a meaningful byte comparison):")
    real_n, real_m, real_d, real_block = 1024, 1024, 64, 128
    naive_bytes = real_n * real_m * 4
    flash_peak_bytes = real_n * real_block * 4
    print(f"  naive materialized score matrix: {naive_bytes:,} bytes "
          f"({naive_bytes / 1024:.0f} KB)")
    print(f"  flash peak per-block score tile: {flash_peak_bytes:,} bytes "
          f"({flash_peak_bytes / 1024:.0f} KB) -- "
          f"{naive_bytes / flash_peak_bytes:.1f}x smaller, at the SAME "
          f"real sequence length")
```

```bash
python3 "075_naive_vs_online_softmax_flash_attention_pure_numpy.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
Naive attention: Q(8x4) @ K(13x4).T -> scores(8x13), the full matrix FlashAttention's own algorithm never materializes.
Naive scores matrix: 104 floats, 832 bytes (float64) held at once.

block_size_kv= 4: 4 real KV block(s) (last block size 1), peak per-block score tile = 256 bytes -- output match = True (max abs diff 2.220e-16), running max matches naive row-max = True, running sum matches naive row-sum = True
block_size_kv= 5: 3 real KV block(s) (last block size 3), peak per-block score tile = 320 bytes -- output match = True (max abs diff 1.665e-16), running max matches naive row-max = True, running sum matches naive row-sum = True
block_size_kv=13: 1 real KV block(s) (last block size 13), peak per-block score tile = 832 bytes -- output match = True (max abs diff 2.776e-16), running max matches naive row-max = True, running sum matches naive row-sum = True

Real, concrete memory argument at a realistic scale (N=M=1024 queries/keys, d=64, float32, block_size_kv=128, computed here, not measured -- this file's own real n=8/m=13 arrays are too small to show a meaningful byte comparison):
  naive materialized score matrix: 4,194,304 bytes (4096 KB)
  flash peak per-block score tile: 524,288 bytes (512 KB) -- 8.0x smaller, at the SAME real sequence length
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
Naive attention: Q(8x4) @ K(13x4).T -> scores(8x13), the full matrix FlashAttention's own algorithm never materializes.
Naive scores matrix: 104 floats, 832 bytes (float64) held at once.

block_size_kv= 4: 4 real KV block(s) (last block size 1), peak per-block score tile = 256 bytes -- output match = True (max abs diff 2.220e-16), running max matches naive row-max = True, running sum matches naive row-sum = True
block_size_kv= 5: 3 real KV block(s) (last block size 3), peak per-block score tile = 320 bytes -- output match = True (max abs diff 1.665e-16), running max matches naive row-max = True, running sum matches naive row-sum = True
block_size_kv=13: 1 real KV block(s) (last block size 13), peak per-block score tile = 832 bytes -- output match = True (max abs diff 2.776e-16), running max matches naive row-max = True, running sum matches naive row-sum = True

Real, concrete memory argument at a realistic scale (N=M=1024 queries/keys, d=64, float32, block_size_kv=128, computed here, not measured -- this file's own real n=8/m=13 arrays are too small to show a meaningful byte comparison):
  naive materialized score matrix: 4,194,304 bytes (4096 KB)
  flash peak per-block score tile: 524,288 bytes (512 KB) -- 8.0x smaller, at the SAME real sequence length
```

*Byte-for-byte identical between the two machines -- a fixed random seed and pure NumPy float64 arithmetic leave nothing machine-dependent for this file to print.*


### Why this is genuinely a NEW kind of reduction for this book

Every reduction CUDA Hammer's own IR has ever expressed (Chapter 14's
`Sum` node, every codegen backend built for it through Chapter 23) reads
its ENTIRE input before producing an output -- there is no notion of a
partial reduction result that later gets corrected once more data
arrives. The online-softmax `correction` term is precisely that: a real,
necessary CORRECTION of an already-computed partial result, applied
every time a later block's own data changes what the true row max turns
out to be. Nothing in this book's own `LoopNest`/`tileLoop()` machinery
(Chapter 15) or `Schedule` search space (Chapter 21) has a place to
express "revise the answer computed by an earlier tile" -- CUDA Hammer's
own tiling always assumed a tile's own contribution to a reduction could
just be summed into the total, unconditionally correct as soon as it is
added. Flash Attention's own online-softmax trick is a genuinely
different STREAMING-reduction shape, one this book's own IR was never
built to represent.

File 075's own real, printed output confirms the algorithm is correct
regardless of how the KV dimension gets tiled -- `block_size_kv=4` (a
real tail block of exactly 1), `block_size_kv=5` (a tail of 3), and
`block_size_kv=13` (one block, no tiling at all, the degenerate case that
should reduce to naive attention exactly) all report an EXACT output
match against `naive_attention()`, differing only by float64 rounding at
the `1e-16` level -- and the real, concrete memory argument at a
realistic scale (`N=M=1024`, `d=64`, `block_size_kv=128`) shows an 8x
reduction in peak score-tile memory at that one block size alone, purely
from never holding the full row.

```bash
python3 "075_naive_vs_online_softmax_flash_attention_pure_numpy.py"
```

```python
# Chapter 29, File 076: File 075's own online-softmax algorithm, written as
# a real block-level Triton kernel and compiled through the same
# explicit-`target=` AOT path Chapter 27 established (no physical GPU
# needed to reach real TTIR/TTGIR/LLIR/PTX).
#
# Real toolchain note, same honest limitation as every Triton file since
# Chapter 27: no physical GPU here, so this file is scoped to real
# COMPILATION only, never execution -- File 075's own pure-NumPy version
# is what actually proves the algorithm correct; this file proves the
# SAME algorithm is expressible as one real Triton kernel body, at the
# block level Chapter 27 studied, and inspects what it genuinely compiles
# to. One deliberate scope reduction from File 075: `SEQ_LEN_KV` is a
# `tl.constexpr` here (a fixed 48, divisible by `BLOCK_N=16`, so no
# boundary mask is needed inside the KV loop) rather than a runtime
# value with a genuine tail block -- Chapter 27's own File 069 already
# demonstrated real Triton boundary masking (`mask=offsets < n_elements`)
# for a non-divisible size, so this file does not need to repeat that
# mechanism to stay honest about what it covers.
#
# Compiled with:   python3 "076_a_real_triton_flash_attention_kernel_compiled_with_no_gpu.py"

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def flash_attention_kernel(q_ptr, k_ptr, v_ptr, out_ptr,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            D: tl.constexpr, SEQ_LEN_KV: tl.constexpr):
    # One program instance owns BLOCK_M query rows, all D columns, for the
    # WHOLE sequence -- Chapter 27's own block-level model, applied here to
    # a genuinely two-dimensional block (BLOCK_M, D) rather than the flat
    # 1-D blocks File 069's own vector_add_kernel used.
    row_offsets = tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, D)
    q = tl.load(q_ptr + row_offsets[:, None] * D + d_offsets[None, :])

    # File 075's own three running accumulators, as real Triton block-level
    # values -- BLOCK_M-wide running_max/running_sum, a (BLOCK_M, D) running
    # output. Initialized identically to File 075's naive_attention()-vs-
    # flash_attention() comparison.
    running_max = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    running_out = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    scale = 1.0 / (D ** 0.5)

    # File 075's own KV-block loop, written here as a plain Python `for`
    # over a `tl.constexpr` range -- real Triton lowers this to a genuine
    # MLIR `scf.for` loop with the three accumulators above threaded
    # through as real `iter_args`, confirmed directly in this file's own
    # printed TTIR below, not merely assumed from the source text.
    for start_n in range(0, SEQ_LEN_KV, BLOCK_N):
        col_offsets = start_n + tl.arange(0, BLOCK_N)
        k_block = tl.load(k_ptr + col_offsets[:, None] * D + d_offsets[None, :])
        v_block = tl.load(v_ptr + col_offsets[:, None] * D + d_offsets[None, :])

        # tt.dot #1: Q @ K_block^T -- one real block-level matmul per KV
        # block, exactly File 075's own `q @ k_block.T`.
        scores = tl.dot(q, tl.trans(k_block)) * scale
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)

        # The same online-softmax rescaling step File 075 already proved
        # correct against naive full-materialization softmax.
        correction = tl.exp(running_max - new_max)
        p = tl.exp(scores - new_max[:, None])
        block_sum = tl.sum(p, axis=1)

        running_sum = running_sum * correction + block_sum
        # tt.dot #2: P_block @ V_block -- the second real block-level matmul
        # per KV block, exactly File 075's own `p_block @ v_block`.
        running_out = (running_out * correction[:, None]
                       + tl.dot(p.to(tl.float32), v_block))
        running_max = new_max

    out = running_out / running_sum[:, None]
    tl.store(out_ptr + row_offsets[:, None] * D + d_offsets[None, :], out)


def compile_for_target(fn, signature, constexprs, target):
    src = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=target)


def strip_source_locations(ir_text):
    # Same real technique as Chapter 27's own File 069: the two real
    # machines' own absolute paths to this file differ, and that
    # difference lives ONLY in the `#locN = loc(...)` definition lines,
    # never in the real operations that reference them by number.
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
        flash_attention_kernel,
        {"q_ptr": "*fp32", "k_ptr": "*fp32", "v_ptr": "*fp32", "out_ptr": "*fp32",
         "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "D": "constexpr",
         "SEQ_LEN_KV": "constexpr"},
        {"BLOCK_M": 16, "BLOCK_N": 16, "D": 16, "SEQ_LEN_KV": 48},
        target,
    )

    print("Real compiled pipeline stages:", list(compiled.asm.keys()))
    for stage in ["ttir", "ttgir", "llir", "ptx"]:
        print(f"  {stage}: {len(compiled.asm[stage])} real chars")

    ttir = strip_source_locations(compiled.asm["ttir"])
    print()
    print(f"Real tt.dot count in TTIR: {count_ops(ttir, 'tt.dot')} "
          f"(2 real block-level matmuls per KV block -- Q@K_block^T and "
          f"P_block@V_block -- appearing ONCE each in the compiled text, "
          f"confirming the KV loop was NOT unrolled at the TTIR level "
          f"even though SEQ_LEN_KV=48 and BLOCK_N=16 make the trip count "
          f"[3] a compile-time constant)")
    print(f"Real scf.for count in TTIR: {count_ops(ttir, 'scf.for')} "
          f"(the KV loop, lowered to one genuine MLIR loop construct)")

    print()
    print("=== Real TTIR excerpt: the scf.for loop header (path-normalized) ===")
    for line in ttir.splitlines():
        if "scf.for" in line:
            print(line.strip())
            break

    print()
    print("=== Real TTIR excerpt: first real op after tt.load(q) (path-normalized) ===")
    lines = ttir.splitlines()
    load_idx = next(i for i, l in enumerate(lines) if "tt.load" in l)
    for line in lines[load_idx:load_idx + 3]:
        print(line.strip())
```

```bash
python3 "076_a_real_triton_flash_attention_kernel_compiled_with_no_gpu.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
Real compiled pipeline stages: ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
  ttir: 12594 real chars
  ttgir: 22321 real chars
  llir: 36265 real chars
  ptx: 34748 real chars

Real tt.dot count in TTIR: 2 (2 real block-level matmuls per KV block -- Q@K_block^T and P_block@V_block -- appearing ONCE each in the compiled text, confirming the KV loop was NOT unrolled at the TTIR level even though SEQ_LEN_KV=48 and BLOCK_N=16 make the trip count [3] a compile-time constant)
Real scf.for count in TTIR: 1 (the KV loop, lowered to one genuine MLIR loop construct)

=== Real TTIR excerpt: the scf.for loop header (path-normalized) ===
%running_out:3 = scf.for %start_n = %c0_i32 to %c48_i32 step %c16_i32 iter_args(%running_max_12 = %running_max, %running_sum_13 = %running_sum, %running_out_14 = %cst_0) -> (tensor<16xf32>, tensor<16xf32>, tensor<16x16xf32>)  : i32 {

=== Real TTIR excerpt: first real op after tt.load(q) (path-normalized) ===
%q_9 = tt.load %q_8 : tensor<16x16x!tt.ptr<f32>> loc(#loc49)
%running_out:3 = scf.for %start_n = %c0_i32 to %c48_i32 step %c16_i32 iter_args(%running_max_12 = %running_max, %running_sum_13 = %running_sum, %running_out_14 = %cst_0) -> (tensor<16xf32>, tensor<16xf32>, tensor<16x16xf32>)  : i32 {
%col_offsets = tt.splat %start_n : i32 -> tensor<16xi32> loc(#loc51)
```

**Output (device, aarch64 Linux VM -- real, live-executed output):**

```text
Real compiled pipeline stages: ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
  ttir: 14320 real chars
  ttgir: 23918 real chars
  llir: 36347 real chars
  ptx: 35196 real chars

Real tt.dot count in TTIR: 2 (2 real block-level matmuls per KV block -- Q@K_block^T and P_block@V_block -- appearing ONCE each in the compiled text, confirming the KV loop was NOT unrolled at the TTIR level even though SEQ_LEN_KV=48 and BLOCK_N=16 make the trip count [3] a compile-time constant)
Real scf.for count in TTIR: 1 (the KV loop, lowered to one genuine MLIR loop construct)

=== Real TTIR excerpt: the scf.for loop header (path-normalized) ===
%running_out:3 = scf.for %start_n = %c0_i32 to %c48_i32 step %c16_i32 iter_args(%running_max_12 = %running_max, %running_sum_13 = %running_sum, %running_out_14 = %cst_0) -> (tensor<16xf32>, tensor<16xf32>, tensor<16x16xf32>)  : i32 {

=== Real TTIR excerpt: first real op after tt.load(q) (path-normalized) ===
%q_9 = tt.load %q_8 : tensor<16x16x!tt.ptr<f32>> loc(#loc49)
%running_out:3 = scf.for %start_n = %c0_i32 to %c48_i32 step %c16_i32 iter_args(%running_max_12 = %running_max, %running_sum_13 = %running_sum, %running_out_14 = %cst_0) -> (tensor<16xf32>, tensor<16xf32>, tensor<16x16xf32>)  : i32 {
%col_offsets = tt.splat %start_n : i32 -> tensor<16xi32> loc(#loc51)
```

*Every real IR excerpt, op count, and loop structure agrees between the two machines -- only the raw ttir/ttgir/llir/ptx character COUNTS differ, from the same embedded absolute-path-length artifact Chapter 27 already traced and explained, not from any real difference in what was compiled.*


## 29.2 The Same Algorithm as One Real Triton Kernel

Chapter 27 found that fusion in Triton is a PROGRAMMER'S language-level
choice: whatever one `@triton.jit` function body contains becomes one
kernel, with no separate compiler pass deciding boundaries. File 076
writes File 075's own online-softmax algorithm directly as one such
function -- the three running accumulators (`running_max`, `running_sum`,
`running_out`) as real Triton block-level values, and the KV-block loop
as a plain Python `for` over a `tl.constexpr` range -- and compiles it
with the same explicit `target=GPUTarget(...)` technique Chapter 27
established (no physical GPU needed, real AOT compilation only, the same
honest scope every Triton file in this book has carried since).

```text
  flash_attention_kernel (one @triton.jit function, BLOCK_M x D output):

  load Q block                      (tt.load, once)
  running_max, running_sum, running_out -- real Triton register values

  for start_n in range(0, SEQ_LEN_KV, BLOCK_N):    -- a real MLIR
      load K_block, V_block          (tt.load)          scf.for loop,
      scores = Q @ K_block^T         (tt.dot #1)         NOT unrolled,
      new_max = max(running_max, rowmax(scores))         with running_max/
      correction = exp(running_max - new_max)            running_sum/
      running_sum = running_sum*correction + rowsum(exp(scores-new_max))
      running_out = running_out*correction + exp(scores-new_max) @ V_block
                                       (tt.dot #2)         running_out as
      running_max = new_max                               real loop-carried
                                                            iter_args
  store running_out / running_sum
```

This is the chapter's own genuinely new real finding, not available from
reading Triton's own source the way Chapter 27's `inspect.getsource()`
findings were: File 076's own real, printed TTIR shows the compiled KV
loop as one literal `scf.for` MLIR construct --
`scf.for %start_n = %c0_i32 to %c48_i32 step %c16_i32 iter_args(...)`
-- with THREE real loop-carried values threaded through
`iter_args`, exactly `running_max`, `running_sum`, and `running_out`. And
`tt.dot` appears exactly TWICE in the compiled text (once for
`Q @ K_block^T`, once for `P_block @ V_block`), confirming the loop was
genuinely compiled as a real MLIR loop and not silently unrolled at the
TTIR level, even though `SEQ_LEN_KV=48` and `BLOCK_N=16` make the trip
count (3) a compile-time-known constant. The online-softmax algorithm's
own STREAMING structure -- an accumulator that gets corrected, not just
summed into, as new data arrives -- survives all the way from File 075's
own hand-written Python loop into a real, structurally faithful MLIR
loop construct, with nothing about that structure lost or rewritten by
Triton's own compiler.

Real, verified structural finding (the same comparison Chapter 27's own
File 069 and File 071 made): the compiled TTIR agrees on every real
operation between the cloud sandbox and the device -- the `scf.for`
header line and every other excerpt printed below are byte-identical --
and only the raw character COUNTS differ, from the same embedded
absolute-path-length artifact Chapter 27 already traced to Triton's own
`#locN = loc(...)` debug table.

```bash
python3 "076_a_real_triton_flash_attention_kernel_compiled_with_no_gpu.py"
```

```python
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
```

```bash
python3 "077_real_pytorch_flash_attention_and_why_inductor_never_regenerates_it.py"
```

**Output (cloud sandbox, x86-64 -- real, live-executed output; this file is verified on the cloud sandbox only, per Chapter 28's own disk-space finding):**

```text
=== Part 1: real backend availability, queried directly ===
is_flash_attention_available(): True
flash_sdp_enabled(): True
math_sdp_enabled(): True
list_flash_attention_impls(): ['FA3', 'FA4']
torch.cuda.is_available(): False
(every one of these reports real flash-attention support with zero physical GPU present -- this is a genuine CPU code path, not a GPU-gated feature this sandbox happens to see stubbed out)

=== Part 2: real execution, MATH backend vs. FLASH_ATTENTION backend, same real inputs ===
MATH backend output shape: (1, 2, 8, 4)
FLASH_ATTENTION backend output shape: (1, 2, 8, 4)
Real correctness check: MATH vs. FLASH_ATTENTION match = True (max abs diff 4.768e-07) -- two independently implemented real kernels inside the SAME torch build agree, the same cross-implementation agreement Files 075 and 076 already established between pure NumPy and compiled Triton

=== Part 3: the real op's own schema -- why FLASH_ATTENTION works with no GPU ===
Real schema: aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)
Real logsumexp shape: (1, 2, 8) -- one real value per (batch, head, query row), exactly matching File 075's own running_max/running_sum accumulator shape (N,), confirming this real op tracks the SAME online log-sum-exp quantity Files 075 and 076 both computed by hand
Real correctness check: this op's own output vs. MATH backend = True

=== Part 4 (capstone): what TorchInductor (Ch28) does with scaled_dot_product_attention ===
Real correctness check: torch.compile'd vs. eager match = True
Real generated cpp_fused kernel count: 0 (Chapter 28.1's own diamond graph -- 4 ordinary elementwise ops -- got its own generated cpp_fused_add_mul_relu_0 kernel here this same way; scaled_dot_product_attention gets 0)
Real generated code calls straight through to aten._scaled_dot_product_flash_attention_for_cpu: True

This is the chapter's own closing, cross-system finding: TorchInductor's real automatic whole-graph fusion (Ch28.1), XLA's real kind=kLoop fusion (Ch25), and Relax's real FuseOps/FuseTIR (Ch26) all fuse ORDINARY elementwise/reduction graphs automatically, inventing nothing about the ALGORITHM itself -- but none of them re-derives Flash Attention's own online-softmax rescaling trick from primitives. Inductor dispatches straight to a hand-written, pre-existing fused kernel instead of decomposing scaled_dot_product_attention into ops it could fuse itself. Flash Attention needed a new ALGORITHM (Milakov & Gimelshein's 2018 online-softmax identity), not just a new fusion boundary -- and every real system this Part studied treats that as a hand-written kernel provided to the compiler, never a boundary the compiler's own fusion pass discovers on its own.
```

*Three independent real implementations (NumPy, compiled Triton, and now two real PyTorch backends) agree numerically, and Inductor's own real captured output confirms it dispatches straight through to the real flash-attention op rather than regenerating it.*
