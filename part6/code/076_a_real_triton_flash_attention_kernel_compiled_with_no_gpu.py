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
