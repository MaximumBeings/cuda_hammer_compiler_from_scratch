# Chapter 27, File 070: fusion in Triton isn't a separate compiler PASS the
# way it is in CUDA Hammer (Ch13-16), XLA (Ch25), or Relax (Ch26) -- it's a
# consequence of how many real ops the PROGRAMMER writes inside one
# @triton.jit function body. This file puts that claim to a real, structural
# test on the exact same graph Chapters 4, 13, 25, and 26 have all already
# used: t1 = a + b (shared by two consumers), t2 = t1 * c, t3 = relu(t1),
# out = t2 + t3.
#
# Two real Triton programs compile the identical math:
#   - `diamond_fused_kernel`   -- ALL FOUR ops live inside ONE @triton.jit
#     function. a/b/c are loaded once each; t1/t2/t3/out only ever exist as
#     block-level SSA values (Triton's own tensor-typed registers), never
#     written to memory.
#   - `diamond_split_kernel_1`/`diamond_split_kernel_2` -- the SAME four ops,
#     split across two real kernels by hand, forcing t1 to be materialized to
#     a real intermediate buffer and read back -- the DELIBERATE control this
#     file uses to make "fusion is a language choice" concrete: nothing about
#     Triton's own compiler changed between the two versions, only how many
#     ops one function body happens to contain.
#
# Compiled with:   python3 "070_block_level_fusion_is_a_language_choice_not_a_compiler_pass.py"

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

TARGET = GPUTarget(backend="cuda", arch=80, warp_size=32)
BLOCK_SIZE = 1024


@triton.jit
def diamond_fused_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n_elements,
                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    t1 = a + b
    t2 = t1 * c
    t3 = tl.maximum(t1, 0.0)
    out = t2 + t3
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def diamond_split_kernel_1(a_ptr, b_ptr, t1_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    t1 = a + b
    tl.store(t1_ptr + offsets, t1, mask=mask)   # t1 forced into real memory


@triton.jit
def diamond_split_kernel_2(t1_ptr, c_ptr, out_ptr, n_elements,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    t1 = tl.load(t1_ptr + offsets, mask=mask)   # t1 read back from memory
    c = tl.load(c_ptr + offsets, mask=mask)
    t2 = t1 * c
    t3 = tl.maximum(t1, 0.0)
    out = t2 + t3
    tl.store(out_ptr + offsets, out, mask=mask)


def compile_kernel(fn, arg_names, ptr_args, extra_scalar_args, block_size):
    signature = {}
    for name in ptr_args:
        signature[name] = "*fp32"
    for name in extra_scalar_args:
        signature[name] = "i32"
    signature["BLOCK_SIZE"] = "constexpr"
    src = triton.compiler.ASTSource(fn=fn, signature=signature,
                                     constexprs={"BLOCK_SIZE": block_size})
    return triton.compile(src, target=TARGET)


def strip_source_locations(ir_text):
    return "\n".join(line for line in ir_text.splitlines()
                      if not line.strip().startswith("#loc"))


def count_ops(ttir_text, op_name):
    return sum(1 for line in ttir_text.splitlines() if f"{op_name} " in line
               or line.strip().startswith(op_name))


if __name__ == "__main__":
    fused = compile_kernel(diamond_fused_kernel,
                            ["a_ptr", "b_ptr", "c_ptr", "out_ptr", "n_elements"],
                            ["a_ptr", "b_ptr", "c_ptr", "out_ptr"], ["n_elements"],
                            BLOCK_SIZE)
    split1 = compile_kernel(diamond_split_kernel_1,
                             ["a_ptr", "b_ptr", "t1_ptr", "n_elements"],
                             ["a_ptr", "b_ptr", "t1_ptr"], ["n_elements"],
                             BLOCK_SIZE)
    split2 = compile_kernel(diamond_split_kernel_2,
                             ["t1_ptr", "c_ptr", "out_ptr", "n_elements"],
                             ["t1_ptr", "c_ptr", "out_ptr"], ["n_elements"],
                             BLOCK_SIZE)

    fused_ttir = strip_source_locations(fused.asm["ttir"])
    split1_ttir = strip_source_locations(split1.asm["ttir"])
    split2_ttir = strip_source_locations(split2.asm["ttir"])

    print("=== Real fused-kernel TTIR (path-normalized) ===")
    print(fused_ttir)

    print()
    print("=== Real split-kernel-1 TTIR (path-normalized) ===")
    print(split1_ttir)

    print()
    print("=== Real split-kernel-2 TTIR (path-normalized) ===")
    print(split2_ttir)

    fused_loads = count_ops(fused_ttir, "tt.load")
    fused_stores = count_ops(fused_ttir, "tt.store")
    split_loads = count_ops(split1_ttir, "tt.load") + count_ops(split2_ttir, "tt.load")
    split_stores = count_ops(split1_ttir, "tt.store") + count_ops(split2_ttir, "tt.store")

    print()
    print(f"Real tt.load count: fused kernel = {fused_loads} (a, b, c), "
          f"split kernels combined = {split_loads} (a, b, t1, c)")
    print(f"Real tt.store count: fused kernel = {fused_stores} (out), "
          f"split kernels combined = {split_stores} (t1, out)")
    print(f"Real kernel count: fused = 1 launch, split = 2 launches")

    # Chapter 12's own arithmetic-intensity accounting, applied to this real
    # Triton pair the same way Chapter 13 already applied it to CUDA
    # Hammer's own unfused-vs-fused IR for this identical graph.
    n = BLOCK_SIZE
    bytes_per_elem = 4  # float32
    fused_bytes = (3 + 1) * n * bytes_per_elem       # a,b,c in; out out
    split_bytes = (2 + 1) * n * bytes_per_elem + (2 + 1) * n * bytes_per_elem
    print()
    print(f"Real bytes moved per {n}-element block (Chapter 12's own accounting):")
    print(f"  fused:  {fused_bytes} bytes (a,b,c read once each, out written once, "
          f"t1/t2/t3 never leave registers)")
    print(f"  split:  {split_bytes} bytes (t1 additionally written by kernel 1 "
          f"AND read back by kernel 2 -- {split_bytes - fused_bytes} extra bytes, "
          f"{100.0 * (split_bytes - fused_bytes) / fused_bytes:.1f}% more traffic, "
          f"purely from materializing t1)")
