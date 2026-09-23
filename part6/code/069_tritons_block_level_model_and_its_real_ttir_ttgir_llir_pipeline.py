# Chapter 27, File 069: Triton's block-level programming model, and the real,
# multi-stage compiler pipeline that lowers it.
#
# Real toolchain investigation (throwaway, before this file was written):
#   - `pip install triton` pulls a real, per-architecture wheel (3.8.0 confirmed
#     installable on both this book's own machines: a genuine x86-64 wheel for
#     the cloud sandbox, a genuine aarch64 wheel for the device).
#   - This sandbox has no physical GPU (the same honest limitation since
#     Chapter 18). Calling `triton.runtime.driver.active` -- the object real
#     Triton normally uses to launch a kernel on real hardware -- raises
#     `RuntimeError: 0 active drivers ([]). There should only be one.`
#   - But `triton.compile()` accepts an explicit `target=` argument (a real
#     `triton.backends.compiler.GPUTarget(backend, arch, warp_size)`), which
#     bypasses live-driver detection entirely. Given a target, real Triton
#     compiles all the way through its own real pipeline -- TTIR, TTGIR, LLIR,
#     PTX, and even a real `cubin` -- with zero physical GPU involved, the
#     same "compile clean, cannot run" honesty Chapter 18 established for CUDA.
#   - Chapter 27 could not go further than Chapter 18 did (real interpreter-mode
#     EXECUTION on CPU): Triton's own `TRITON_INTERPRET=1` mode exists, but its
#     kernel arguments must be real `torch.Tensor` objects (the interpreter's
#     own `tl.load` expects a `.type` attribute a plain NumPy array does not
#     have). A real, working `torch` build could not be installed in this
#     sandbox in reasonable time: PyTorch's own CPU wheel index is not reachable
#     through this sandbox's proxy allowlist, and a plain PyPI `pip install
#     torch` timed out without completing. So, like Chapter 18's CUDA backend,
#     this chapter is honestly scoped to real AOT COMPILATION only -- never
#     real execution, and (new this chapter) not even CPU-interpreted execution.
#   - `@triton.jit`-decorated functions require `inspect.getsource()` to
#     succeed internally, exactly like TVMScript's `@I.ir_module` did in
#     Chapter 26 -- this file must exist as a real file on disk, never be
#     piped through a bash heredoc.
#
# Compiled with:   python3 "069_tritons_block_level_model_and_its_real_ttir_ttgir_llir_pipeline.py"

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This whole function body operates on one BLOCK of BLOCK_SIZE elements at
    # a time -- never on one scalar element, the way Chapter 18's own
    # `generateCudaElementwiseKernel()` (one CUDA thread per output element)
    # does. `pid` identifies WHICH block this program instance owns; every
    # tensor operation below (tl.load, +, tl.store) applies to the whole
    # block at once.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def compile_for_target(fn, signature, constexprs, target):
    src = triton.compiler.ASTSource(fn=fn, signature=signature, constexprs=constexprs)
    return triton.compile(src, target=target)


def strip_source_locations(ir_text):
    # Every one of Triton's own MLIR-based IR stages (TTIR, TTGIR) defines a
    # numbered `#locN = loc("...")` table at the bottom of the text, and each
    # real operation above only references those numbers (`loc(#loc18)`) --
    # so the ABSOLUTE PATH this book's own two real machines can never agree
    # on (their own paths to this same file differ) lives ONLY in the `#locN
    # = loc(...)` definition lines themselves, never in the operations that
    # reference them. Dropping just those definition lines leaves every real
    # operation, type, and instruction untouched for a structural comparison.
    kept = []
    for line in ir_text.splitlines():
        if line.strip().startswith("#loc"):
            continue
        kept.append(line)
    return "\n".join(kept)


if __name__ == "__main__":
    try:
        _ = triton.runtime.driver.active
        print("driver.active: unexpectedly succeeded (a real GPU is present)")
    except RuntimeError as e:
        print(f"driver.active real error (expected, no physical GPU): {e}")

    target = GPUTarget(backend="cuda", arch=80, warp_size=32)
    compiled = compile_for_target(
        vector_add_kernel,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
         "n_elements": "i32", "BLOCK_SIZE": "constexpr"},
        {"BLOCK_SIZE": 1024},
        target,
    )

    print()
    print("Real compiled pipeline stages:", list(compiled.asm.keys()))
    for stage in ["ttir", "ttgir", "llir", "ptx"]:
        print(f"  {stage}: {len(compiled.asm[stage])} real chars")
    print(f"  cubin: {len(compiled.asm['cubin'])} real bytes (a genuine compiled "
          f"binary for sm_80, produced with no physical GPU present)")

    print()
    print("=== Real TTIR (Triton IR -- one block-level op per line, path-normalized) ===")
    print(strip_source_locations(compiled.asm["ttir"]))

    print()
    print("=== Real TTGIR (Triton GPU IR -- excerpt: the auto-assigned thread layout) ===")
    ttgir_lines = compiled.asm["ttgir"].splitlines()
    print(ttgir_lines[0])   # the #blocked layout attribute itself
    print(ttgir_lines[6])   # module attributes line: num-warps/num-ctas/target
    print("  ... (remaining TTGIR lines carry the same real ops as TTIR above,")
    print("      now annotated with that #blocked layout on every tensor type)")

    print()
    print("=== Real PTX (excerpt: first 12 lines of a genuine NVPTX function) ===")
    for line in compiled.asm["ptx"].splitlines()[:12]:
        print(line)
