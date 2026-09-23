# Chapter 27: Triton's Block-Level Programming Model

Chapters 25 and 26 each studied a real, independent compiler that takes a
whole TENSOR GRAPH as input and decides, as a separate pass, how to group
it into kernels -- XLA's own `kind=kLoop` fusion, Relax's own
`AnnotateTIROpPattern`-driven `FuseOps`. Triton is a genuinely different
kind of system: it is not a graph compiler at all. A Triton program is a
single Python function, decorated `@triton.jit`, written by a human to
operate on one BLOCK of data at a time -- the compiler's job is to lower
that one block-level function down to real GPU machine code, never to
decide which operations belong together. `pip install triton` pulls a
real, per-architecture wheel (3.8.0, confirmed installable on both this
book's own machines: a genuine x86-64 wheel for the cloud sandbox, a
genuine aarch64 wheel for the device), so everything in this chapter is
real, installed Triton, inspected directly.

Real toolchain investigation, before any of this chapter's own files were
written: this sandbox has no physical GPU (the same honest limitation
since Chapter 18). `triton.runtime.driver.active` -- the object real
Triton normally uses to find and launch on real hardware -- raises a real
`RuntimeError: 0 active drivers ([]). There should only be one.` Chapter
18's own way around this (compile with nvcc, never run) has a real
Triton analogue: `triton.compile()` accepts an explicit `target=` (a real
`triton.backends.compiler.GPUTarget(backend, arch, warp_size)`), which
bypasses live-driver detection entirely and compiles all the way through
Triton's OWN real multi-stage pipeline -- TTIR, TTGIR, LLIR, PTX, and a
real `cubin` -- with zero physical GPU touched. This chapter could not
repeat Chapter 26's own trick of finding a CPU-only execution path,
either: Triton's real `TRITON_INTERPRET=1` mode exists, but its own
`tl.load` expects a real `torch.Tensor` (a plain NumPy array lacks the
`.type` attribute the interpreter's own semantics module reads), and a
working `torch` build could not be installed in this sandbox in
reasonable time (PyTorch's own CPU wheel index is not reachable through
this sandbox's proxy allowlist, and a plain PyPI `pip install torch`
timed out without completing) -- so, honestly, this chapter is scoped to
real AOT COMPILATION only, the same limitation Chapter 18 named for CUDA,
now without even CPU-interpreted execution as a fallback. And
`@triton.jit` functions need `inspect.getsource()` to succeed internally,
exactly like TVMScript's `@I.ir_module` did in Chapter 26 -- every file
in this chapter had to exist as a real file on disk from the start, never
piped through a heredoc.

```text
+------------------------------------------------------------------+
|  Chapter 27's own shape, section by section                      |
|                                                                    |
|  27.1  The block-level model itself, and Triton's own real,      |
|        FOUR-stage compiler pipeline (TTIR -> TTGIR -> LLIR ->     |
|        PTX), all produced with no physical GPU present.           |
|                                                                    |
|  27.2  The exact same Ch4/Ch13/Ch25/Ch26 diamond graph, proving   |
|        directly that fusion in Triton is a property of what the   |
|        PROGRAMMER writes in one kernel body, not a separate       |
|        compiler pass over a graph.                                |
|                                                                    |
|  27.3  Real triton.autotune/Config/Autotuner, laid next to        |
|        CUDA Hammer's own Schedule (Ch21) and TuningCache (Ch24).  |
+------------------------------------------------------------------+
```

## 27.1 The Block-Level Model, and Triton's Own Real Compiler Pipeline

Every CUDA kernel this book has generated since Chapter 18
(`generateCudaElementwiseKernel()`) assigns one GPU thread to one output
element -- the programmer (or, here, the code generator) reasons at the
SCALAR level, and the hardware's own SIMT execution model is what turns
that into parallel work. A Triton kernel is written at a different
grain entirely: `tl.program_id` identifies which BLOCK of `BLOCK_SIZE`
elements this program instance owns, and every operation inside the
function -- `tl.load`, `+`, `tl.store` -- is written once but operates on
the WHOLE block as a single tensor-typed value. Nothing in the source
text ever names an individual thread.

File 069 compiles a real block-level vector-add kernel with no physical
GPU present, using the explicit-`target=` technique above, and prints
every real stage of Triton's own compiler pipeline it produces:

```text
  source (Python)
      |
      +--  Triton's own AST-to-MLIR frontend
      |
  TTIR   (Triton IR: tt.load / tt.store / tt.make_range -- block-level,
          hardware-agnostic; %offsets is still a rank-1 tensor of 1024
          int32 elements)
      |
      +--  Triton's own layout-assignment pass
      |
  TTGIR  (Triton GPU IR: the SAME ops, now annotated with a REAL,
          compiler-chosen #blocked layout -- sizePerThread/threadsPerWarp/
          warpsPerCTA -- the thread-level plan the programmer never wrote)
      |
      +--  Triton's own MLIR-to-LLVM lowering
      |
  LLIR   (real LLVM IR, target-independent)
      |
      +--  LLVM's own NVPTX backend
      |
  PTX    (real NVIDIA assembly text, .target sm_80)
      |
      +--  ptxas (bundled with the wheel; no physical GPU needed to run it)
      |
  cubin  (a real compiled binary for sm_80)
```

```python
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
```

```bash
python3 "069_tritons_block_level_model_and_its_real_ttir_ttgir_llir_pipeline.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated Triton output):**

```text
driver.active real error (expected, no physical GPU): 0 active drivers ([]). There should only be one.

Real compiled pipeline stages: ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
  ttir: 3790 real chars
  ttgir: 4196 real chars
  llir: 10575 real chars
  ptx: 11228 real chars
  cubin: 9064 real bytes (a genuine compiled binary for sm_80, produced with no physical GPU present)

=== Real TTIR (Triton IR -- one block-level op per line, path-normalized) ===
module {
  tt.func public @vector_add_kernel(%x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc)), %y_ptr: !tt.ptr<f32> loc("y_ptr"(#loc)), %out_ptr: !tt.ptr<f32> loc("out_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %pid = tt.get_program_id x : i32 loc(#loc18)
    %block_start = arith.muli %pid, %c1024_i32 : i32 loc(#loc19)
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc20)
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32> loc(#loc21)
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32> loc(#loc21)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc22)
    %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<1024xi32> loc(#loc22)
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc23)
    %x_3 = tt.addptr %x, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc23)
    %x_4 = tt.load %x_3, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc25)
    %y_5 = tt.addptr %y, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc25)
    %y_6 = tt.load %y_5, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc26)
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc11)
    %1 = tt.addptr %0, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc11)
    %2 = arith.addf %x_4, %y_6 : tensor<1024xf32> loc(#loc12)
    tt.store %1, %2, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

=== Real TTGIR (Triton GPU IR -- excerpt: the auto-assigned thread layout) ===
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  ... (remaining TTGIR lines carry the same real ops as TTIR above,
      now annotated with that #blocked layout on every tensor type)

=== Real PTX (excerpt: first 12 lines of a genuine NVPTX function) ===
//
// Generated by LLVM NVPTX Back-End
//

.version 8.8
.target sm_80
.address_size 64

	// .globl	vector_add_kernel       // -- Begin function vector_add_kernel
                                        // @vector_add_kernel
.visible .entry vector_add_kernel(
	.param .u64 .ptr .global .align 1 vector_add_kernel_param_0,
```

**Output (device, aarch64 Linux VM -- real, live-generated Triton output):**

```text
driver.active real error (expected, no physical GPU): 0 active drivers ([]). There should only be one.

Real compiled pipeline stages: ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']
  ttir: 4401 real chars
  ttgir: 4807 real chars
  llir: 10622 real chars
  ptx: 11639 real chars
  cubin: 9192 real bytes (a genuine compiled binary for sm_80, produced with no physical GPU present)

=== Real TTIR (Triton IR -- one block-level op per line, path-normalized) ===
module {
  tt.func public @vector_add_kernel(%x_ptr: !tt.ptr<f32> loc("x_ptr"(#loc)), %y_ptr: !tt.ptr<f32> loc("y_ptr"(#loc)), %out_ptr: !tt.ptr<f32> loc("out_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %pid = tt.get_program_id x : i32 loc(#loc18)
    %block_start = arith.muli %pid, %c1024_i32 : i32 loc(#loc19)
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc20)
    %offsets_0 = tt.splat %block_start : i32 -> tensor<1024xi32> loc(#loc21)
    %offsets_1 = arith.addi %offsets_0, %offsets : tensor<1024xi32> loc(#loc21)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc22)
    %mask_2 = arith.cmpi slt, %offsets_1, %mask : tensor<1024xi32> loc(#loc22)
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc23)
    %x_3 = tt.addptr %x, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc23)
    %x_4 = tt.load %x_3, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc25)
    %y_5 = tt.addptr %y, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc25)
    %y_6 = tt.load %y_5, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc26)
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc11)
    %1 = tt.addptr %0, %offsets_1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc11)
    %2 = arith.addf %x_4, %y_6 : tensor<1024xf32> loc(#loc12)
    tt.store %1, %2, %mask_2 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

=== Real TTGIR (Triton GPU IR -- excerpt: the auto-assigned thread layout) ===
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  ... (remaining TTGIR lines carry the same real ops as TTIR above,
      now annotated with that #blocked layout on every tensor type)

=== Real PTX (excerpt: first 12 lines of a genuine NVPTX function) ===
//
// Generated by LLVM NVPTX Back-End
//

.version 8.8
.target sm_80
.address_size 64

	// .globl	vector_add_kernel       // -- Begin function vector_add_kernel
                                        // @vector_add_kernel
.visible .entry vector_add_kernel(
	.param .u64 .ptr .global .align 1 vector_add_kernel_param_0,
```

*Every real IR line agrees between the two machines -- the only differences are the raw character/byte COUNTS, which trace entirely to each machine's own longer or shorter absolute file path embedded in Triton's own debug-location table, discussed above.*


### What the compiler decided that the programmer never wrote

File 069's own kernel source never mentions a thread, a warp, or a
register. TTGIR's own `#blocked = #ttg.blocked<{sizePerThread = [1],
threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>` is where that
decision actually gets made -- a REAL, compiler-chosen mapping from the
block-level tensor down to 4 warps of 32 threads each, one element per
thread. Nothing this book's own Chapter 19 `Isa` abstraction did is
directly comparable: Chapter 19 hid the DIFFERENCE between two real
instruction sets (AVX2 vs. NEON) behind one generator, but the human
author still chose the vector width by hand. Here, the compiler chooses
the thread/warp/CTA layout itself, from nothing but the block size and
target -- a genuinely higher level of automation than anything CUDA
Hammer's own codegen (Ch17-20) or even TVM's own explicit
`s_tir.Schedule.split()`/`vectorize()` (Ch26.1) asked of their callers.

A real, verified structural finding closes this section: diffing the
cloud sandbox's own captured output against the device's own captured
output (both printed below) shows every real IR line agrees exactly --
TTIR, the TTGIR excerpt, and the PTX excerpt are BYTE-IDENTICAL between
the two machines. Only the raw character/byte COUNTS printed just above
those sections differ (`ttir: 3790 real chars` on the cloud sandbox vs.
`ttir: 4401 real chars` on the device), and that gap traces to exactly
one cause: each machine's own absolute path to this file is embedded in
Triton's own uncounted `#locN = loc("...")` debug-location table, and the
device's own path string is simply longer. This is the direct opposite
of Chapter 19's own real finding: AVX2 and NEON code genuinely differ
because the HOST CPU differs; Triton's own compiled GPU-target code does
NOT depend on the host CPU running the compiler at all, only on the
explicit `target=` passed to `triton.compile()` -- a real, structural
consequence of Triton targeting an external GPU architecture rather than
the machine it happens to compile on.

```bash
python3 "069_tritons_block_level_model_and_its_real_ttir_ttgir_llir_pipeline.py"
```

```python
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
```

```bash
python3 "070_block_level_fusion_is_a_language_choice_not_a_compiler_pass.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated Triton output):**

```text
=== Real fused-kernel TTIR (path-normalized) ===
module {
  tt.func public @diamond_fused_kernel(%a_ptr: !tt.ptr<f32> loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f32> loc("b_ptr"(#loc)), %c_ptr: !tt.ptr<f32> loc("c_ptr"(#loc)), %out_ptr: !tt.ptr<f32> loc("out_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %t3 = arith.constant dense<0.000000e+00> : tensor<1024xf32> loc(#loc23)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %pid = tt.get_program_id x : i32 loc(#loc24)
    %offsets = arith.muli %pid, %c1024_i32 : i32 loc(#loc25)
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc26)
    %offsets_1 = tt.splat %offsets : i32 -> tensor<1024xi32> loc(#loc25)
    %offsets_2 = arith.addi %offsets_1, %offsets_0 : tensor<1024xi32> loc(#loc25)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc27)
    %mask_3 = arith.cmpi slt, %offsets_2, %mask : tensor<1024xi32> loc(#loc27)
    %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc28)
    %a_4 = tt.addptr %a, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc28)
    %a_5 = tt.load %a_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc29)
    %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc30)
    %b_6 = tt.addptr %b, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc30)
    %b_7 = tt.load %b_6, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc31)
    %c = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc32)
    %c_8 = tt.addptr %c, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc32)
    %c_9 = tt.load %c_8, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc33)
    %t1 = arith.addf %a_5, %b_7 : tensor<1024xf32> loc(#loc34)
    %t2 = arith.mulf %t1, %c_9 : tensor<1024xf32> loc(#loc35)
    %t3_10 = arith.maxnumf %t1, %t3 : tensor<1024xf32> loc(#loc23)
    %out = arith.addf %t2, %t3_10 : tensor<1024xf32> loc(#loc36)
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc16)
    %1 = tt.addptr %0, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc16)
    tt.store %1, %out, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc17)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

=== Real split-kernel-1 TTIR (path-normalized) ===
module {
  tt.func public @diamond_split_kernel_1(%a_ptr: !tt.ptr<f32> loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f32> loc("b_ptr"(#loc)), %t1_ptr: !tt.ptr<f32> loc("t1_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %pid = tt.get_program_id x : i32 loc(#loc17)
    %offsets = arith.muli %pid, %c1024_i32 : i32 loc(#loc18)
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc19)
    %offsets_1 = tt.splat %offsets : i32 -> tensor<1024xi32> loc(#loc18)
    %offsets_2 = arith.addi %offsets_1, %offsets_0 : tensor<1024xi32> loc(#loc18)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc20)
    %mask_3 = arith.cmpi slt, %offsets_2, %mask : tensor<1024xi32> loc(#loc20)
    %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc21)
    %a_4 = tt.addptr %a, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc21)
    %a_5 = tt.load %a_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc22)
    %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc23)
    %b_6 = tt.addptr %b, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc23)
    %b_7 = tt.load %b_6, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %t1 = arith.addf %a_5, %b_7 : tensor<1024xf32> loc(#loc25)
    %0 = tt.splat %t1_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc11)
    %1 = tt.addptr %0, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc11)
    tt.store %1, %t1, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc12)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

=== Real split-kernel-2 TTIR (path-normalized) ===
module {
  tt.func public @diamond_split_kernel_2(%t1_ptr: !tt.ptr<f32> loc("t1_ptr"(#loc)), %c_ptr: !tt.ptr<f32> loc("c_ptr"(#loc)), %out_ptr: !tt.ptr<f32> loc("out_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %t3 = arith.constant dense<0.000000e+00> : tensor<1024xf32> loc(#loc19)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %pid = tt.get_program_id x : i32 loc(#loc20)
    %offsets = arith.muli %pid, %c1024_i32 : i32 loc(#loc21)
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc22)
    %offsets_1 = tt.splat %offsets : i32 -> tensor<1024xi32> loc(#loc21)
    %offsets_2 = arith.addi %offsets_1, %offsets_0 : tensor<1024xi32> loc(#loc21)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc23)
    %mask_3 = arith.cmpi slt, %offsets_2, %mask : tensor<1024xi32> loc(#loc23)
    %t1 = tt.splat %t1_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %t1_4 = tt.addptr %t1, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc24)
    %t1_5 = tt.load %t1_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc25)
    %c = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc26)
    %c_6 = tt.addptr %c, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc26)
    %c_7 = tt.load %c_6, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc27)
    %t2 = arith.mulf %t1_5, %c_7 : tensor<1024xf32> loc(#loc28)
    %t3_8 = arith.maxnumf %t1_5, %t3 : tensor<1024xf32> loc(#loc19)
    %out = arith.addf %t2, %t3_8 : tensor<1024xf32> loc(#loc29)
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc13)
    %1 = tt.addptr %0, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc13)
    tt.store %1, %out, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc14)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

Real tt.load count: fused kernel = 3 (a, b, c), split kernels combined = 4 (a, b, t1, c)
Real tt.store count: fused kernel = 1 (out), split kernels combined = 2 (t1, out)
Real kernel count: fused = 1 launch, split = 2 launches

Real bytes moved per 1024-element block (Chapter 12's own accounting):
  fused:  16384 bytes (a,b,c read once each, out written once, t1/t2/t3 never leave registers)
  split:  24576 bytes (t1 additionally written by kernel 1 AND read back by kernel 2 -- 8192 extra bytes, 50.0% more traffic, purely from materializing t1)
```

**Output (device, aarch64 Linux VM -- real, live-generated Triton output):**

```text
=== Real fused-kernel TTIR (path-normalized) ===
module {
  tt.func public @diamond_fused_kernel(%a_ptr: !tt.ptr<f32> loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f32> loc("b_ptr"(#loc)), %c_ptr: !tt.ptr<f32> loc("c_ptr"(#loc)), %out_ptr: !tt.ptr<f32> loc("out_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %t3 = arith.constant dense<0.000000e+00> : tensor<1024xf32> loc(#loc23)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %pid = tt.get_program_id x : i32 loc(#loc24)
    %offsets = arith.muli %pid, %c1024_i32 : i32 loc(#loc25)
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc26)
    %offsets_1 = tt.splat %offsets : i32 -> tensor<1024xi32> loc(#loc25)
    %offsets_2 = arith.addi %offsets_1, %offsets_0 : tensor<1024xi32> loc(#loc25)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc27)
    %mask_3 = arith.cmpi slt, %offsets_2, %mask : tensor<1024xi32> loc(#loc27)
    %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc28)
    %a_4 = tt.addptr %a, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc28)
    %a_5 = tt.load %a_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc29)
    %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc30)
    %b_6 = tt.addptr %b, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc30)
    %b_7 = tt.load %b_6, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc31)
    %c = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc32)
    %c_8 = tt.addptr %c, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc32)
    %c_9 = tt.load %c_8, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc33)
    %t1 = arith.addf %a_5, %b_7 : tensor<1024xf32> loc(#loc34)
    %t2 = arith.mulf %t1, %c_9 : tensor<1024xf32> loc(#loc35)
    %t3_10 = arith.maxnumf %t1, %t3 : tensor<1024xf32> loc(#loc23)
    %out = arith.addf %t2, %t3_10 : tensor<1024xf32> loc(#loc36)
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc16)
    %1 = tt.addptr %0, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc16)
    tt.store %1, %out, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc17)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

=== Real split-kernel-1 TTIR (path-normalized) ===
module {
  tt.func public @diamond_split_kernel_1(%a_ptr: !tt.ptr<f32> loc("a_ptr"(#loc)), %b_ptr: !tt.ptr<f32> loc("b_ptr"(#loc)), %t1_ptr: !tt.ptr<f32> loc("t1_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %pid = tt.get_program_id x : i32 loc(#loc17)
    %offsets = arith.muli %pid, %c1024_i32 : i32 loc(#loc18)
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc19)
    %offsets_1 = tt.splat %offsets : i32 -> tensor<1024xi32> loc(#loc18)
    %offsets_2 = arith.addi %offsets_1, %offsets_0 : tensor<1024xi32> loc(#loc18)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc20)
    %mask_3 = arith.cmpi slt, %offsets_2, %mask : tensor<1024xi32> loc(#loc20)
    %a = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc21)
    %a_4 = tt.addptr %a, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc21)
    %a_5 = tt.load %a_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc22)
    %b = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc23)
    %b_6 = tt.addptr %b, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc23)
    %b_7 = tt.load %b_6, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %t1 = arith.addf %a_5, %b_7 : tensor<1024xf32> loc(#loc25)
    %0 = tt.splat %t1_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc11)
    %1 = tt.addptr %0, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc11)
    tt.store %1, %t1, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc12)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

=== Real split-kernel-2 TTIR (path-normalized) ===
module {
  tt.func public @diamond_split_kernel_2(%t1_ptr: !tt.ptr<f32> loc("t1_ptr"(#loc)), %c_ptr: !tt.ptr<f32> loc("c_ptr"(#loc)), %out_ptr: !tt.ptr<f32> loc("out_ptr"(#loc)), %n_elements: i32 loc("n_elements"(#loc))) attributes {noinline = false} {
    %t3 = arith.constant dense<0.000000e+00> : tensor<1024xf32> loc(#loc19)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %pid = tt.get_program_id x : i32 loc(#loc20)
    %offsets = arith.muli %pid, %c1024_i32 : i32 loc(#loc21)
    %offsets_0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc22)
    %offsets_1 = tt.splat %offsets : i32 -> tensor<1024xi32> loc(#loc21)
    %offsets_2 = arith.addi %offsets_1, %offsets_0 : tensor<1024xi32> loc(#loc21)
    %mask = tt.splat %n_elements : i32 -> tensor<1024xi32> loc(#loc23)
    %mask_3 = arith.cmpi slt, %offsets_2, %mask : tensor<1024xi32> loc(#loc23)
    %t1 = tt.splat %t1_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc24)
    %t1_4 = tt.addptr %t1, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc24)
    %t1_5 = tt.load %t1_4, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc25)
    %c = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc26)
    %c_6 = tt.addptr %c, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc26)
    %c_7 = tt.load %c_6, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc27)
    %t2 = arith.mulf %t1_5, %c_7 : tensor<1024xf32> loc(#loc28)
    %t3_8 = arith.maxnumf %t1_5, %t3 : tensor<1024xf32> loc(#loc19)
    %out = arith.addf %t2, %t3_8 : tensor<1024xf32> loc(#loc29)
    %0 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc13)
    %1 = tt.addptr %0, %offsets_2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc13)
    tt.store %1, %out, %mask_3 : tensor<1024x!tt.ptr<f32>> loc(#loc14)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

Real tt.load count: fused kernel = 3 (a, b, c), split kernels combined = 4 (a, b, t1, c)
Real tt.store count: fused kernel = 1 (out), split kernels combined = 2 (t1, out)
Real kernel count: fused = 1 launch, split = 2 launches

Real bytes moved per 1024-element block (Chapter 12's own accounting):
  fused:  16384 bytes (a,b,c read once each, out written once, t1/t2/t3 never leave registers)
  split:  24576 bytes (t1 additionally written by kernel 1 AND read back by kernel 2 -- 8192 extra bytes, 50.0% more traffic, purely from materializing t1)
```

*Byte-for-byte identical between the two machines -- this file's own printed output never includes a raw IR character count, so nothing here is sensitive to the two machines' differing absolute paths.*


## 27.2 Fusion Is a Language-Level Choice, Not a Compiler Pass

Chapters 13 and 16 each built a real PASS -- `elementwiseFusionPass()`,
`boundedReductionFusionPass()` -- that walks a GRAPH and decides, node by
node, what gets grouped into one kernel. Chapters 25 and 26 found the
same shape in two independent real production compilers: XLA's own
`kind=kLoop` fusion and Relax's own `FuseOps`/`FuseTIR` both operate on a
graph IR, deciding fusion boundaries as a distinct compilation step
AFTER the program has already been expressed as a graph of separate
operations.

Triton has no such pass, because it has no such graph to run one over.
A `@triton.jit` function's own body IS the unit the compiler lowers --
whatever operations the programmer writes inside it become one kernel,
full stop. File 070 tests this directly on the exact same diamond graph
Chapters 4, 13, 25, and 26 have all already used (`t1 = a + b`, shared by
`t2 = t1 * c` and `t3 = relu(t1)`, joining at `out = t2 + t3`), written
two different ways:

```text
  diamond_fused_kernel            diamond_split_kernel_1 / _2
  (ONE @triton.jit function)      (the SAME four ops, split by hand)

  load a, b, c (3 loads)          kernel 1: load a, b (2 loads)
  t1 = a + b                                t1 = a + b
  t2 = t1 * c        [registers]            store t1        -- forced to
  t3 = relu(t1)       only, no                                 real memory
  out = t2 + t3        memory]     kernel 2: load t1, c (2 loads)
  store out (1 store)                       t2 = t1 * c
                                             t3 = relu(t1)
                                             out = t2 + t3
                                             store out (1 store)
```

Both versions are compiled with the identical `target=` technique from
27.1 -- no GPU needed for either. The real, printed TTIR for
`diamond_fused_kernel` contains all four arithmetic ops (`arith.addf`,
`arith.mulf`, `arith.maxnumf`, a second `arith.addf`) between its own
3 `tt.load`s and single `tt.store`, with `t1`, `t2`, and `t3` never once
appearing as an argument to `tt.addptr` -- they are pure SSA values,
exactly Chapter 13's own FusedStep register model, never round-tripped
through memory. The split version's own TWO separately compiled kernels
show `t1` crossing a REAL `tt.store`/`tt.load` boundary between them.

Chapter 12's own bytes-moved accounting, applied here exactly as
Chapter 13 already applied it to CUDA Hammer's own IR for this identical
graph, makes the cost of that boundary concrete: for a 1024-element
block, the fused kernel moves 16,384 bytes (`a`, `b`, `c` read once each,
`out` written once); the split version moves 24,576 bytes -- 50% more,
entirely from writing and re-reading `t1`. Nothing about Triton's own
compiler changed between the two runs; only how many real operations one
Python function happened to contain did. This is the chapter's own
central, honest structural finding: in a graph compiler (XLA, Relax,
CUDA Hammer itself), fusion is a COMPILER decision made after the fact;
in Triton, it is a PROGRAMMER decision made in advance, and the compiler
never gets a say in where the boundary falls.

```bash
python3 "070_block_level_fusion_is_a_language_choice_not_a_compiler_pass.py"
```

```python
# Chapter 27, File 071 (capstone): real Triton autotuning, laid directly next
# to CUDA Hammer's own real Schedule (Ch21), hybrid autotuner (Ch23), and
# TuningCache (Ch24) -- the same "no GPU here for a real autotuning RUN, so
# lay the real record types side by side" pattern Chapter 25.3 used for XLA's
# own TritonGemmKey/AutotuningLog and Chapter 26.3 used for MetaSchedule's own
# TuningRecord/Database. Every field and method name below is read directly
# from this sandbox's own real, installed `triton` package via
# `inspect.getsource()` -- cited from the real source, not fabricated -- but,
# like Chapter 25.3 and 26.3, no real search or real measurement runs here:
# Triton's own `Autotuner._bench()` ultimately needs
# `driver.active.get_benchmarker()`, which needs the same live GPU driver
# File 069 already found does not exist in this sandbox.
#
# Compiled with:   python3 "071_real_triton_autotune_config_search_next_to_cuda_hammers_own_schedule_and_tuningcache.py"

import inspect
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

TARGET = GPUTarget(backend="cuda", arch=80, warp_size=32)


# Real triton.Config objects -- direct analogue of Chapter 21's own
# Schedule{tileSizePerLoop, loopOrder, unrollFactor}: a named bundle of
# tunable meta-parameters, not a value the kernel body computes with.
REAL_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=REAL_CONFIGS, key=["n_elements"])
@triton.jit
def tuned_vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


if __name__ == "__main__":
    autotuner = tuned_vector_add_kernel
    print(f"Real Autotuner object: {type(autotuner).__name__}")
    print(f"  self.configs: {len(autotuner.configs)} real triton.Config objects")
    for cfg in autotuner.configs:
        print(f"    {cfg}")
    print(f"  self.keys (cache-key argument names): {autotuner.keys}")
    print(f"  self.cache (in-process memo, empty until first real call): "
          f"{autotuner.cache!r}")
    print(f"  self.cache_results (persistent on-disk cache enabled): "
          f"{autotuner.cache_results}")
    print(f"  self.perf_model (cost-model-based pruning before measurement): "
          f"{autotuner.perf_model}")
    print(f"  self.configs_top_k (fraction/count kept after perf_model ranks): "
          f"{autotuner.configs_top_k}")

    print()
    print("Real Autotuner.run()'s own cache-key construction (Ch24's own "
          "loopNestKeyForTarget(), read directly from installed source):")
    run_src_lines = inspect.getsource(type(autotuner).run).splitlines()
    for line in run_src_lines[1:9]:
        print(f"  {line}")

    print()
    print("Real Autotuner.prune_configs()'s own cost-model step (Ch22/23's "
          "own estimateScheduleCost()-then-measure shape, read directly "
          "from installed source):")
    prune_src_lines = inspect.getsource(type(autotuner).prune_configs).splitlines()
    for line in prune_src_lines[8:19]:
        print(f"  {line}")

    print()
    print("Real Autotuner.check_disk_cache()'s own persistent-cache write "
          "(Ch24's own TuningCache, read directly from installed source):")
    disk_src_lines = inspect.getsource(type(autotuner).check_disk_cache).splitlines()
    for line in disk_src_lines[-9:]:
        print(f"  {line}")

    # No GPU to run the real search, but every candidate Config's own kwargs
    # still compile through the exact same real AOT path File 069/070 used --
    # confirming the search space is a search space of real, independently
    # compilable kernels, the same "every candidate is a real compiled
    # program" discipline Chapter 23.1 established for CUDA Hammer's own
    # `generateScheduledElementwiseFunction()`.
    print()
    print("Real AOT compilation of every candidate Config (no GPU needed, "
          "same technique as Files 069/070):")
    base_fn = tuned_vector_add_kernel.fn  # the underlying @triton.jit function
    for cfg in REAL_CONFIGS:
        block_size = cfg.kwargs["BLOCK_SIZE"]
        src = triton.compiler.ASTSource(
            fn=base_fn,
            signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                       "n_elements": "i32", "BLOCK_SIZE": "constexpr"},
            constexprs={"BLOCK_SIZE": block_size},
        )
        compiled = triton.compile(src, target=TARGET)
        print(f"  BLOCK_SIZE={block_size:5d} num_warps={cfg.num_warps} "
              f"num_stages={cfg.num_stages}: compiled OK, "
              f"ptx={len(compiled.asm['ptx'])} chars, "
              f"cubin={len(compiled.asm['cubin'])} bytes")
```

```bash
python3 "071_real_triton_autotune_config_search_next_to_cuda_hammers_own_schedule_and_tuningcache.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated Triton output):**

```text
Real Autotuner object: Autotuner
  self.configs: 4 real triton.Config objects
    BLOCK_SIZE: 256, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
    BLOCK_SIZE: 512, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
    BLOCK_SIZE: 1024, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None
    BLOCK_SIZE: 2048, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None
  self.keys (cache-key argument names): ['n_elements']
  self.cache (in-process memo, empty until first real call): {}
  self.cache_results (persistent on-disk cache enabled): False
  self.perf_model (cost-model-based pruning before measurement): None
  self.configs_top_k (fraction/count kept after perf_model ranks): 1.0

Real Autotuner.run()'s own cache-key construction (Ch24's own loopNestKeyForTarget(), read directly from installed source):
          self.nargs = dict(zip(self.arg_names, args))
          used_cached_result = True
          if len(self.configs) > 1:
              all_args = {**self.nargs, **kwargs}
              _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
              key = [_args[key] for key in self.keys if key in _args]
              for _, arg in _args.items():
                  if hasattr(arg, "dtype"):

Real Autotuner.prune_configs()'s own cost-model step (Ch22/23's own estimateScheduleCost()-then-measure shape, read directly from installed source):
              top_k = self.configs_top_k
              if isinstance(top_k, float) and top_k <= 1.0:
                  top_k = int(len(self.configs) * top_k)
              elif not isinstance(top_k, int):
                  # Slice index must be an integer
                  raise TypeError("Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int")
  
              if len(pruned_configs) > top_k:
                  est_timing = {
                      config: self.perf_model(
                          **self.nargs,

Real Autotuner.check_disk_cache()'s own persistent-cache write (Ch24's own TuningCache, read directly from installed source):
          bench_fn()
          cache.put(
              json.dumps({
                  "key":
                  tuning_key,
                  "configs_timings":
                  [(config.__dict__, timings) for config, timings in self.configs_timings.items() if not config.pre_hook],
              }), file_name, binary=False)
          return False

Real AOT compilation of every candidate Config (no GPU needed, same technique as Files 069/070):
  BLOCK_SIZE=  256 num_warps=4 num_stages=2: compiled OK, ptx=7592 chars, cubin=6120 bytes
  BLOCK_SIZE=  512 num_warps=4 num_stages=2: compiled OK, ptx=9081 chars, cubin=7016 bytes
  BLOCK_SIZE= 1024 num_warps=8 num_stages=3: compiled OK, ptx=12135 chars, cubin=9320 bytes
  BLOCK_SIZE= 2048 num_warps=8 num_stages=3: compiled OK, ptx=18475 chars, cubin=14184 bytes
```

**Output (device, aarch64 Linux VM -- real, live-generated Triton output):**

```text
Real Autotuner object: Autotuner
  self.configs: 4 real triton.Config objects
    BLOCK_SIZE: 256, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
    BLOCK_SIZE: 512, num_warps: 4, num_ctas: 1, num_stages: 2, maxnreg: None
    BLOCK_SIZE: 1024, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None
    BLOCK_SIZE: 2048, num_warps: 8, num_ctas: 1, num_stages: 3, maxnreg: None
  self.keys (cache-key argument names): ['n_elements']
  self.cache (in-process memo, empty until first real call): {}
  self.cache_results (persistent on-disk cache enabled): False
  self.perf_model (cost-model-based pruning before measurement): None
  self.configs_top_k (fraction/count kept after perf_model ranks): 1.0

Real Autotuner.run()'s own cache-key construction (Ch24's own loopNestKeyForTarget(), read directly from installed source):
          self.nargs = dict(zip(self.arg_names, args))
          used_cached_result = True
          if len(self.configs) > 1:
              all_args = {**self.nargs, **kwargs}
              _args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
              key = [_args[key] for key in self.keys if key in _args]
              for _, arg in _args.items():
                  if hasattr(arg, "dtype"):

Real Autotuner.prune_configs()'s own cost-model step (Ch22/23's own estimateScheduleCost()-then-measure shape, read directly from installed source):
              top_k = self.configs_top_k
              if isinstance(top_k, float) and top_k <= 1.0:
                  top_k = int(len(self.configs) * top_k)
              elif not isinstance(top_k, int):
                  # Slice index must be an integer
                  raise TypeError("Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int")

              if len(pruned_configs) > top_k:
                  est_timing = {
                      config: self.perf_model(
                          **self.nargs,

Real Autotuner.check_disk_cache()'s own persistent-cache write (Ch24's own TuningCache, read directly from installed source):
          bench_fn()
          cache.put(
              json.dumps({
                  "key":
                  tuning_key,
                  "configs_timings":
                  [(config.__dict__, timings) for config, timings in self.configs_timings.items() if not config.pre_hook],
              }), file_name, binary=False)
          return False

Real AOT compilation of every candidate Config (no GPU needed, same technique as Files 069/070):
  BLOCK_SIZE=  256 num_warps=4 num_stages=2: compiled OK, ptx=8003 chars, cubin=6248 bytes
  BLOCK_SIZE=  512 num_warps=4 num_stages=2: compiled OK, ptx=9492 chars, cubin=7144 bytes
  BLOCK_SIZE= 1024 num_warps=8 num_stages=3: compiled OK, ptx=12546 chars, cubin=9320 bytes
  BLOCK_SIZE= 2048 num_warps=8 num_stages=3: compiled OK, ptx=18886 chars, cubin=14184 bytes
```

*Agree on every real field, method excerpt, and compiled-candidate count; the compiled PTX/cubin sizes differ slightly for the same embedded-path reason as File 069, not from any real difference in what each candidate Config compiles to.*


## 27.3 Real triton.autotune, Next to CUDA Hammer's Own Schedule and TuningCache

No physical GPU means no real autotuning RUN here either -- the same
honest gap Chapter 25.3 hit for XLA's own autotuner and Chapter 26.3 hit
for MetaSchedule (Triton's own `Autotuner._bench()` ultimately calls
`driver.active.get_benchmarker()`, which needs the exact live driver File
069 already found does not exist). So, as in both of those closing
sections, File 071 lays the real types and real algorithm directly next
to CUDA Hammer's own, using field and method names read straight out of
this sandbox's own installed `triton` package via `inspect.getsource()`.

`triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3)` is the
direct analogue of Chapter 21's own `Schedule{tileSizePerLoop, loopOrder,
unrollFactor}` -- a named bundle of tunable meta-parameters, not a value
the kernel computes with. `@triton.autotune(configs=[...],
key=["n_elements"])` wraps a `@triton.jit` function in a real
`Autotuner` object, and its own real `run()` method (printed directly
from source in File 071's own output) builds a cache key from exactly
the argument VALUES named in `key`, plus each argument's own dtype --
the same (workload, target) shape as Chapter 24.3's own
`loopNestKeyForTarget()`, arrived at independently by a real, unrelated
production system. A cache MISS calls `prune_configs()`, which (when a
real `perf_model` callback is supplied) estimates every candidate's cost
and keeps only the cheapest `configs_top_k` before real measurement --
structurally identical to Chapter 23.3's own hybrid autotuner
(`estimateScheduleCost()` ranks all 128 real schedules, keeps the
cheapest `topK=8`, only THOSE get real timed measurement). A HIT returns
`self.cache[key]` in-process, exactly Chapter 24.1's own `TuningCache`
MISS-then-HIT shape -- except Triton's own default cache lives only for
the life of the Python process. Passing `cache_results=True` turns on
`check_disk_cache()`, a REAL persistent cache: a SHA256 hash of the
Triton version, the compiler backend, the function's own source hash,
relevant environment variables, and the tuning key becomes a file name,
and that file holds real JSON -- `{"key": ..., "configs_timings": [(config
dict, measured time), ...]}` for EVERY candidate measured, not just the
winner. This is a real, substantive difference from Chapter 24's own
`TuningCache`, which persists only the single winning `Schedule` as a
plain-text line keyed by a human-readable `loopNestKey()` string: Triton's
own real cache keeps the whole measured ranking, keyed by an opaque hash
built from the exact software environment that produced it.

File 071 closes by compiling all four of its own real `Config` candidates
through the same explicit-`target=` AOT path as 27.1 and 27.2 -- no GPU,
but a real, independently compiled kernel per candidate, confirming (the
same discipline as Chapter 23.1's own `generateScheduledElementwiseFunction()`)
that a real search space is a search space of real, individually
compilable programs, not an abstraction that only makes sense once
hardware is attached.

```bash
python3 "071_real_triton_autotune_config_search_next_to_cuda_hammers_own_schedule_and_tuningcache.py"
```
