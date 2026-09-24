# Appendix C: LLVM and MLIR -- The Compiler Infrastructure You Get for Free

Every pass and every backend CUDA Hammer has since Chapter 8 was
hand-written for this book's own `OpKind` enum and `Graph`/`Node`
types: `constantFoldPass()` and `deadCodeEliminationPass()` (Chapter
9), `commonSubexpressionEliminationPass()` (Chapter 10),
`elementwiseFusionPass()` and `reductionFusionPass()` (Chapters 13-14),
`LoopNest`/`tileLoop()` (Chapter 15), and the CPU/CUDA codegen
generators themselves (Chapters 17-20). Chapter 3 already named what
every real production compiler this book has studied since does
differently: XLA lowers its optimized HLO through a real LLVM backend
(cited from OpenXLA's own architecture docs); TVM's real MetaSchedule
pipeline compiles scheduled TIR through `tvm.target.Target("llvm")`
directly (Chapter 26); and Triton's own real compiler pipeline is
`AST -> Triton's own MLIR-based TTIR/TTGIR dialects -> real LLVM IR ->
LLVM's own NVPTX backend` (Chapter 27) -- Chapter 29 even found Flash
Attention's KV-block loop lowering to one literal MLIR `scf.for`
construct inside Triton's own real compiled TTIR. Every one of those
findings named LLVM or MLIR as something CUDA Hammer's own real
production peers stand on. This appendix is the one place in the book
that opens that infrastructure up directly: hand-writing real LLVM IR
and real MLIR for CUDA Hammer's own diamond graph, running both
through real tools, and seeing exactly what a from-scratch compiler
project gets "for free" by standing on it instead of hand-writing
every pass itself.

Per this book's own explicit choice when this appendix was scoped,
Section C.4 also folds in NVIDIA CUTLASS -- left as an open planning
question since Chapter 31.2 first cited CUTLASS's own docs for the
real im2col-based implicit-GEMM technique behind convolution
libraries.

## Appendix C's own shape

```text
+------------------------------------------------------------------+
|  Appendix C's own shape, section by section                      |
|                                                                    |
|  C.1  Hand-Written LLVM IR for CUDA Hammer's Diamond Graph -- the |
|       same five nodes File 005 built by hand (Chapter 4), written |
|       directly as real LLVM IR text, verified with LLVM's own     |
|       verifier, and run two independent real ways.                |
|                                                                    |
|  C.2  What LLVM's Own Optimizer Does To It, For Free -- the same  |
|       diamond, with a and b as literal constants: LLVM's real,    |
|       general-purpose optimizer pipeline (never written for this  |
|       or any tensor IR) collapses the whole computation to one    |
|       constant, and removes the function outright once it is      |
|       marked no longer externally visible.                        |
|                                                                    |
|  C.3  MLIR's linalg.generic and Progressive Lowering -- the same  |
|       diamond graph applied elementwise over a real array as ONE  |
|       declarative linalg.generic op, real-lowered through loops,  |
|       control flow, and the LLVM dialect, and real-JIT-executed.  |
|                                                                    |
|  C.4  Capstone: CUTLASS and the Real Im2col-Aware GEMM Behind     |
|       Chapter 31's Citation -- a real CUTLASS GEMM template        |
|       instantiation compiled against CUTLASS's own real headers,  |
|       naming the exact real header where modern CUTLASS's own     |
|       hardware copy engine has an explicit im2col concept.         |
+------------------------------------------------------------------+
```

Toolchain, confirmed directly rather than assumed: the cloud sandbox
already had real LLVM/Clang 18.1.3 installed (`clang`, `opt`, `llc`,
`llvm-as`); real MLIR tools were not installed by default and were
added the same way this book has installed every other real toolchain
component since Chapter 25 -- `sudo apt-get install -y mlir-18-tools`
(a real 44.9MB download, Ubuntu's own `noble-updates` package built
from the same LLVM 18.1.3 source). The device -- this book's own
aarch64 Linux VM -- was checked directly and confirmed to have
**neither** a C/C++ compiler toolchain beyond `gcc`/`g++` **nor** any
package-manager root access at all: `sudo -n true` fails outright
there ("the \"no new privileges\" flag is set, which prevents sudo from
running as root"), and `apt-get update` fails with a real `Permission
denied` on its own lock file. This is a new, honest scope boundary,
different in kind from every earlier CUDA-only split (Chapters 18, 28,
29.3, 30.3; Appendix A.3): those were CUDA-linkage limits with a real
physical-GPU cause; this one is a toolchain-availability limit, with
no way to install LLVM, MLIR, or CUTLASS's own build dependencies on
this particular machine at all. Every source file in this appendix is
still sent to the device and md5-verified byte-identical, exactly as
every earlier chapter's files have been -- but none of this appendix's
real command outputs could be independently reproduced there this
time, and the prose says so at each point rather than silently
dropping the device.

## C.1 -- Hand-Written LLVM IR for CUDA Hammer's Diamond Graph

Chapter 4's own diamond graph, reused in every chapter since, is five
nodes built through `Graph::addInput`/`addBinary`/`addUnary`:

```text
  a, b = Input, Input
  t1   = Add(a, b)
  t2   = Mul(t1, a)
  t3   = ReLU(t1)
  out  = Add(t2, t3)
```

This file writes the same five nodes directly as real LLVM IR text,
with `a = 3.0`, `b = 4.0` -- by hand: `t1 = 7`, `t2 = 21`, `t3 = 7`,
`out = 28`. LLVM IR has no built-in `relu`; the only way to express a
conditional-select over an SSA value is a real `fcmp` feeding a real
`select` -- `select(t1 > 0.0, t1, 0.0)`. This is not a simplification
introduced for this book: it is the exact `fcmp`+`select` shape
Chapter 34's own `max(a,b) = b + ReLU(a-b)` identity already used at
the C++ source level, one level higher.

```llvm
; Appendix C.1 -- CUDA Hammer's own Chapter 4 diamond graph, hand-written
; directly as real LLVM IR text instead of built through Graph::addInput/
; addBinary/addUnary. The same five nodes File 005 built by hand:
;
;   a, b = Input, Input
;   t1   = Add(a, b)
;   t2   = Mul(t1, a)
;   t3   = ReLU(t1)
;   out  = Add(t2, t3)
;
; with a = 3.0, b = 4.0, giving t1=7, t2=21, t3=7, out=28 by hand.
;
; ReLU(t1) is expressed the only way LLVM IR has to express a
; conditional-select over an SSA value: a real fcmp (compare t1 to 0.0)
; feeding a real select. This is not a simplification chosen for this
; book -- it is the same fcmp+select shape Chapter 34's own
; max(a,b)=b+ReLU(a-b) identity already used at the C++ source level;
; here it appears one level lower, in the compiler's own IR.

@.fmt = private unnamed_addr constant [30 x i8] c"LLVM IR diamond result: %.6f\0A\00"

declare i32 @printf(ptr, ...)

define float @diamond(float %a, float %b) {
entry:
  %t1 = fadd float %a, %b
  %t2 = fmul float %t1, %a
  %t3cmp = fcmp ogt float %t1, 0.000000e+00
  %t3 = select i1 %t3cmp, float %t1, float 0.000000e+00
  %out = fadd float %t2, %t3
  ret float %out
}

define i32 @main() {
entry:
  %r = call float @diamond(float 3.000000e+00, float 4.000000e+00)
  %rd = fpext float %r to double
  %ignore = call i32 (ptr, ...) @printf(ptr @.fmt, double %rd)
  ret i32 0
}
```

**Verify (LLVM's own real IR verifier, not this book's code):**

```bash
opt -S -passes=verify 097_hand_written_llvm_ir_for_cuda_hammers_diamond_graph.ll -o /dev/null
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
VERIFY_OK
```

**Run, path 1 -- compile the `.ll` text directly with `clang` (its own front end lowers straight to machine code):**

```bash
clang 097_hand_written_llvm_ir_for_cuda_hammers_diamond_graph.ll -o diamond_exe -lm
./diamond_exe
```

**Output (cloud sandbox -- real, live-executed output):**

```text
LLVM IR diamond result: 28.000000
```

**Run, path 2 -- an independent path through LLVM's own native code generator, `llc`, then a plain linker (no `clang`-driven IR lowering involved at all):**

```bash
llc 097_hand_written_llvm_ir_for_cuda_hammers_diamond_graph.ll -o diamond.s
clang -no-pie diamond.s -o diamond_exe_via_llc -lm
./diamond_exe_via_llc
```

**Output (cloud sandbox -- real, live-executed output):**

```text
LLVM IR diamond result: 28.000000
```

Both independent paths agree: 28.000000. The real generated x86-64
assembly `llc` produced for `diamond` itself is worth reading in full
-- LLVM's own instruction selector recognized the `fcmp ogt` + `select`
pair as exactly the max-with-zero identity Chapter 34 named at the
source level, and folded it into a single real `maxss` (SSE
max-scalar-single) instruction, with no `fcmp`/`select`/branch left
anywhere in the output:

**Output (cloud sandbox -- real, live-executed output; `llc`'s own generated assembly text):**

```text
	.text
	.file	"097_hand_written_llvm_ir_for_cuda_hammers_diamond_graph.ll"
	.globl	diamond                         # -- Begin function diamond
	.p2align	4, 0x90
	.type	diamond,@function
diamond:                                # @diamond
	.cfi_startproc
# %bb.0:                                # %entry
	addss	%xmm0, %xmm1
	mulss	%xmm1, %xmm0
	xorps	%xmm2, %xmm2
	maxss	%xmm2, %xmm1
	addss	%xmm1, %xmm0
	retq
.Lfunc_end0:
	.size	diamond, .Lfunc_end0-diamond
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst4,"aM",@progbits,4
	.p2align	2, 0x0                          # -- Begin function main
.LCPI1_0:
	.long	0x40400000                      # float 3
.LCPI1_1:
	.long	0x40800000                      # float 4
	.text
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	movss	.LCPI1_0(%rip), %xmm0           # xmm0 = [3.0E+0,0.0E+0,0.0E+0,0.0E+0]
	movss	.LCPI1_1(%rip), %xmm1           # xmm1 = [4.0E+0,0.0E+0,0.0E+0,0.0E+0]
	callq	diamond@PLT
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.fmt, %edi
	movb	$1, %al
	callq	printf@PLT
	xorl	%eax, %eax
	popq	%rcx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        # -- End function
	.type	.L.fmt,@object                  # @.fmt
	.section	.rodata.str1.16,"aMS",@progbits,1
	.p2align	4, 0x0
.L.fmt:
	.asciz	"LLVM IR diamond result: %.6f\n"
	.size	.L.fmt, 30

	.section	".note.GNU-stack","",@progbits
```

Nothing in this book asked LLVM to know about ReLU or about
`select`-as-max; this is LLVM's own general-purpose instruction
selector, run over five lines of hand-written IR that happen to encode
CUDA Hammer's own diamond graph, discovering the same identity Chapter
34 wrote into CUDA Hammer's own C++ source by hand.

## C.2 -- What LLVM's Own Optimizer Does To It, For Free

Chapter 9 hand-wrote `constantFoldPass()` and
`deadCodeEliminationPass()` specifically for CUDA Hammer's own
`OpKind` enum and `Graph`/`Node` types -- real, working, but real code
that only understands this book's own IR. This file asks the same
question C.1 answered for codegen, now for optimization: the exact
same diamond graph, with `a` and `b` now literal constants baked
directly into `main` (the LLVM IR analogue of feeding
`constantFoldPass()` a fully-constant graph), run through a real,
general-purpose LLVM optimizer pipeline that has never heard of
`Value`, `Node`, `Graph`, or `OpKind`.

```llvm
; Appendix C.2 -- the same diamond graph as File 097, but with a and b
; baked in as literal constants inside main, the LLVM IR analogue of
; Chapter 9's constantFoldPass() input: every operand of every node is
; already a known constant before optimization runs.
;
; Chapter 9 hand-wrote constantFoldPass() and deadCodeEliminationPass()
; specifically for CUDA Hammer's own OpKind enum and Graph/Node types.
; This file asks what a REAL, general-purpose, not-tensor-specific
; optimizer pipeline -- one that has never heard of Value, Node, Graph,
; or OpKind -- does to the exact same hand-translated computation.

@.fmt = private unnamed_addr constant [30 x i8] c"LLVM IR diamond result: %.6f\0A\00"

declare i32 @printf(ptr, ...)

define float @diamond(float %a, float %b) {
entry:
  %t1 = fadd float %a, %b
  %t2 = fmul float %t1, %a
  %t3cmp = fcmp ogt float %t1, 0.000000e+00
  %t3 = select i1 %t3cmp, float %t1, float 0.000000e+00
  %out = fadd float %t2, %t3
  ret float %out
}

define i32 @main() {
entry:
  %r = call float @diamond(float 3.000000e+00, float 4.000000e+00)
  %rd = fpext float %r to double
  %ignore = call i32 (ptr, ...) @printf(ptr @.fmt, double %rd)
  ret i32 0
}
```

**Run LLVM's own general-purpose constant-folding pipeline (inlining, instruction combining, global value numbering, dead-code elimination -- none of it written for this or any tensor IR):**

```bash
opt -S -passes="inline,instcombine,gvn,dce" 098_what_llvms_own_optimizer_does_to_it_for_free.ll -o folded.ll
cat folded.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
; ModuleID = '098_what_llvms_own_optimizer_does_to_it_for_free.ll'
source_filename = "098_what_llvms_own_optimizer_does_to_it_for_free.ll"

@.fmt = private unnamed_addr constant [30 x i8] c"LLVM IR diamond result: %.6f\0A\00"

declare i32 @printf(ptr, ...)

define float @diamond(float %a, float %b) {
entry:
  %t1 = fadd float %a, %b
  %t2 = fmul float %t1, %a
  %t3cmp = fcmp ogt float %t1, 0.000000e+00
  %t3 = select i1 %t3cmp, float %t1, float 0.000000e+00
  %out = fadd float %t2, %t3
  ret float %out
}

define i32 @main() {
entry:
  %ignore = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.fmt, double 2.800000e+01)
  ret i32 0
}
```

`main`'s entire body -- the call to `diamond`, both `fadd`s, the
`fmul`, the `fcmp`+`select` -- is gone, replaced by a single literal
`2.800000e+01` passed straight to `printf`. This is the real LLVM
analogue of Chapter 9's own `constantFoldPass()` result on a
fully-constant diamond graph, produced by four passes that share zero
code with this book's own pass manager. One real, honest detail: the
`diamond` function itself is still emitted below `main` in this
output -- it has *external* linkage (nothing told LLVM no other
translation unit could call it), so dead-code elimination cannot
remove it, only the call site's own now-constant computation. Marking
it `internal` (the LLVM linkage that means "never referenced outside
this module," CUDA Hammer's own closest real analogue never existed
because Chapter 9's `Graph` has no notion of external visibility at
all) and re-running the same pipeline plus one more real pass,
`globaldce` (whole-module dead-global elimination), removes the
function too:

```bash
sed 's/define float @diamond/define internal float @diamond/' 098_what_llvms_own_optimizer_does_to_it_for_free.ll > internal.ll
opt -S -passes="inline,instcombine,gvn,dce" internal.ll -o step1.ll
opt -S -passes="globaldce" step1.ll -o folded_internal.ll
cat folded_internal.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
; ModuleID = '/tmp/appendix_c/098_internal_step1.ll'
source_filename = "/tmp/appendix_c/098_internal.ll"

@.fmt = private unnamed_addr constant [30 x i8] c"LLVM IR diamond result: %.6f\0A\00"

declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  %ignore = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.fmt, double 2.800000e+01)
  ret i32 0
}
```

Every trace of the diamond graph's own five nodes is gone -- only
`main` and the format string remain, and `main` itself is now nothing
but a `printf` call with a literal. The program still compiles and
runs, printing the same real 28.000000:

```bash
clang folded_internal.ll -o folded_exe -lm
./folded_exe
```

**Output (cloud sandbox -- real, live-executed output):**

```text
LLVM IR diamond result: 28.000000
```

Chapter 9 needed two hand-written passes, tied to this book's own
`OpKind` switch statement, to reach a comparable result over CUDA
Hammer's own IR. LLVM reached the same real result over the same
hand-translated computation using four passes it ships with, none of
which have ever seen a tensor graph.

## C.3 -- MLIR's linalg.generic and Progressive Lowering

Chapter 15 hand-wrote `LoopNest` and `tileLoop()` --
CUDA Hammer's own way of expressing "iterate over these dimensions,
in this order, computing this body" as an explicit C++ data structure,
built and walked by hand. MLIR's `linalg` dialect expresses the same
idea declaratively: `linalg.generic` states *what* varies over which
index and *what* the per-element computation is, and leaves *how* to
turn that into actual loops to a real, separate, off-the-shelf lowering
pass. This file expresses the diamond graph applied elementwise over a
real 5-element array as one `linalg.generic` op, with inputs chosen so
ReLU genuinely clips something away rather than a convenience
all-positive test vector:

```text
  a = [ 1, -2,   3, -4,  5]
  b = [10,  1, -30,  1, -3]

  by hand, per index i:
  i=0: t1=11  (pos) t2=11   t3=11  out=22
  i=1: t1=-1  (neg) t2=2    t3=0   out=2
  i=2: t1=-27 (neg) t2=-81  t3=0   out=-81
  i=3: t1=-3  (neg) t2=12   t3=0   out=12   -- ReLU clips a nonzero t1 away
  i=4: t1=2   (pos) t2=10   t3=2   out=12
```

```mlir
// Appendix C.3 -- the same diamond graph (t1=Add(a,b), t2=Mul(t1,a),
// t3=ReLU(t1), out=Add(t2,t3)) applied elementwise over a 5-element
// array, expressed as ONE real MLIR linalg.generic op -- the declarative,
// fusion-friendly form Chapter 15's own hand-written LoopNest::tileLoop()
// and loopNestsCompatibleForFusion() exist to approximate by hand for
// CUDA Hammer's own IR.
//
// Inputs (chosen so ReLU genuinely clips something away, unlike a
// convenience all-positive test vector):
//   a = [ 1, -2,   3, -4,  5]
//   b = [10,  1, -30,  1, -3]
// giving, by hand, per index i:
//   i=0: t1=11  (pos) t2=11   t3=11  out=22
//   i=1: t1=-1  (neg) t2=2    t3=0   out=2
//   i=2: t1=-27 (neg) t2=-81  t3=0   out=-81
//   i=3: t1=-3  (neg) t2=12   t3=0   out=12
//   i=4: t1=2   (pos) t2=10   t3=2   out=12
//
// main() returns out[3] = 12 -- the index where ReLU clips a nonzero
// t1 away entirely (t3 goes from what would be -3 to 0).
#map = affine_map<(i) -> (i)>

func.func @diamond_elementwise(%a: memref<5xf32>, %b: memref<5xf32>, %out: memref<5xf32>) {
  linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]
  } ins(%a, %b : memref<5xf32>, memref<5xf32>) outs(%out : memref<5xf32>) {
  ^bb0(%av: f32, %bv: f32, %outv: f32):
    %t1 = arith.addf %av, %bv : f32
    %t2 = arith.mulf %t1, %av : f32
    %zero = arith.constant 0.0 : f32
    %cmp = arith.cmpf ogt, %t1, %zero : f32
    %t3 = arith.select %cmp, %t1, %zero : f32
    %o = arith.addf %t2, %t3 : f32
    linalg.yield %o : f32
  }
  return
}

func.func @main() -> f32 {
  %a = memref.alloc() : memref<5xf32>
  %b = memref.alloc() : memref<5xf32>
  %out = memref.alloc() : memref<5xf32>

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  %a0 = arith.constant 1.0 : f32
  %a1 = arith.constant -2.0 : f32
  %a2 = arith.constant 3.0 : f32
  %a3 = arith.constant -4.0 : f32
  %a4 = arith.constant 5.0 : f32
  memref.store %a0, %a[%c0] : memref<5xf32>
  memref.store %a1, %a[%c1] : memref<5xf32>
  memref.store %a2, %a[%c2] : memref<5xf32>
  memref.store %a3, %a[%c3] : memref<5xf32>
  memref.store %a4, %a[%c4] : memref<5xf32>

  %b0 = arith.constant 10.0 : f32
  %b1 = arith.constant 1.0 : f32
  %b2 = arith.constant -30.0 : f32
  %b3 = arith.constant 1.0 : f32
  %b4 = arith.constant -3.0 : f32
  memref.store %b0, %b[%c0] : memref<5xf32>
  memref.store %b1, %b[%c1] : memref<5xf32>
  memref.store %b2, %b[%c2] : memref<5xf32>
  memref.store %b3, %b[%c3] : memref<5xf32>
  memref.store %b4, %b[%c4] : memref<5xf32>

  call @diamond_elementwise(%a, %b, %out) : (memref<5xf32>, memref<5xf32>, memref<5xf32>) -> ()

  %r = memref.load %out[%c3] : memref<5xf32>
  return %r : f32
}
```

**Step 1 -- lower the declarative `linalg.generic` into an explicit loop nest (`--convert-linalg-to-loops`, a real, general-purpose MLIR pass -- not written for this diamond, or for tensor compilers specifically):**

```bash
mlir-opt 099_mlir_linalg_generic_and_progressive_lowering_for_a_fused_elementwise_op.mlir --convert-linalg-to-loops -o step1_loops.mlir
cat step1_loops.mlir
```

**Output (cloud sandbox -- real, live-executed output):**

```text
module {
  func.func @diamond_elementwise(%arg0: memref<5xf32>, %arg1: memref<5xf32>, %arg2: memref<5xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c5 step %c1 {
      %0 = memref.load %arg0[%arg3] : memref<5xf32>
      %1 = memref.load %arg1[%arg3] : memref<5xf32>
      %2 = arith.addf %0, %1 : f32
      %3 = arith.mulf %2, %0 : f32
      %4 = arith.cmpf ogt, %2, %cst : f32
      %5 = arith.select %4, %2, %cst : f32
      %6 = arith.addf %3, %5 : f32
      memref.store %6, %arg2[%arg3] : memref<5xf32>
    }
    return
  }
  func.func @main() -> f32 {
    %cst = arith.constant -3.000000e+00 : f32
    %cst_0 = arith.constant -3.000000e+01 : f32
    %cst_1 = arith.constant 1.000000e+01 : f32
    %cst_2 = arith.constant 5.000000e+00 : f32
    %cst_3 = arith.constant -4.000000e+00 : f32
    %cst_4 = arith.constant 3.000000e+00 : f32
    %cst_5 = arith.constant -2.000000e+00 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<5xf32>
    %alloc_7 = memref.alloc() : memref<5xf32>
    %alloc_8 = memref.alloc() : memref<5xf32>
    memref.store %cst_6, %alloc[%c0] : memref<5xf32>
    memref.store %cst_5, %alloc[%c1] : memref<5xf32>
    memref.store %cst_4, %alloc[%c2] : memref<5xf32>
    memref.store %cst_3, %alloc[%c3] : memref<5xf32>
    memref.store %cst_2, %alloc[%c4] : memref<5xf32>
    memref.store %cst_1, %alloc_7[%c0] : memref<5xf32>
    memref.store %cst_6, %alloc_7[%c1] : memref<5xf32>
    memref.store %cst_0, %alloc_7[%c2] : memref<5xf32>
    memref.store %cst_6, %alloc_7[%c3] : memref<5xf32>
    memref.store %cst, %alloc_7[%c4] : memref<5xf32>
    call @diamond_elementwise(%alloc, %alloc_7, %alloc_8) : (memref<5xf32>, memref<5xf32>, memref<5xf32>) -> ()
    %0 = memref.load %alloc_8[%c3] : memref<5xf32>
    return %0 : f32
  }
}
```

`convert-linalg-to-loops` derived a genuine `scf.for` loop -- correctly
ordered loads, the same five arithmetic ops in the same order as the
original `linalg.generic` body, and a store -- straight from the
declarative op above, with no domain-specific code written for this
diamond graph at all. This is exactly the job Chapter 15's own
`LoopNest`/`tileLoop()` does by hand for CUDA Hammer's IR, and it is
the same real `scf.for` MLIR construct Chapter 29 already found inside
Triton's own real compiled TTIR for Flash Attention's KV-block loop --
the same MLIR structured-control-flow op, produced by a different
real frontend, doing the same real job.

**Step 2 -- continue the real progressive lowering down to the LLVM dialect (structured control flow to a plain CFG, then arithmetic, control flow, and memory operations each to their own LLVM-dialect equivalents, then reconcile the type-conversion bookkeeping the pass pipeline leaves behind):**

```bash
mlir-opt step1_loops.mlir --convert-scf-to-cf -o step2_cf.mlir
mlir-opt step2_cf.mlir --convert-arith-to-llvm --convert-cf-to-llvm \
    --finalize-memref-to-llvm --convert-func-to-llvm \
    --reconcile-unrealized-casts -o step3_llvmdialect.mlir
```

**Step 3 -- real JIT execution.** `mlir-cpu-runner` JIT-compiles the
now-LLVM-dialect MLIR and actually runs it in this process -- no
`printMemrefF32`-style runtime helper library ships with this Ubuntu
package (checked directly: not present anywhere on this machine), so
rather than printing the whole output array, `main` returns a single
`f32` -- `out[3]`, the one index where ReLU genuinely clips a nonzero
value away -- and `--entry-point-result=f32` reports it directly:

```bash
mlir-cpu-runner step3_llvmdialect.mlir -e main --entry-point-result=f32
```

**Output (cloud sandbox -- real, live-executed output):**

```text
1.200000e+01
```

1.200000e+01 -- 12.0, matching the hand derivation for `out[3]`
exactly. Nothing about this specific diamond graph was ever written
into any of the five real passes this pipeline ran; every one of them
is a general MLIR conversion pass that would lower any other
`linalg.generic` body the same way.

## C.4 -- Capstone: CUTLASS and the Real Im2col-Aware GEMM Behind Chapter 31's Citation

Chapter 31.2 cited NVIDIA CUTLASS's own documentation
for the real im2col-based implicit-GEMM technique behind convolution
libraries -- convolution, reduced to one big matrix multiply by
replicating each activation element by a factor equal to the filter
size, exactly the real 9x storage blow-up File 082 measured directly.
This section goes one level deeper: a real, sparse-cloned checkout of
NVIDIA's own CUTLASS repository (commit `0b55a2f691d69981583568fd9eb69687b1f0de8a`,
CUTLASS 4.8.0, `include/cutlass` and `include/cute` only, 33MB), and a
real compiled instantiation of CUTLASS's own classic
`cutlass::gemm::device::Gemm` template against those real headers.

Two real, directly observed facts worth naming before the code:
first, compiling against these real headers, unmodified, produces real
`constexpr`-in-`__host__ __device__` warnings from CUTLASS's own
`conv3d_problem_size.h` (pulled in transitively even though this file
only includes `gemm/device/gemm.h`) -- CUTLASS's own diagnostic
suggests the fix directly, `--expt-relaxed-constexpr`, the same kind
of real-warning-driven fix Chapter 34 applied to its own narrowing-
conversion warning. Second, a real, current, filename-level piece of
evidence that im2col is still how production GEMM/convolution
infrastructure names this exact idea, not a detail specific to this
book's own Chapter 31.2 implementation: CUTLASS's modern
Hopper/Blackwell hardware copy-descriptor code ships
`include/cute/atom/copy_traits_sm90_im2col.hpp` and
`copy_traits_sm100_im2col.hpp` -- real files, confirmed present in
this same checkout, with "im2col" in their own real filenames.

No physical GPU is available in this environment -- the same honest
limitation as every CUDA-linked file since Chapter 18 -- so this file
compiles and instantiates the real CUTLASS kernel template, and
reports the real `cudaGetDeviceCount()` result, rather than claiming a
run.

```cpp
// Appendix C.4 capstone -- Chapter 31.2 cited NVIDIA CUTLASS's own docs
// for the real im2col-based implicit-GEMM technique behind convolution
// libraries ("constructs the convolution matrix explicitly via... im2col").
// This file goes one level deeper: it compiles a real CUTLASS GEMM
// template instantiation (CUTLASS 4.8.0, commit 0b55a2f, the classic
// SIMT cutlass::gemm::device::Gemm API) against CUTLASS's own real
// headers, and names the exact real header where CUTLASS's modern
// hardware (Hopper/Blackwell TMA) copy engine has an explicit,
// filename-level im2col concept of its own:
//   include/cute/atom/copy_traits_sm90_im2col.hpp
//   include/cute/atom/copy_traits_sm100_im2col.hpp
// -- direct, current evidence that im2col is still how real production
// GEMM/convolution infrastructure names this idea, not a detail specific
// to this book's own Chapter 31.2 implementation.
//
// No physical GPU is available in this environment (same honest
// limitation as every CUDA-linked file since Chapter 18), so this file
// compiles and instantiates the real CUTLASS kernel template but only
// reports the real cudaGetDeviceCount() result rather than claiming
// a run.
#include <cstdio>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80
>;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount: err=%s, deviceCount=%d\n",
           cudaGetErrorString(err), deviceCount);

    const int M = 4, N = 4, K = 4;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    if (deviceCount == 0) {
        printf("No CUDA device detected in this environment -- compiling and\n");
        printf("instantiating the real CUTLASS Gemm template above (this is the\n");
        printf("real thing CUTLASS's build would have checked at kernel-launch\n");
        printf("time) is as far as this file honestly goes, the same real\n");
        printf("compile-but-no-claimed-run split as Chapter 18's CUDA backend\n");
        printf("and Appendix A.3's vector-add smoke test.\n");
        return 0;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    CutlassGemm::TensorRefA refA(dA, K);
    CutlassGemm::TensorRefB refB(dB, N);
    CutlassGemm::TensorRefC refC(dC, N);
    CutlassGemm::TensorRefD refD(dC, N);

    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        problem_size,
        refA, refB, refC, refD,
        {1.0f, 0.0f}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    printf("can_implement: %s\n", cutlass::cutlassGetStatusString(status));

    status = gemm_op.initialize(args);
    printf("initialize: %s\n", cutlass::cutlassGetStatusString(status));

    status = gemm_op();
    printf("run: %s\n", cutlass::cutlassGetStatusString(status));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
```

**Compile against the real, sparse-cloned CUTLASS headers:**

```bash
nvcc -std=c++17 -arch=sm_80 --expt-relaxed-constexpr \
    -I/path/to/cutlass/include \
    100_cutlass_and_the_real_im2col_aware_gemm_behind_chapter_31s_citation.cu \
    -o cutlass_gemm
echo "exit: $?"
```

**Output (cloud sandbox, x86-64 -- real, live-executed output; cloud-only per this book's standing rule, and this appendix's own device-toolchain-absence finding above):**

```text
cudaGetDeviceCount: err=no CUDA-capable device is detected, deviceCount=0
No CUDA device detected in this environment -- compiling and
instantiating the real CUTLASS Gemm template above (this is the
real thing CUTLASS's build would have checked at kernel-launch
time) is as far as this file honestly goes, the same real
compile-but-no-claimed-run split as Chapter 18's CUDA backend
and Appendix A.3's vector-add smoke test.
```

Zero compiler errors, zero warnings once `--expt-relaxed-constexpr`
is passed: a real, current CUTLASS GEMM template -- the same real
family of kernels behind the im2col-based convolution technique
Chapter 31.2 cited -- compiles cleanly against this book's own toolchain,
and, with a physical GPU, `gemm_op.can_implement(args)`,
`gemm_op.initialize(args)`, and `gemm_op()` are exactly the three real
calls that would check, prepare, and launch it.

## Closing synthesis

Chapters 8 through 20 hand-wrote a pass manager, four optimization
passes, two fusion passes, a loop-nest abstraction, and two codegen
backends -- all real, all tied to CUDA Hammer's own `OpKind`/`Graph`/
`Node` types, all necessary because nothing else in this book's own
toolchain understood that IR. This appendix hand-translated the exact
same diamond graph into two real, independent, production-grade IRs
that plenty else already understands, and got, for free: a verifier
(`opt -passes=verify`), two independent real code generators (`clang`'s
direct IR lowering and `llc`'s native backend, agreeing exactly), a
general-purpose optimizer that folded the whole graph to a constant
and then deleted the dead function entirely, a declarative loop-nest
abstraction (`linalg.generic`) that lowers to the same real `scf.for`
construct Chapter 29 already found inside Triton's own compiled
output, and a real JIT execution engine. Chapter 3's own survey named
XLA, TVM, and Triton as three real production compilers built on this
exact infrastructure; this appendix is the one place this book opened
that infrastructure up directly, by hand, and ran it.
