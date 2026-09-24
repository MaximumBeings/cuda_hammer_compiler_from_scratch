# Appendix D: Polyhedral Compilation -- What a Real Scheduler Finds That CUDA Hammer's Search Never Did

Chapter 33.2's own honest boundary named a real gap directly: CUDA
Hammer's IR has never had a loop or control-flow construct, so its own
finite-difference heat stencil could only be expressed by unrolling a
fixed, compile-time-known step count straight into the host driver --
5 separate, sequentially-dependent graphs, never one real loop. Chapter
21 built a real search space over tile sizes, loop orders, and unroll
factors (128 schedules on one small `LoopNest`), and Chapter 23's real
hybrid autotuner pruned that space with a cost model, then MEASURED the
survivors on real hardware. This appendix asks what happens when the
actual loop form Chapter 33.2 could never write is handed to a real
polyhedral compiler instead: Polly, LLVM's own polyhedral loop
optimizer, part of the exact same real LLVM 18.1.3 toolchain Appendix C
already used (built on the real Integer Set Library, isl 0.26, already
installed alongside `clang`/`opt`/`llc` since before Appendix C began).
Polly analyzes a loop nest's exact iteration space and every array
access as sets of integer points and affine maps, decides automatically
what is safe to reorder or run in parallel, and picks a transformation
by a real heuristic cost function -- with zero real-hardware
measurement, the opposite of Chapter 23's own discipline.

## Appendix D's own shape

```text
+------------------------------------------------------------------+
|  Appendix D's own shape, section by section                       |
|                                                                    |
|  D.1  A Genuine SCoP -- Chapter 33.2's stencil update rule,        |
|       written as an ACTUAL nested loop instead of a host-unrolled  |
|       chain: a real static control part (SCoP), detected and       |
|       dependence-analyzed by Polly directly.                       |
|                                                                    |
|  D.2  What Polly's Own Scheduler Finds, With Zero Measurement --   |
|       a real automatic tiling AND loop-skewing transformation,     |
|       picked by isl's own cost heuristic; real codegen, compiled,  |
|       and run, confirmed to still produce the exact same real      |
|       numbers as the untransformed program.                        |
|                                                                    |
|  D.3  Capstone: What's Actually Safe to Parallelize -- Polly's     |
|       real dependence-based reasoning about two different SCoPs,   |
|       reaching two different, correct real answers, backed by a    |
|       real runtime non-aliasing check where the compiler cannot    |
|       prove safety statically.                                     |
+------------------------------------------------------------------+
```

Toolchain: real LLVM/Polly 18.1.3, the same install Appendix C used --
no new packages needed this time. The device -- confirmed in Appendix C
to have neither a compiler toolchain beyond `gcc`/`g++` nor any
package-manager root access at all -- remains unable to run any of this
appendix's real commands, for the same reason. Every source file is
still sent to the device and md5-verified byte-identical, same as every
earlier chapter.

## D.1 -- A Genuine SCoP

This file writes Chapter 33's own real update rule --
`w[t+1][i] = r*w[t][i-1] + (1-2r)*w[t][i] + r*w[t][i+1]`, `r=0.25` --
as an actual nested `for` loop, with both loop bounds and every array
subscript an affine function of the surrounding indices `t` and `i`:
exactly the shape a polyhedral compiler needs. `stencil()` is kept
separate from `main()` so Polly's own function-scoped analysis below
stays focused on the one function that matters.

```c
/* Appendix D -- the real loop CUDA Hammer's own IR has never been able
 * to express. Chapter 33.2's own honest boundary named this directly:
 * "CUDA Hammer's IR has never had a loop/control-flow construct...
 * unrolls a FIXED, compile-time-known step count directly into the
 * host driver." This file writes that same update rule --
 *
 *   w[t+1][i] = r*w[t][i-1] + (1-2r)*w[t][i] + r*w[t][i+1]
 *
 * -- with r=0.25 (Chapter 33's own real stability condition and
 * formula) -- as an ACTUAL nested for-loop, not a host-unrolled chain.
 * `stencil()` is a real static control part (SCoP): every loop bound
 * and every array subscript is an affine function of the surrounding
 * loop indices, exactly what a polyhedral compiler like Polly (part of
 * this book's own LLVM 18.1.3 toolchain since Appendix C) analyzes and
 * transforms directly.
 */
#include <stdio.h>
#define N 10
#define STEPS 3

void stencil(double w[STEPS + 1][N], double r) {
    int t, i;
    for (t = 0; t < STEPS; t++) {
        for (i = 1; i < N - 1; i++) {
            w[t + 1][i] = r * w[t][i - 1] + (1.0 - 2.0 * r) * w[t][i] + r * w[t][i + 1];
        }
        w[t + 1][0] = w[t][0];
        w[t + 1][N - 1] = w[t][N - 1];
    }
}

int main() {
    double w[STEPS + 1][N];
    int i;
    for (i = 0; i < N; i++) w[0][i] = 0.0;
    w[0][N / 2] = 100.0;

    stencil(w, 0.25);

    for (i = 0; i < N; i++) printf("%.4f ", w[STEPS][i]);
    printf("\n");
    return 0;
}
```

**Compile to canonical LLVM IR (the standard real preparation Polly's own docs describe -- disable the default optimizer so Polly's own canonicalization passes run on clean, unoptimized IR, then run those passes explicitly):**

```bash
clang -O1 -Xclang -disable-llvm-passes -S -emit-llvm 101_the_real_loop_form_of_chapter_33s_stencil_a_genuine_scop.c -o stencil.ll
opt -polly-canonicalize stencil.ll -S -o stencil_canon.ll
```

**Detect SCoPs -- first attempt, Polly's own default settings:**

```bash
opt -polly-only-func=stencil -polly-print-detect -disable-output stencil_canon.ll
```

**Output (cloud sandbox, x86-64 -- real, live-executed output):**

```text
Printing analysis 'Polly - Detect static control parts (SCoPs)' for function 'stencil':

Printing analysis 'Polly - Detect static control parts (SCoPs)' for function 'main':
```

Nothing detected -- not even reported as an invalid candidate. This
loop nest is small enough (3 timesteps, 8 interior points) that Polly's
own real profitability heuristic filters it out before analysis even
begins, the same kind of "not worth the compiler's own overhead"
judgment CUDA Hammer's Chapter 16 makes with `maxChainLength`, just
applied one level earlier. Passing the one real flag that disables that
heuristic (`-polly-process-unprofitable`, intended for exactly this
kind of inspection) finds it:

```bash
opt -polly-process-unprofitable -polly-only-func=stencil -polly-print-detect -disable-output stencil_canon.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
Printing analysis 'Polly - Detect static control parts (SCoPs)' for function 'stencil':
Valid Region for Scop: .preheader => %35

Printing analysis 'Polly - Detect static control parts (SCoPs)' for function 'main':
```

A real, valid SCoP, `.preheader => %35`. With the region confirmed,
Polly's own real dependence analysis computes every genuine
read-after-write (RAW) dependence in the loop nest directly, as exact
sets of integer points (`isl` notation: `Stmt1[i0, i1] -> Stmt1[1 + i0,
o1]` reads as "iteration `(i0,i1)` of statement `Stmt1` must run before
iteration `(1+i0, o1)` of the same statement, for every `o1` in the
stated range") -- not asserted, not estimated, computed:

```bash
opt -polly-process-unprofitable -polly-only-func=stencil -polly-print-dependences -disable-output stencil_canon.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
Printing analysis 'Polly - Calculate dependences' for region: '.preheader => %35' in function 'stencil':
	RAW dependences:
		{ Stmt2[i0] -> Stmt1[1 + i0, 0] : 0 <= i0 <= 1; Stmt2[i0] -> Stmt2[1 + i0] : 0 <= i0 <= 1; Stmt2b[i0] -> Stmt2b[1 + i0] : 0 <= i0 <= 1; Stmt2b[i0] -> Stmt1[1 + i0, 7] : 0 <= i0 <= 1; Stmt1[i0, i1] -> Stmt1[1 + i0, o1] : 0 <= i0 <= 1 and 0 <= i1 <= 7 and o1 >= -1 + i1 and 0 <= o1 <= 7 and o1 <= 1 + i1 }
	WAR dependences:
		{  }
	WAW dependences:
		{  }
	Reduction dependences:
		{  }
	Transitive closure of reduction dependences:
		{  }
```

Every one of these five real dependence relations traces back to
`t+1` appearing on the write side and `t` on the read side of File
101's own update rule -- Polly derived the exact real dependence
structure of a stencil from the loop's own array-subscript arithmetic,
with no stencil-specific code anywhere in Polly itself.

## D.2 -- What Polly's Own Scheduler Finds, With Zero Measurement

Chapter 21 enumerated 128 real schedules over one
small `LoopNest` by brute force; Chapter 23's hybrid autotuner pruned
with a cost model and then MEASURED the survivors on real hardware.
Polly's own real scheduler does neither: it calls into isl's own
scheduling algorithm (Feautrier/Pluto-style, a real published
polyhedral scheduling technique), which picks a transformation purely
from the dependence structure D.1 just computed and a built-in cost
heuristic -- no candidate schedule is ever run, timed, or cached.
File 101's own `stencil()`, unchanged, is the input:

```bash
opt -polly-process-unprofitable -polly-only-func=stencil -polly-opt-isl -polly-print-opt-isl -disable-output stencil_canon.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
Printing analysis 'Polly - Optimize schedule of SCoP' for region: '.preheader => %35' in function 'stencil':
Calculated schedule:
domain: "{ Stmt2b[i0] : 0 <= i0 <= 2; Stmt2[i0] : 0 <= i0 <= 2; Stmt1[i0, i1] : 0 <= i0 <= 2 and 0 <= i1 <= 7 }"
child:
  sequence:
  - filter: "{ Stmt2[i0] }"
    child:
      schedule: "[{ Stmt2[i0] -> [(i0)] }]"
      permutable: 1
  - filter: "{ Stmt2b[i0] }"
    child:
      schedule: "[{ Stmt2b[i0] -> [(i0)] }]"
      permutable: 1
  - filter: "{ Stmt1[i0, i1] }"
    child:
      mark: "1st level tiling - Tiles"
      child:
        schedule: "[{ Stmt1[i0, i1] -> [(floor((i0)/32))] }, { Stmt1[i0, i1] -> [(floor((i0 + i1)/32))] }]"
        permutable: 1
        child:
          mark: "1st level tiling - Points"
          child:
            schedule: "[{ Stmt1[i0, i1] -> [((i0) mod 32)] }, { Stmt1[i0, i1] -> [((i0 + i1) mod 32)] }]"
            permutable: 1
```

Two real, substantive decisions are visible directly in this real isl
schedule tree. First, `Stmt1` (the interior-point update) is tiled --
`"1st level tiling - Tiles"` / `"1st level tiling - Points"`, the exact
same idea as Chapter 15's own hand-written `tileLoop()`, chosen
automatically here rather than requested by a human. Second, and more
interesting: the tile's own schedule functions are `floor((i0)/32)` and
`floor((i0 + i1)/32)` -- the SECOND dimension is scheduled on `i0 + i1`,
not `i1` alone. That `+i0` term is a real loop SKEW: isl's scheduler
found that iterating diagonally across the `(t, i)` iteration space,
not row-by-row, is what its own cost heuristic prefers for this
dependence pattern -- the classic real wavefront transformation
polyhedral-compilation literature uses for stencils, discovered here
automatically rather than looked up. The generated AST shows exactly
this, plainly, over the real numbers this book's own stencil actually
uses (3 timesteps, 8 interior points):

```bash
opt -polly-process-unprofitable -polly-only-func=stencil -polly-opt-isl -polly-ast -polly-print-ast -disable-output stencil_canon.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '.preheader => %35' in function 'stencil':
Printing analysis 'Polly - Generate an AST of the SCoP (isl)'.preheader => %35' in function 'stencil':
:: isl ast :: stencil :: %.preheader---%35

if (1)

    {
      for (int c0 = 0; c0 <= 2; c0 += 1)
        Stmt2(c0);
      for (int c0 = 0; c0 <= 2; c0 += 1)
        Stmt2b(c0);
      // 1st level tiling - Tiles
      // 1st level tiling - Points
      for (int c2 = 0; c2 <= 2; c2 += 1)
        for (int c3 = c2; c3 <= c2 + 7; c3 += 1)
          Stmt1(c2, -c2 + c3);
    }

else
    {  /* original code */ }
```

`Stmt1(c2, -c2 + c3)` is the real skewed access -- the loop now walks
`c3` from `c2` to `c2+7`, and `-c2+c3` recovers the original `i` index
inside the transformed iteration space, exactly the classic loop-skew
identity. A transformation is only worth trusting if it still computes
the same real answer, so the whole pipeline is compiled and run two
real ways: once with Polly entirely disabled (an ordinary `-O2` build),
and once with Polly enabled and explicitly told to tile:

```bash
clang -O2 101_the_real_loop_form_of_chapter_33s_stencil_a_genuine_scop.c -o baseline_exe
./baseline_exe
```

**Output (cloud sandbox -- real, live-executed output):**

```text
0.0000 0.0000 1.5625 9.3750 23.4375 31.2500 23.4375 9.3750 1.5625 0.0000 
```

```bash
clang -O2 -mllvm -polly -mllvm -polly-process-unprofitable \
    -mllvm -polly-only-func=stencil -mllvm -polly-tiling \
    101_the_real_loop_form_of_chapter_33s_stencil_a_genuine_scop.c -o polly_exe
./polly_exe
```

**Output (cloud sandbox -- real, live-executed output):**

```text
0.0000 0.0000 1.5625 9.3750 23.4375 31.2500 23.4375 9.3750 1.5625 0.0000 
```

Byte-identical, and independently cross-checked against a direct
Python re-implementation of the same update rule this session. Polly's
own real skewed, tiled schedule is a genuine reordering of when each
`(t,i)` update runs -- and the real isl dependence analysis from D.1 is
exactly what guarantees that reordering cannot change the answer:
every dependence it found is still satisfied by the new schedule, or
isl would never have proposed it.

## D.3 -- Capstone: What's Actually Safe to Parallelize

A second, independent real SCoP, chosen to contrast
directly with File 101's stencil: every iteration of this loop touches
completely independent array elements, with no loop-carried dependence
at all.

```c
/* Appendix D.3 -- a second, independent real SCoP, chosen to contrast
 * directly with File 101's stencil: every iteration of this loop reads
 * and writes completely independent elements, with no loop-carried
 * dependence at all. Used to show what Polly's real dependence analysis
 * concludes is SAFE to parallelize, laid next to File 101's stencil
 * (reused unchanged), where the same real analysis correctly finds the
 * opposite answer for the outer, time-carried loop.
 */
#define N 100

void axpy(double a[N], double b[N], double s) {
    int i;
    for (i = 0; i < N; i++) {
        a[i] = a[i] + s * b[i];
    }
}
```

**Ask Polly's real dependence analysis what is safe to run in parallel (`-polly-parallel`, printing the resulting AST with real parallel markers):**

```bash
opt -polly-process-unprofitable -polly-parallel -polly-ast -polly-ast-print-accesses -polly-print-ast -disable-output axpy_canon.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '%4 => %13' in function 'axpy':
Printing analysis 'Polly - Generate an AST of the SCoP (isl)'%4 => %13' in function 'axpy':
:: isl ast :: axpy :: %4---%13

if (1 && (&MemRef1[100] <= &MemRef0[0] || &MemRef0[100] <= &MemRef1[0]))

    // Loop with Metadata
    #pragma simd
    #pragma known-parallel
    for (int c0 = 0; c0 <= 99; c0 += 1)
      Stmt0(
        /* read  */ &MemRef0[c0]
        /* read  */ &MemRef1[c0]
        /* read  */ &MemRef2
        /* write */  MemRef0[c0]
      );

else
    {  /* original code */ }
```

Two real things worth naming. First, the whole loop is marked real
`#pragma known-parallel` -- Polly's own dependence analysis correctly
found zero loop-carried dependences here, the same conclusion File 101's
own analysis reached for its INNER spatial loop only. Second, and easy
to miss: the generated code is guarded by a real RUNTIME check,
`&MemRef1[100] <= &MemRef0[0] || &MemRef0[100] <= &MemRef1[0]` -- "array
`b` ends before array `a` starts, or array `a` ends before array `b`
starts." `a` and `b` are two separate `double[N]` PARAMETERS; nothing in
this function's own C source rules out a caller passing overlapping
memory for both, so Polly cannot prove non-aliasing FROM THE SOURCE
ALONE -- it inserts a genuine runtime check instead, falling back to
`/* original code */` if the check fails. This is a real, honest
compiler-correctness technique this book has not needed before: a
static analysis that cannot prove a fact at compile time can still
enforce it at runtime, rather than either assuming it or giving up
the optimization entirely.

Now File 101's own `stencil()`, reused unchanged, gets the same real
question asked of it:

```bash
opt -polly-process-unprofitable -polly-only-func=stencil -polly-parallel -polly-ast -polly-print-ast -disable-output stencil_canon.ll
```

**Output (cloud sandbox -- real, live-executed output):**

```text
Printing analysis 'Polly - Generate an AST from the SCoP (isl)' for region: '.preheader => %35' in function 'stencil':
Printing analysis 'Polly - Generate an AST of the SCoP (isl)'.preheader => %35' in function 'stencil':
:: isl ast :: stencil :: %.preheader---%35

if (1)

    // Loop with Metadata
    #pragma minimal dependence distance: 1
    for (int c0 = 0; c0 <= 2; c0 += 1) {
      // Loop with Metadata
      #pragma simd
      #pragma known-parallel
      for (int c1 = 0; c1 <= 7; c1 += 1)
        Stmt1(c0, c1);
      Stmt2(c0);
      Stmt2b(c0);
    }

else
    {  /* original code */ }
```

Two different real, correct answers from the same real analysis
technique, in the same real AST: the OUTER, time-carried loop is
labeled `#pragma minimal dependence distance: 1` -- Polly's own
diagnostic naming the exact real dependence D.1 already computed
(iteration `t` genuinely depends on iteration `t-1`) and correctly
refusing to mark it parallel; the INNER, spatial loop -- every point in
one timestep depends only on already-computed values from the PREVIOUS
timestep, never on another point in the SAME timestep -- is marked real
`#pragma known-parallel`, with no runtime check needed at all, since
every access here is provably non-overlapping from the loop's own
affine index arithmetic. Chapter 33.2 could only ever unroll this
computation by hand, one host-orchestrated step at a time, with no way
to ask which parts were safe to reorder or run concurrently. Polly's
real dependence analysis answers that question directly, correctly,
and for free, the moment the computation is written as the loop it
always structurally was.

## Closing synthesis

Chapter 21 built a real, general search space over schedules and
enumerated all of it; Chapter 23 pruned that space with a cost model
and then spent real wall-clock time measuring the survivors on real
hardware, twice finding that the true best schedule was not even in
the cost model's own top-8. Polly took a different real path entirely:
no candidate is ever run, no hardware is ever touched, and the whole
transformation -- tiling, skewing, and a provably-safe parallelization
decision -- comes from one static analysis of the loop's own dependence
structure, computed once. Neither approach is simply better: Chapter 23's
own real finding (the cost model's own top-8 missing the true best
schedule, twice, on two different real machines) is exactly the kind of
gap a purely static, measurement-free scheduler like Polly's can never
close by construction, while Polly's own real skewed, tiled, correctness-
preserving schedule for a whole stencil is exactly the kind of loop-level
transformation Chapter 21's schedule search was never structured to
consider at all -- CUDA Hammer's own `Schedule` never had a skew
parameter. Chapter 33.2's own honest boundary is the reason this appendix
exists: the moment that stencil is written as the real loop it always
was, structurally, a real, independent, off-the-shelf polyhedral
compiler can reason about it in ways CUDA Hammer's own from-scratch
autotuner, built for a completely different IR shape, never could.
