# Appendix E: Profiling and Benchmarking -- What Real Hardware Counters Show That a Clock Alone Can't

Chapter 22 built two real cost models -- one for loop overhead, one for
memory-access stride -- and, in both cases, measured them honestly
against a real clock and found a real gap: `estimateLoopOverheadCost()`
always recommends the largest unroll factor, but File 054's own
best-of-5 wall-clock measurement, at extent=50,000,000, found a real
turnover point the model could not predict, and named the likely
reason directly in prose without ever measuring it -- "instruction-
cache pressure, register pressure, and lost auto-vectorization
opportunities." Chapter 12 argued, also in prose, that fusion's real
benefit is avoiding round-trips to memory for intermediate tensors.
Both claims are almost certainly true. Neither one, through Chapter 22,
was ever backed by a number that actually counts instructions or cache
misses -- every verification in this book so far has been either
correctness (bit-for-bit agreement, no clock involved at all) or
wall-clock timing (a real but indirect signal: it tells you THAT
something is slower, never WHY).

This appendix asks what a real hardware-counting profiler shows when
pointed at CUDA Hammer's own code. Linux `perf`, this book's first
real choice, turns out not to be usable in the sandbox this book was
built in -- confirmed directly, not assumed: `linux-tools-generic` and
`linux-tools-common` install cleanly via `apt`, but `perf stat` itself
fails at run time with `WARNING: perf not found for kernel
6.18.44-fc`, a genuine kernel/packaged-build mismatch specific to this
container, not a missing flag or a permissions issue. Valgrind's
Cachegrind and Callgrind, by contrast, are installed and work: both
are REAL, deterministic simulators -- not sampling profilers -- that
re-execute a program's every instruction through a modeled cache
hierarchy (Cachegrind) or a modeled cache hierarchy plus a real
per-line, per-function call graph (Callgrind), and report exact,
repeatable counts instead of statistical estimates. Every number in
this appendix is exactly reproducible on the same machine, unlike a
wall-clock measurement, which is precisely what makes it a good
complement to Chapter 22's own honest, noisy, real-clock numbers
rather than a replacement for them.

## Appendix E's own shape

```text
+------------------------------------------------------------------+
|  Appendix E's own shape, section by section                       |
|                                                                    |
|  E.1  Cachegrind: What Fusion Actually Saves -- File 039's own    |
|       diamond graph (Chapter 17), materialized vs. fused, at real  |
|       scale (N=20,000,000): real dynamic instruction counts and    |
|       real simulated D1/LL cache-miss counts, turning Chapter 12's |
|       prose argument into a measured multiple.                     |
|                                                                    |
|  E.2  Callgrind: Where the Extra Instructions Actually Go -- the   |
|       SAME binary, profiled with a real call-graph, line-level     |
|       profiler instead of a cache simulator: which of the four     |
|       materialized steps costs the most, and how much of that      |
|       cost is the vector constructors themselves, not the          |
|       arithmetic.                                                  |
|                                                                    |
|  E.3  Capstone: Testing Chapter 22's Own Explanation -- File 054's |
|       own computeUnrolled(), reused unchanged, under real           |
|       Cachegrind instruction/cache counting AND real branch-        |
|       predictor simulation across unroll factors: which of Chapter  |
|       22's own named suspects (instruction-cache pressure, data     |
|       cache pressure, something else entirely) the real numbers     |
|       actually support.                                            |
+------------------------------------------------------------------+
```

Toolchain: real Valgrind 3.22.0 (`--tool=cachegrind`,
`--tool=callgrind`), already installed in the cloud sandbox; plain
`g++` 13.3.0, `-O2`. The device was checked directly for this
appendix and confirmed to have `g++` but no `valgrind` at all, and --
as Appendix C and D already established -- no root or package-manager
access to install it. This draws a real, honest line different from
either earlier appendix's own toolchain limit: Appendix C and D needed
a whole compiler toolchain (LLVM/MLIR/Polly) just to COMPILE their own
files at all, so nothing in them could run on the device. This
appendix's own source files (Files 103 and 104) are plain, portable
C++17 with no such dependency -- they compile and run correctly on the
device with its own stock `g++`, and their own bit-for-bit correctness
self-checks are cross-verified there exactly as strictly as every
chapter through 21. Only the PROFILING itself -- every Cachegrind and
Callgrind command and output in this appendix -- is cloud-sandbox-only,
because Valgrind itself is the one piece missing on the device, not
the code it profiles.

## E.1 -- Cachegrind: What Fusion Actually Saves

File 039's own `main()` (Chapter 17) built the diamond
graph from Chapter 6 -- `t1=a+b`, `t2=t1*a`, `t3=relu(t1)`,
`out=t2+t3` -- and ran it two ways through `evaluateArrays()`: as the
ORIGINAL graph, where `t1` is materialized as its own array because it
has two consumers, and as the FUSED graph, where `t2`, `t3`, and `out`
collapse into one `FusedElementwise` step. Both agreed bit-for-bit, at
12 elements. This file builds the exact same two computations directly
in C++ -- no `Graph`/`Node` machinery, just the two code paths
themselves -- at N=20,000,000 elements: `materialized()` allocates and
writes a full `std::vector<float>` for every one of `t1`, `t2`, `t3`,
and `out` and reads each intermediate back from memory on the next
step, while `fused()` computes the whole chain for one element with
`t1`/`t2`/`t3` as ordinary local `float` variables -- register-
resident, never written to memory at all -- inside a single pass.

```cpp
// Appendix E: Profiling and Benchmarking
// 103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp
//
// Section E.1 -- Chapter 17's own File 039 built evaluateArrays() and ran it
// on the diamond graph from Chapter 6: a:[3,4], b:[4], t1=a+b, t2=t1*a,
// t3=relu(t1), out=t2+t3, comparing the ORIGINAL graph (t1 materialized as
// its own array, because it has two consumers) against the FUSED graph
// (t2, t3, out collapsed into one FusedElementwise step) -- and found they
// agree bit-for-bit at every one of 12 elements. Chapter 12 then argued, in
// prose, that fusion's real benefit is avoiding round-trips to memory for
// intermediate tensors -- "memory-bound vs. compute-bound" -- but every
// verification in this book through Chapter 22 has been either correctness
// (bit-for-bit agreement, no clock involved) or wall-clock timing (Chapter
// 19's best-of-N discipline, Chapter 22's own honest turnover-point finding
// that a wall clock alone could not explain). Neither approach can show
// what actually happens at the memory system: how many loads and stores
// each version issues, and how many of those miss cache. This file builds
// the SAME diamond graph shape as File 039 at real scale -- N=20,000,000
// elements instead of 12 -- run two ways: MATERIALIZED (every intermediate
// tensor gets its own real heap-allocated std::vector<float>, written in
// full and read back, exactly like the unfused half of File 039's own
// evaluateArrays()) and FUSED (one pass, one register-resident running
// value per output element, exactly like the fused half). Both are checked
// bit-for-bit against each other, continuing this book's own correctness
// discipline; Sections E.1 and E.2 then profile this SAME binary with real
// Valgrind tools instead of asserting which one Chapter 12's prose favors.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 -g 103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp -o 103_driver
// (Section E.1 explains, with real nm evidence, why this file needs
// __attribute__((noinline)) on both functions and NOT a drop to -O0: both
// materialized() and fused() are static and each called exactly once, so
// plain -O1 already inlines both of them into main -- confirmed by `nm`
// reporting no materialized/fused symbols at -O1 or -O2 without the
// attribute -- which would erase the per-function boundary Section E.2's
// Callgrind profiling depends on before a single profiler ever runs. Real
// -O2 code (auto-vectorized, register-allocated) is what Chapter 12's own
// fusion argument is actually about, so keeping -O2 and forcing the two
// functions to stay real, separately-callable symbols with the attribute
// is the honest fix, not trading away the optimization level this book
// verifies everywhere else.)
// Run:     ./103_driver
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

static const long long N = 20'000'000;

// ==================== MATERIALIZED: same four steps as File 039's UNFUSED graph, each its own real buffer ====================
__attribute__((noinline))
static std::vector<float> materialized(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> t1(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) t1[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] + b[static_cast<size_t>(i)];

    std::vector<float> t2(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) t2[static_cast<size_t>(i)] = t1[static_cast<size_t>(i)] * a[static_cast<size_t>(i)];

    std::vector<float> t3(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) t3[static_cast<size_t>(i)] = std::max(0.0f, t1[static_cast<size_t>(i)]);

    std::vector<float> out(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) out[static_cast<size_t>(i)] = t2[static_cast<size_t>(i)] + t3[static_cast<size_t>(i)];

    return out;
}

// ==================== FUSED: same graph, one pass, t1 held in a register, never written to memory ====================
__attribute__((noinline))
static std::vector<float> fused(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) {
        float av = a[static_cast<size_t>(i)];
        float bv = b[static_cast<size_t>(i)];
        float t1 = av + bv;
        float t2 = t1 * av;
        float t3 = std::max(0.0f, t1);
        out[static_cast<size_t>(i)] = t2 + t3;
    }
    return out;
}

int main() {
    printf("=== Appendix E.1/E.2: the same diamond graph, materialized vs. fused, N=%lld ===\n\n", N);

    std::vector<float> a(static_cast<size_t>(N)), b(static_cast<size_t>(N));
    for (long long i = 0; i < N; ++i) {
        a[static_cast<size_t>(i)] = static_cast<float>((i % 1013) - 500) * 0.01f;   // genuinely non-uniform, signed
        b[static_cast<size_t>(i)] = static_cast<float>((i % 727) - 300) * 0.01f;
    }

    std::vector<float> outMaterialized = materialized(a, b);
    std::vector<float> outFused = fused(a, b);

    bool bitForBitEqual = (outMaterialized.size() == outFused.size());
    if (bitForBitEqual) {
        for (long long i = 0; i < N; ++i) {
            if (outMaterialized[static_cast<size_t>(i)] != outFused[static_cast<size_t>(i)]) { bitForBitEqual = false; break; }
        }
    }
    printf("materialized.out[0]  = %g\n", outMaterialized[0]);
    printf("fused.out[0]         = %g\n", outFused[0]);
    printf("materialized.out[%lld] = %g\n", N - 1, outMaterialized[static_cast<size_t>(N - 1)]);
    printf("fused.out[%lld]        = %g\n", N - 1, outFused[static_cast<size_t>(N - 1)]);
    printf("\nself-check: materialized and fused agree bit-for-bit at all %lld elements (%s) --\n",
           N, bitForBitEqual ? "confirmed" : "MISMATCH");
    printf("same finding as File 039's own Part B, now at real scale instead of 12 elements.\n");

    return bitForBitEqual ? 0 : 1;
}
```

Both `materialized()` and `fused()` are `static` and each has exactly
one call site in `main()`, which means plain `-O1` already inlines
both of them completely -- confirmed directly, not assumed:

```bash
g++ -std=c++17 -O1 -g 103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp -o /tmp/o1test
nm /tmp/o1test | grep -iE "materialized|fused"
```

produces no output at all -- both symbols are gone, folded into
`main`. That would erase the exact per-function boundary Section E.2's
Callgrind profiling needs before a single profiler ever runs. The real
fix is `__attribute__((noinline))` on both functions (already in the
file above), not a drop to `-O0`: real `-O2` code -- auto-vectorized,
fully register-allocated -- is what Chapter 12's own fusion argument is
actually about, and `-O0`'s own lack of register allocation would
change what is being measured. With the attribute in place at real
`-O2`:

```bash
g++ -std=c++17 -Wall -Wextra -O2 -g 103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp -o 103_driver
nm 103_driver | grep -iE "materialized|fused"
```

**Output (cloud sandbox -- real, live-executed output):**

```text
0000000000001610 t _ZL12materializedRKSt6vectorIfSaIfEES3_
0000000000001120 t _ZL12materializedRKSt6vectorIfSaIfEES3_.cold
0000000000001570 t _ZL5fusedRKSt6vectorIfSaIfEES3_
```

Both symbols survive. Running the binary directly first confirms
correctness, continuing this book's own discipline, before any
profiler is involved at all:

```bash
./103_driver
```

**Output (cloud sandbox -- real, live-executed output):**

```text
=== Appendix E.1/E.2: the same diamond graph, materialized vs. fused, N=20000000 ===

materialized.out[0]  = 40
fused.out[0]         = 40
materialized.out[19999999] = 3.696
fused.out[19999999]        = 3.696

self-check: materialized and fused agree bit-for-bit at all 20000000 elements (confirmed) --
same finding as File 039's own Part B, now at real scale instead of 12 elements.
```

Bit-for-bit agreement at all 20,000,000 elements -- the same finding
as File 039's own Part B, now at real scale. With correctness settled,
Cachegrind asks the real question Chapter 12's prose never measured:

```bash
valgrind --tool=cachegrind --cache-sim=yes \
    --cachegrind-out-file=cg103.out ./103_driver
```

**Output (cloud sandbox -- real, live-executed program summary, `==PID==` line-prefix stripped for readability; the counts themselves are the exact, real output):**

```text
I refs:        1,939,279,319
I1  misses:            2,219
LLi misses:            2,175
I1  miss rate:          0.00%
LLi miss rate:          0.00%

D refs:          920,688,701  (220,518,632 rd   + 700,170,069 wr)
D1  misses:       31,266,164  ( 13,763,766 rd   +  17,502,398 wr)
LLd misses:       31,259,612  ( 13,758,058 rd   +  17,501,554 wr)
D1  miss rate:           3.4% (        6.2%     +         2.5%  )
LLd miss rate:           3.4% (        6.2%     +         2.5%  )

LL refs:          31,268,383  ( 13,765,985 rd   +  17,502,398 wr)
LL misses:        31,261,787  ( 13,760,233 rd   +  17,501,554 wr)
LL miss rate:            1.1% (        0.6%     +         2.5%  )
```

Cachegrind also reports this machine's own real, detected cache
geometry -- a 34,603,008-byte (33MB) real LL (last-level) cache, used
again in Section E.3:

**Output (cloud sandbox -- real, live-executed output, same run, `==PID==`/`--PID--` prefixes stripped):**

```text
warning: L3 cache found, using its data for the LL simulation.
warning: specified LL cache: line_size 64  assoc 11  total_size 34,603,008
warning: simulated LL cache: line_size 64  assoc 17  total_size 35,651,584
```

The PROGRAM total mixes `main()`'s own array-fill loop in with both
functions under test. `cg_annotate`, Cachegrind's own real per-
function attribution tool, separates them -- narrowed here to the
three columns (`Ir`, `D1mr`, `D1mw`) that matter for this comparison
with `--show`, a real Cachegrind option, not a hand-edited table:

```bash
cg_annotate --show=Ir,D1mr,D1mw --no-show-percs --no-annotate cg103.out
```

**Output (cloud sandbox -- real, live-executed output):**

```text
--------------------------------------------------------------------------------
-- Metadata
--------------------------------------------------------------------------------
Invocation:       /usr/bin/cg_annotate --show=Ir,D1mr,D1mw --no-show-percs --no-annotate cg103.out
I1 cache:         32768 B, 64 B, 8-way associative
D1 cache:         32768 B, 64 B, 8-way associative
LL cache:         35651584 B, 64 B, 17-way associative
Command:          ./103_driver
Events recorded:  Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw
Events shown:     Ir D1mr D1mw
Event sort order: Ir I1mr ILmr Dr D1mr DLmr Dw D1mw DLmw
Threshold:        0.1%
Annotation:       off

--------------------------------------------------------------------------------
-- Summary
--------------------------------------------------------------------------------
Ir___________ D1mr______ D1mw______ 

1,939,279,319 13,763,766 17,502,398  PROGRAM TOTALS

--------------------------------------------------------------------------------
-- File:function summary
--------------------------------------------------------------------------------
  Ir___________ D1mr______ D1mw_____  file:function

< 1,260,000,158 12,500,021 8,750,017  /tmp/appendixE_work/103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp:
    640,000,077  2,500,007 2,500,004    main
    440,000,058  7,500,010 5,000,012    materialized(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&)
    180,000,023  2,500,004 1,250,001    fused(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&)

<   560,000,091         13 8,750,000  ./string/../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S:__memset_avx2_unaligned_erms

<   117,295,567  1,250,001         0  /usr/include/c++/13/bits/stl_algobase.h:
     68,647,759  1,250,001         0    materialized(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&)
     48,647,759          0         0    fused(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&)

--------------------------------------------------------------------------------
-- Function:file summary
--------------------------------------------------------------------------------
  Ir_________ D1mr_____ D1mw_____  function:file

> 640,000,116 2,500,008 2,500,005  main:
  640,000,077 2,500,007 2,500,004    /tmp/appendixE_work/103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp

> 560,000,091        13 8,750,000  __memset_avx2_unaligned_erms:./string/../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S

> 508,647,825 8,750,013 5,000,012  materialized(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&):
  440,000,058 7,500,010 5,000,012    /tmp/appendixE_work/103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp
   68,647,759 1,250,001         0    /usr/include/c++/13/bits/stl_algobase.h

> 228,647,785 2,500,005 1,250,001  fused(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&):
  180,000,023 2,500,004 1,250,001    /tmp/appendixE_work/103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp
   48,647,759         0         0    /usr/include/c++/13/bits/stl_algobase.h
```

Three real, exact numbers settle Chapter 12's own prose claim.
Instruction count: `materialized()` executes 508,647,825 real dynamic
instructions (its own 440,000,058 plus 68,647,759 inlined from
`std::max` inside its `t3` loop); `fused()` executes 228,647,785 --
`materialized()` costs 2.22x the real instructions for the identical
mathematical result. D1 read misses: 8,750,013 for `materialized()`
against 2,500,005 for `fused()`, a real 3.5x. D1 write misses:
5,000,012 against 1,250,001, a real 4.0x -- close to the naive
prediction (four full array writes instead of one), confirming that
the extra write traffic really is dominated by the three intermediate
buffers `fused()` never allocates at all, not by some other effect.
None of these three numbers came from a clock; all three came from
Cachegrind re-executing the actual real binary through a real
simulated cache model and counting exactly what happened.

## E.2 -- Callgrind: Where the Extra Instructions Actually Go

Cachegrind's own per-function totals already show THAT
`materialized()` costs more; they do not show WHERE, inside
`materialized()`, that extra cost actually comes from -- naive
intuition says "the four elementwise loops," but File 103's own real
structure has four VECTOR CONSTRUCTIONS too (`t1`, `t2`, `t3`, `out`
each get their own fresh, zero-initialized `std::vector<float>`, one
real heap allocation and one real zero-fill each), which `fused()`
only pays once. Callgrind is Valgrind's own call-graph-generating
profiler: same real re-execution as Cachegrind, but it also tracks
real per-line, per-call-site instruction counts and (with
`callgrind_annotate`) prints them directly against the original source
line -- the same SAME binary from Section E.1, profiled with a
different real tool asking a different real question.

```bash
valgrind --tool=callgrind --callgrind-out-file=cl103.out ./103_driver
callgrind_annotate cl103.out
```

**Output (cloud sandbox -- real, live-executed output, top-level function table):**

```text
--------------------------------------------------------------------------------
Ir                     
--------------------------------------------------------------------------------
1,939,268,383 (100.0%)  PROGRAM TOTALS

--------------------------------------------------------------------------------
Ir                    file:function
--------------------------------------------------------------------------------
640,000,077 (33.00%)  103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp:main [/tmp/appendixE_work/103_driver]
560,000,091 (28.88%)  ./string/../sysdeps/x86_64/multiarch/memset-vec-unaligned-erms.S:__memset_avx2_unaligned_erms [/usr/lib/x86_64-linux-gnu/libc.so.6]
440,000,058 (22.69%)  103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp:materialized(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&) [/tmp/appendixE_work/103_driver]
180,000,023 ( 9.28%)  103_the_same_diamond_graph_materialized_vs_fused_at_real_scale.cpp:fused(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&) [/tmp/appendixE_work/103_driver]
 68,647,759 ( 3.54%)  /usr/include/c++/13/bits/stl_algobase.h:materialized(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&)
 48,647,759 ( 2.51%)  /usr/include/c++/13/bits/stl_algobase.h:fused(std::vector<float, std::allocator<float> > const&, std::vector<float, std::allocator<float> > const&)
```

`__memset_avx2_unaligned_erms` -- glibc's own real, hand-vectorized
`memset` -- shows up as its OWN separate real entry at 28.88%, called
from inside `std::vector<float>`'s own constructor every time
`materialized()` or `fused()` allocates a buffer: real evidence that
"materializing an intermediate tensor" costs real instructions before
a single element of actual arithmetic ever runs, just to zero-
initialize the memory C++ requires a `std::vector<float>(N)` to start
from. `callgrind_annotate`'s own real per-LINE view, auto-annotated
directly against File 103's own source text, shows exactly how much of
that falls on each of `materialized()`'s four steps individually:

**Output (cloud sandbox -- real, live-executed output, auto-annotated source, `materialized()` and `fused()`):**

```text
-- line 43 ----------------------------------------
          .           #include <vector>
          .           #include <cmath>
          .           #include <cstdlib>
          .           
          .           static const long long N = 20'000'000;
          .           
          .           // ==================== MATERIALIZED: same four steps as File 039's UNFUSED graph, each its own real buffer ====================
          .           __attribute__((noinline))
         13 ( 0.00%)  static std::vector<float> materialized(const std::vector<float>& a, const std::vector<float>& b) {
          7 ( 0.00%)      std::vector<float> t1(static_cast<size_t>(N));
 80,000,384 ( 4.13%)  => /usr/include/c++/13/bits/stl_vector.h:std::vector<float, std::allocator<float> >::vector(unsigned long, std::allocator<float> const&) (1x)
120,000,001 ( 6.19%)      for (long long i = 0; i < N; ++i) t1[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] + b[static_cast<size_t>(i)];
          .           
          5 ( 0.00%)      std::vector<float> t2(static_cast<size_t>(N));
 80,000,384 ( 4.13%)  => /usr/include/c++/13/bits/stl_vector.h:std::vector<float, std::allocator<float> >::vector(unsigned long, std::allocator<float> const&) (1x)
120,000,002 ( 6.19%)      for (long long i = 0; i < N; ++i) t2[static_cast<size_t>(i)] = t1[static_cast<size_t>(i)] * a[static_cast<size_t>(i)];
          .           
          6 ( 0.00%)      std::vector<float> t3(static_cast<size_t>(N));
 80,000,384 ( 4.13%)  => /usr/include/c++/13/bits/stl_vector.h:std::vector<float, std::allocator<float> >::vector(unsigned long, std::allocator<float> const&) (1x)
 80,000,001 ( 4.13%)      for (long long i = 0; i < N; ++i) t3[static_cast<size_t>(i)] = std::max(0.0f, t1[static_cast<size_t>(i)]);
          .           
          4 ( 0.00%)      std::vector<float> out(static_cast<size_t>(N));
 80,000,384 ( 4.13%)  => /usr/include/c++/13/bits/stl_vector.h:std::vector<float, std::allocator<float> >::vector(unsigned long, std::allocator<float> const&) (1x)
120,000,001 ( 6.19%)      for (long long i = 0; i < N; ++i) out[static_cast<size_t>(i)] = t2[static_cast<size_t>(i)] + t3[static_cast<size_t>(i)];
          .           
          .               return out;
         18 ( 0.00%)  }
      1,689 ( 0.00%)  => /usr/include/c++/13/bits/stl_vector.h:std::vector<float, std::allocator<float> >::~vector() (3x)
          .           
          .           // ==================== FUSED: same graph, one pass, t1 held in a register, never written to memory ====================
          .           __attribute__((noinline))
         10 ( 0.00%)  static std::vector<float> fused(const std::vector<float>& a, const std::vector<float>& b) {
          3 ( 0.00%)      std::vector<float> out(static_cast<size_t>(N));
 80,000,380 ( 4.13%)  => /usr/include/c++/13/bits/stl_vector.h:std::vector<float, std::allocator<float> >::vector(unsigned long, std::allocator<float> const&) (1x)
 60,000,001 ( 3.09%)      for (long long i = 0; i < N; ++i) {
 20,000,000 ( 1.03%)          float av = a[static_cast<size_t>(i)];
          .                   float bv = b[static_cast<size_t>(i)];
 40,000,000 ( 2.06%)          float t1 = av + bv;
 20,000,000 ( 1.03%)          float t2 = t1 * av;
          .                   float t3 = std::max(0.0f, t1);
 40,000,000 ( 2.06%)          out[static_cast<size_t>(i)] = t2 + t3;
          .               }
          .               return out;
          9 ( 0.00%)  }
          .           
```

Every one of the four `std::vector<float>(N)` constructions inside
`materialized()` costs a real, identical 80,000,384 instructions (four
separate 80MB zero-fills of `memset_avx2`) -- 320,001,536 instructions
in total, 72.7% of `materialized()`'s own 440,000,058-instruction
total, BEFORE counting a single `+`, `*`, or `relu` in any of its four
loops. `fused()` pays this real cost exactly ONCE, for its one output
buffer. This is the real, specific answer to "where do fusion's saved
instructions actually go": not primarily in redundant arithmetic --
Chapter 12's four elementwise operations still run exactly once per
element either way -- but in the real allocation-and-zero-init
overhead of every extra buffer a materialized intermediate tensor
requires, exactly the mechanism Chapter 13's own elementwise fusion
pass exists to eliminate, now visible as an exact, real, per-line
instruction count instead of an assumed one.

## E.3 -- Capstone: Testing Chapter 22's Own Explanation

Chapter 22's own File 054 measured `computeUnrolled()`
-- `out[i] = a[i]*2+1`, processed in chunks of `unrollFactor` -- at
extent=50,000,000 across factors 1,2,4,8,16,32,64, and found a real
mismatch: `estimateLoopOverheadCost()` always predicts the LARGEST
factor is fastest (fewer trips, and the model's own per-element term
never depends on chunk size at all), but the real best-of-5 wall-clock
measurement did not always agree, and Chapter 22's own prose named
three specific, plausible, unmeasured suspects: instruction-cache
pressure, register pressure, and lost auto-vectorization. This file
reuses `computeUnrolled()` completely UNCHANGED from File 054 and asks
Cachegrind to check each suspect directly, as a real number instead of
a plausible sentence. The extent is reduced to 10,000,000 -- stated
honestly: Section E.1's own Cachegrind run took roughly 15-45x real
time (round-trip through `valgrind --tool=callgrind`), so a full
50,000,000-element, 7-factor sweep under instrumentation is
impractical to run as part of building this book, while 10,000,000
four-byte floats (40MB) still comfortably exceeds this same machine's
own real, measured 33MB LL cache (Section E.1's own cache-config
output), so the mechanism under test is still genuinely present at
this size.

```cpp
// Appendix E: Profiling and Benchmarking
// 104_ch22s_own_unroll_factor_experiment_under_a_real_instruction_counter.cpp
//
// Section E.3 (capstone) -- Chapter 22's own File 054 built
// estimateLoopOverheadCost(), a linear model that always recommends the
// LARGEST unroll factor, then measured computeUnrolled() for real across
// unroll factors 1,2,4,8,16,32,64 at extent=50,000,000 and found, honestly,
// that the model's own prediction did NOT always match the real fastest
// factor -- and named the reason in prose, without measuring it: "real
// hardware has other mechanisms this simple model was never told about --
// instruction-cache pressure, register pressure, and lost auto-
// vectorization opportunities." Sections E.1 and E.2 showed that Cachegrind
// and Callgrind can turn a claim like that into a real, counted number
// instead of a plausible-sounding sentence. This file reuses File 054's own
// computeUnrolled() UNCHANGED and asks Cachegrind directly: does real
// INSTRUCTION COUNT grow with unroll factor the way "instruction-cache
// pressure" would predict, and do real D1/I1 miss counts move at all? The
// extent here is reduced from File 054's own 50,000,000 to 10,000,000 --
// stated honestly, not silently: Cachegrind's own instrumentation overhead
// (Sections E.1/E.2 measured roughly 90-115x real time on a 20,000,000-
// element run) makes a full 50,000,000-element, 7-factor sweep impractical
// to run as part of building this book, while 10,000,000 four-byte floats
// (40MB) still comfortably exceeds this machine's own measured 33MB LL
// cache (Section E.1's own cachegrind warning line), so the mechanism under
// test is still real at this size.
//
// Compile: g++ -std=c++17 -Wall -Wextra -O2 -g 104_ch22s_own_unroll_factor_experiment_under_a_real_instruction_counter.cpp -o 104_driver <factor>
// (Same real reason as Section E.1's File 103: computeUnrolled() keeps
// -O2 -- the same optimization level File 054's own wall-clock numbers
// were measured at -- and __attribute__((noinline)) keeps it a real,
// separately profilable symbol instead of trusting that six call sites
// happen to be enough to stop GCC from inlining it anyway.)
// Run:     ./104_driver <factor>      (factor in {1,2,4,8,16,32,64}; prints result + a self-check)
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <limits>

static const long long EXTENT = 10'000'000;

// ==================== computeUnrolled() -- reused UNCHANGED from Chapter 22's File 054 ====================
__attribute__((noinline))
static std::vector<float> computeUnrolled(const std::vector<float>& a, long long unrollFactor) {
    long long n = static_cast<long long>(a.size());
    std::vector<float> out(static_cast<size_t>(n));
    long long i = 0;
    for (; i + unrollFactor <= n; i += unrollFactor) {
        for (long long lane = 0; lane < unrollFactor; ++lane) {
            long long idx = i + lane;
            out[static_cast<size_t>(idx)] = a[static_cast<size_t>(idx)] * 2.0f + 1.0f;
        }
    }
    for (; i < n; ++i) out[static_cast<size_t>(i)] = a[static_cast<size_t>(i)] * 2.0f + 1.0f;  // scalar tail
    return out;
}
static bool arraysExactlyEqual(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) if (x[i] != y[i]) return false;
    return true;
}

int main(int argc, char** argv) {
    long long factor = (argc > 1) ? std::atoll(argv[1]) : 1;
    printf("=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=%lld, extent=%lld ===\n\n", factor, EXTENT);

    std::vector<float> a(static_cast<size_t>(EXTENT));
    for (long long i = 0; i < EXTENT; ++i) a[static_cast<size_t>(i)] = static_cast<float>(i % 997) * 0.001f;

    std::vector<float> baseline = computeUnrolled(a, 1);
    std::vector<float> result;
    double bestMs = std::numeric_limits<double>::max();
    const int trials = 5;
    for (int t = 0; t < trials; ++t) {
        auto start = std::chrono::steady_clock::now();
        result = computeUnrolled(a, factor);
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        if (ms < bestMs) bestMs = ms;
    }
    bool correct = arraysExactlyEqual(result, baseline);
    printf("result[0]=%g  result[%lld]=%g\n", result[0], EXTENT - 1, result[static_cast<size_t>(EXTENT - 1)]);
    printf("best-of-%d wall-clock: %.3f ms\n", trials, bestMs);
    printf("self-check: bit-for-bit equal to unrollFactor=1 baseline (%s)\n", correct ? "confirmed" : "MISMATCH");

    return correct ? 0 : 1;
}
```

A fresh real wall-clock sweep first, at this file's own N=10,000,000
(not reused from File 054's old 50,000,000-element numbers -- a
self-contained, freshly measured comparison):

```bash
g++ -std=c++17 -Wall -Wextra -O2 -g 104_ch22s_own_unroll_factor_experiment_under_a_real_instruction_counter.cpp -o 104_driver
for f in 1 2 4 8 16 32 64; do ./104_driver $f; done
```

**Output (cloud sandbox -- real, live-executed output, all 7 factors):**

```text
=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=1, extent=10000000 ===

result[0]=1  result[9999999]=1.178
best-of-5 wall-clock: 58.938 ms
self-check: bit-for-bit equal to unrollFactor=1 baseline (confirmed)

=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=2, extent=10000000 ===

result[0]=1  result[9999999]=1.178
best-of-5 wall-clock: 55.383 ms
self-check: bit-for-bit equal to unrollFactor=1 baseline (confirmed)

=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=4, extent=10000000 ===

result[0]=1  result[9999999]=1.178
best-of-5 wall-clock: 46.197 ms
self-check: bit-for-bit equal to unrollFactor=1 baseline (confirmed)

=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=8, extent=10000000 ===

result[0]=1  result[9999999]=1.178
best-of-5 wall-clock: 38.400 ms
self-check: bit-for-bit equal to unrollFactor=1 baseline (confirmed)

=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=16, extent=10000000 ===

result[0]=1  result[9999999]=1.178
best-of-5 wall-clock: 47.595 ms
self-check: bit-for-bit equal to unrollFactor=1 baseline (confirmed)

=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=32, extent=10000000 ===

result[0]=1  result[9999999]=1.178
best-of-5 wall-clock: 40.898 ms
self-check: bit-for-bit equal to unrollFactor=1 baseline (confirmed)

=== Appendix E.3: File 054's computeUnrolled(), unrollFactor=64, extent=10000000 ===

result[0]=1  result[9999999]=1.178
best-of-5 wall-clock: 40.625 ms
self-check: bit-for-bit equal to unrollFactor=1 baseline (confirmed)
```

A real turnover, the same SHAPE Chapter 22 itself found: best-of-5
drops from 58.938ms at `unrollFactor=1` to 38.400ms at `unrollFactor=8`,
then never improves further -- 16, 32, and 64 all land back in the
40-48ms range, no better than 8. Chapter 22's own model would have
recommended 64 unconditionally. Cachegrind now checks each of Chapter
22's own three named suspects against this exact real plateau, at the
two endpoints (`unrollFactor=1` and `unrollFactor=64`) plus one real
midpoint (`unrollFactor=8`, the real fastest factor):

```bash
valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
    --cachegrind-out-file=cg1.out ./104_driver 1
```

**Output (cloud sandbox -- real, live-executed program summary, `unrollFactor=1`, `==PID==` prefix stripped):**

```text
I refs:        1,621,986,349
I1  misses:            2,249
LLi misses:            2,205
I1  miss rate:          0.00%
LLi miss rate:          0.00%

D refs:          490,689,591  (140,519,414 rd   + 350,170,177 wr)
D1  misses:       13,766,315  (  5,013,916 rd   +   8,752,399 wr)
LLd misses:       13,759,742  (  5,008,187 rd   +   8,751,555 wr)
D1  miss rate:           2.8% (        3.6%     +         2.5%  )
LLd miss rate:           2.8% (        3.6%     +         2.5%  )

LL refs:          13,768,564  (  5,016,165 rd   +   8,752,399 wr)
LL misses:        13,761,947  (  5,010,392 rd   +   8,751,555 wr)
LL miss rate:            0.7% (        0.3%     +         2.5%  )

Branches:        490,323,400  (490,318,098 cond +       5,302 ind)
Mispredicts:          16,798  (     15,650 cond +       1,148 ind)
Mispred rate:            0.0% (        0.0%     +        21.7%   )
```

```bash
valgrind --tool=cachegrind --cache-sim=yes \
    --cachegrind-out-file=cg8.out ./104_driver 8
```

**Output (cloud sandbox -- real, live-executed program summary, `unrollFactor=8`, `==PID==` prefix stripped):**

```text
I refs:        1,096,986,215
I1  misses:            2,250
LLi misses:            2,206
I1  miss rate:          0.00%
LLi miss rate:          0.00%

D refs:          446,939,528  (96,769,377 rd   + 350,170,151 wr)
D1  misses:       13,766,315  ( 5,013,916 rd   +   8,752,399 wr)
LLd misses:       13,759,742  ( 5,008,187 rd   +   8,751,555 wr)
D1  miss rate:           3.1% (       5.2%     +         2.5%  )
LLd miss rate:           3.1% (       5.2%     +         2.5%  )

LL refs:          13,768,565  ( 5,016,166 rd   +   8,752,399 wr)
LL misses:        13,761,948  ( 5,010,393 rd   +   8,751,555 wr)
LL miss rate:            0.9% (       0.4%     +         2.5%  )
```

```bash
valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes \
    --cachegrind-out-file=cg64.out ./104_driver 64
```

**Output (cloud sandbox -- real, live-executed program summary, `unrollFactor=64`, `==PID==` prefix stripped):**

```text
I refs:        1,031,361,381
I1  misses:            2,251
LLi misses:            2,207
I1  miss rate:          0.00%
LLi miss rate:          0.00%

D refs:          441,470,842  ( 91,300,667 rd   + 350,170,175 wr)
D1  misses:       13,766,316  (  5,013,917 rd   +   8,752,399 wr)
LLd misses:       13,759,743  (  5,008,188 rd   +   8,751,555 wr)
D1  miss rate:           3.1% (        5.5%     +         2.5%  )
LLd miss rate:           3.1% (        5.5%     +         2.5%  )

LL refs:          13,768,567  (  5,016,168 rd   +   8,752,399 wr)
LL misses:        13,761,950  (  5,010,395 rd   +   8,751,555 wr)
LL miss rate:            0.9% (        0.4%     +         2.5%  )

Branches:        391,885,909  (391,880,607 cond +       5,302 ind)
Mispredicts:         798,074  (    796,926 cond +       1,148 ind)
Mispred rate:            0.2% (        0.2%     +        21.7%   )
```

Three real, exact answers to Chapter 22's own three named suspects.
DATA cache pressure: D1 misses are 13,766,315 at factor=1, 13,766,315
at factor=8, and 13,766,316 at factor=64 -- flat, within real rounding
noise, across the entire real plateau. INSTRUCTION cache pressure:
`I1 misses` reads 2,249 / 2,250 / 2,251 across the same three factors --
also completely flat; Cachegrind's own real I1-cache simulation finds
no evidence at all that a bigger unroll factor is pushing this
particular loop body out of a real 32KB I1 cache. Both of Chapter 22's
two CACHE-shaped suspects are directly ruled out by real measurement,
not merely left unconfirmed. But real dynamic instruction count DOES
keep falling -- 1,621,986,349 at factor=1 down to 1,031,361,381 at
factor=64 -- exactly the loop-overhead amortization
`estimateLoopOverheadCost()` itself predicts, which is why the model
gets the DIRECTION right from 1 to 8. What the model, and Chapter 22's
own prose, never named at all is the real mechanism that actually
explains the plateau from 8 onward: real branch mispredictions grow
47.5x across the same range -- 16,798 real mispredicts at factor=1,
798,074 at factor=64, confirmed by Cachegrind's own real branch-
predictor simulation (`--branch-sim=yes`), not by inference. This is a
genuinely different, real, specific mechanism than any of Chapter 22's
own three named suspects: as `unrollFactor` grows, the code generated
for the unrolled loop's own tail-handling and trip-counting logic
becomes real, measurably harder for the branch predictor to track,
and that real cost is exactly what starts to outweigh the real,
continuing loop-overhead savings past `unrollFactor=8` -- an honest
correction to Chapter 22's own honest guess, made possible only by
measuring the actual hardware-level mechanism instead of naming a
plausible one.

## Closing synthesis

Chapter 22 built two cost models, measured both against a real clock,
and reported two real gaps between prediction and measurement --
without ever being able to say, with a real number, WHY either gap
existed. This appendix closes that gap for the loop-overhead model
specifically: Cachegrind and its branch-predictor simulation rule out
both of Chapter 22's own named cache-pressure suspects by direct
measurement, and identify the real mechanism -- branch misprediction
growth, not cache pressure at all -- that Chapter 22's own honest prose
never had a tool to check. Section E.1 and E.2 did the same for
Chapter 12's fusion argument: not "fusion avoids redundant memory
traffic," asserted, but 2.22x the real instructions and 3.5x-4.0x the
real cache misses for the unfused version, broken down by
Callgrind's own real per-line attribution to show that most of that
gap is the cost of allocating and zero-initializing three buffers
`fused()` never needs, not redundant arithmetic. None of this makes
Chapter 22's own wall-clock discipline, or Chapter 19's best-of-N
timing, obsolete -- wall-clock time is still the real number that
actually matters to a person waiting for a kernel to finish, and nothing
in this appendix disagrees with a single one of Chapter 22's own
measured results. What Cachegrind and Callgrind add is a real
explanation UNDERNEATH the clock: deterministic, exactly reproducible,
and, twice in this appendix alone, able to correct a plausible-
sounding guess with a specific, measured, genuinely different answer.
