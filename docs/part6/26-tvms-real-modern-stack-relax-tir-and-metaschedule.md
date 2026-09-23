# Chapter 26: TVM's Real Modern Stack -- Relax, TIR, and MetaSchedule

Chapter 25 studied XLA by installing the real thing (`pip install jax`)
and asking it, live, what it actually does. This chapter does the same
for TVM -- and the first real finding arrives before a single line of
this chapter's own code runs. Chapter 3's own survey, and this book's own
TOC placeholder for this chapter, both used the names "Relay" and
"Ansor" -- the names most TVM tutorials and even TVM's own older
documentation still use. `pip install apache-tvm` pulls a real, current,
per-architecture wheel (0.26.0, confirmed installable on both this
book's own machines: a genuine x86-64 wheel for the cloud sandbox, a
genuine aarch64 wheel for the device) -- and inspecting it directly
(`dir(tvm)`) shows both of those names are gone. `tvm.relay` does not
exist. `tvm.auto_scheduler` and `tvm.autotvm` do not exist. Even
`tvm.tir`, the namespace every classic TVM tutorial schedules through,
does not exist under that name in this release. What actually ships:
`tvm.relax` (Relay's real successor), `tvm.s_tir` (the real, current
home of the schedulable IR and, inside it, `meta_schedule`), and
`tvm.tirx` (a lower-level IR builder). None of this was known in
advance -- it was found by trying the names Chapter 3's own citation
used and reading the real `ImportError`s that followed. A citation
written before this chapter's own real installation is now itself
slightly out of date. That is a genuine, humbling lesson about
production software's own churn -- one a from-scratch toy compiler like
CUDA Hammer never has to face, because nothing outside this book depends
on `Schedule` staying named `Schedule`.

```text
+------------------------------------------------------------------+
|  Chapter 26's own shape, section by section                      |
|                                                                    |
|  26.1  TVM's real scheduling primitives (split, vectorize) --     |
|        the direct analogue of Ch21's own Schedule, built and      |
|        run through a real LLVM backend.                           |
|                                                                    |
|  26.2  Relax -- Relay's real successor -- run on the exact same   |
|        diamond graph File 063 (Ch25) ran through real XLA, to     |
|        compare fusion boundaries directly.                        |
|                                                                    |
|  26.3  MetaSchedule's own real TuningRecord/Database, the real    |
|        successor to both AutoTVM and Ansor, and the real          |
|        system Chapter 24's own TuningCache citation was about.    |
+------------------------------------------------------------------+
```

## 26.1 TIR's Own Real Schedule Primitives

TVM has scheduled tensor computations since before Relay or Ansor
existed, through a layer this book's own earlier citations called TIR
(Tensor IR). The real, current binding for that schedulable layer in
this release is `tvm.s_tir`, and its `Schedule` class exposes real,
named primitives -- `split`, `reorder`, `parallel`, `vectorize`,
`unroll` -- that read almost like a direct translation of Chapter 21's
own `Schedule{tileSizePerLoop, loopOrder, unrollFactor}` into an API
someone else already built and shipped.

```text
  Ch21's own Schedule struct        TVM's own real s_tir.Schedule
  --------------------------        ------------------------------
  tileSizePerLoop[dim] = 8    -->   sch.split(loop, factors=[6, 8])
  loopOrder = [1, 0]          -->   sch.reorder(loop_1, loop_0)
  unrollFactor = 4            -->   sch.unroll(loop)
  (Ch19's own vecAdd/vecFma)  -->   sch.vectorize(loop)
```

File 066 builds a real TIR `PrimFunc` from a Tensor Expression (`te.compute`,
the same "elementwise add over one loop" shape as every Chapter 21
worked example), applies a real `split(factors=[6, 8])` -- Chapter 21's
own tile-size split, on the exact same `n=48` extent Chapter 21's own
capstone search space used -- followed by a real `vectorize`, then
compiles the scheduled TIR through a real LLVM backend and runs it,
checked against a NumPy reference.
```python
#!/usr/bin/env python3
# Chapter 26, Section 26.1: TVM's own real, foundational scheduling layer --
# the direct analogue of Chapter 21's own `Schedule`/`tileLoop()`, applied
# to a real TIR (Tensor IR) function and actually compiled and run.
#
# A note on API names before anything else: this file targets `apache-tvm`
# 0.26.0, installed for real via `pip install apache-tvm` on both machines
# (a real, current, per-architecture wheel exists for both x86-64 and
# aarch64 -- confirmed by installing it, not assumed). Chapter 3's own
# survey, and this book's own TOC placeholder for this chapter, both used
# the names "Relay" and "Ansor" -- the names TVM's own documentation and
# most tutorials still use. Installing the real, current package and
# inspecting it directly (`dir(tvm)`) shows those names are gone from this
# release: `tvm.relay` does not exist (replaced by `tvm.relax`, confirmed
# below in File 067); `tvm.auto_scheduler` and `tvm.autotvm` do not exist
# (replaced by `tvm.s_tir.meta_schedule`, confirmed in File 068); even
# `tvm.tir` itself is not the classic scheduling namespace anymore -- the
# real, current binding is `tvm.s_tir` for the schedulable ("scheduled
# TIR") layer this file uses, with a separate `tvm.tirx` for the lower-level
# IR builder. None of this was known in advance; it was found by trying to
# import the names this book's own earlier citations used and following the
# real errors to what actually ships. A citation from Chapter 3, made
# before this chapter's own real installation, is now itself slightly out
# of date -- a genuine lesson about production software that a from-scratch
# toy compiler like CUDA Hammer never has to face, since nothing outside
# this book depends on `Schedule` staying named `Schedule`.
#
# Part 1 builds a real TIR PrimFunc from a Tensor Expression (`te.compute`),
# the same "elementwise add over one loop" shape as every Ch21 example.
# Part 2 applies a real `split` (Ch21's own tile-size split) followed by a
# real `vectorize` (this book's own Ch19 SIMD codegen, but TVM's own
# scheduling layer decides it declaratively instead of CUDA Hammer's own
# hand-written `vecAdd()`/`vecFma()` intrinsic wrappers), then compiles the
# scheduled TIR with a real LLVM backend and runs it, checked against a
# NumPy reference -- the same "run it for real, verify it for real"
# discipline as every chapter since Chapter 17.

import numpy as np
import tvm
from tvm import te, s_tir


def build_unscheduled_module(n: int) -> tvm.IRModule:
    a = te.placeholder((n,), name="A", dtype="float32")
    b = te.placeholder((n,), name="B", dtype="float32")
    c = te.compute((n,), lambda i: a[i] + b[i], name="C")
    prim_func = te.create_prim_func([a, b, c])
    return tvm.IRModule({"main": prim_func})


def main():
    n = 48  # the same LoopNest extent Chapter 21's own capstone used
    print(f"tvm version: {tvm.__version__}")
    print()

    print("=" * 78)
    print("PART 1: an unscheduled TIR PrimFunc from a Tensor Expression")
    print("=" * 78)
    mod = build_unscheduled_module(n)
    print(mod)

    print("=" * 78)
    print("PART 2: the same PrimFunc after a real split(factors=[6, 8]) "
          "and vectorize -- Ch21's own tile-size split, TVM's own way")
    print("=" * 78)
    sch = s_tir.Schedule(mod)
    block_c = sch.get_sblock("C")
    (loop_i,) = sch.get_loops(block_c)
    outer, inner = sch.split(loop_i, factors=[6, 8])
    sch.vectorize(inner)
    print(sch.mod)

    print("=" * 78)
    print("PART 3: real LLVM compile + real execution, checked against NumPy")
    print("=" * 78)
    target = tvm.target.Target("llvm")
    built = tvm.compile(sch.mod, target=target)
    dev = tvm.runtime.cpu(0)

    rng = np.random.default_rng(0)
    a_np = rng.uniform(size=n).astype("float32")
    b_np = rng.uniform(size=n).astype("float32")
    a_tvm = tvm.runtime.tensor(a_np, dev)
    b_tvm = tvm.runtime.tensor(b_np, dev)
    c_tvm = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)

    built["main"](a_tvm, b_tvm, c_tvm)
    ref = a_np + b_np
    max_err = float(np.max(np.abs(c_tvm.numpy() - ref)))
    print(f"max abs error vs. NumPy reference: {max_err}")
    print(f"bit-exact match: {max_err == 0.0}")


if __name__ == "__main__":
    main()
```

```bash
python3 "066_tirs_own_real_schedule_primitives_split_and_vectorize.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated TVM output):**

```text
tvm version: 0.26.0

==============================================================================
PART 1: an unscheduled TIR PrimFunc from a Tensor Expression
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main(A: T.Buffer((48,), "float32"), B: T.Buffer((48,), "float32"), C: T.Buffer((48,), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i in range(48):
            with T.sblock("C"):
                v_i = T.axis.spatial(48, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = A[v_i] + B[v_i]
==============================================================================
PART 2: the same PrimFunc after a real split(factors=[6, 8]) and vectorize -- Ch21's own tile-size split, TVM's own way
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main(A: T.Buffer((48,), "float32"), B: T.Buffer((48,), "float32"), C: T.Buffer((48,), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i_0 in range(6):
            for i_1 in T.vectorized(8):
                with T.sblock("C"):
                    v_i = T.axis.spatial(48, i_0 * 8 + i_1)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] + B[v_i]
==============================================================================
PART 3: real LLVM compile + real execution, checked against NumPy
==============================================================================
max abs error vs. NumPy reference: 0.0
bit-exact match: True
```

**Output (device, aarch64 Linux VM -- real, live-generated TVM output):**

```text
tvm version: 0.26.0

==============================================================================
PART 1: an unscheduled TIR PrimFunc from a Tensor Expression
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main(A: T.Buffer((48,), "float32"), B: T.Buffer((48,), "float32"), C: T.Buffer((48,), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i in range(48):
            with T.sblock("C"):
                v_i = T.axis.spatial(48, i)
                T.reads(A[v_i], B[v_i])
                T.writes(C[v_i])
                C[v_i] = A[v_i] + B[v_i]
==============================================================================
PART 2: the same PrimFunc after a real split(factors=[6, 8]) and vectorize -- Ch21's own tile-size split, TVM's own way
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main(A: T.Buffer((48,), "float32"), B: T.Buffer((48,), "float32"), C: T.Buffer((48,), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        for i_0 in range(6):
            for i_1 in T.vectorized(8):
                with T.sblock("C"):
                    v_i = T.axis.spatial(48, i_0 * 8 + i_1)
                    T.reads(A[v_i], B[v_i])
                    T.writes(C[v_i])
                    C[v_i] = A[v_i] + B[v_i]
==============================================================================
PART 3: real LLVM compile + real execution, checked against NumPy
==============================================================================
max abs error vs. NumPy reference: 0.0
bit-exact match: True
```

*Both machines agree exactly, on both the scheduled TIR text and the bit-exact correctness check -- a fully deterministic result, the same as Chapter 21's own schedule-enumeration chapter.*


### What TVM's own scheduling API doesn't ask of the caller

The most immediately striking thing about File 066's own Part 2 output
is what is absent: no hand-written flat-index arithmetic. Chapter 23's
own `generateScheduleFunction()` had to compute `i_0 * 8 + i_1` itself,
as C++ text, and get it right for every combination of tile size and
loop order by hand. TVM's own `sch.split()` prints that exact same
expression (`v_i = T.axis.spatial(48, i_0 * 8 + i_1)`) -- but CUDA
Hammer's own codegen has to derive it, while TVM's own scheduling layer
derives it FOR the caller, as a byproduct of a single declarative
`split()` call. This is the real, concrete shape of what "a decade of
production engineering" buys: not a different idea (both systems
tile the same loop the same way) but a layer of machinery that makes
the idea safe to apply without re-deriving the index math by hand every
time -- exactly the gap this book's own Chapter 25 closing section named
between a toy autotuner and a production one.

```bash
python3 "066_tirs_own_real_schedule_primitives_split_and_vectorize.py"
```

```python
#!/usr/bin/env python3
# Chapter 26, Section 26.2: CUDA Hammer's own Ch4/Ch13 diamond graph, this
# time run through TVM's real, current graph-level IR -- Relax, not Relay
# (see File 066's own header for how that was discovered, not assumed).
#
# Part 1 builds the diamond graph (t1 = a+b feeds t2 = t1*c and t3 =
# relu(t1); out = t2+t3) directly in Relax and prints its unoptimized IR.
#
# Part 2 lowers it through TVM's own real fusion pipeline. A first attempt
# at this (not shown here, but honestly reported: see this chapter's own
# prose) called only `FuseOps()` and found it fused NOTHING -- not even a
# plain, unshared linear chain with no sharing at all. The real reason,
# found by listing every name in `relax.transform` and reading the one that
# fit, is that `FuseOps()` groups PrimFuncs by an `op_pattern` ATTRIBUTE
# (elementwise, broadcast, injective, reduction, opaque) that nothing
# assigns by default -- `AnnotateTIROpPattern()` has to run first. The real
# pipeline is `LegalizeOps -> AnnotateTIROpPattern -> FuseOps -> FuseTIR`,
# and Part 2 uses exactly that, on the exact same diamond graph File 063
# (Chapter 25) ran through real XLA.
#
# Part 3 asks the same escape-boundary question Chapter 25's own File 063
# asked of XLA: does making t1 ALSO a live output change what gets fused?
#
# Part 4 builds and runs the fused module for real, checked against NumPy.

import numpy as np
import tvm
from tvm import relax
from tvm.script import ir as I, relax as R


@I.ir_module
class DiamondModule:
    @R.function
    def main(a: R.Tensor((6, 8), "float32"), b: R.Tensor((6, 8), "float32"),
             c: R.Tensor((6, 8), "float32")):
        with R.dataflow():
            t1 = R.add(a, b)
            t2 = R.multiply(t1, c)
            t3 = R.nn.relu(t1)
            out = R.add(t2, t3)
            R.output(out)
        return out


@I.ir_module
class DiamondModuleMultiOutput:
    @R.function
    def main(a: R.Tensor((6, 8), "float32"), b: R.Tensor((6, 8), "float32"),
             c: R.Tensor((6, 8), "float32")):
        with R.dataflow():
            t1 = R.add(a, b)
            t2 = R.multiply(t1, c)
            t3 = R.nn.relu(t1)
            out = R.add(t2, t3)
            R.output(out, t1)
        return (out, t1)


def run_real_fusion_pipeline(mod: tvm.IRModule) -> tvm.IRModule:
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    return mod


def count_primfuncs(mod: tvm.IRModule) -> int:
    return sum(1 for gv in mod.functions if gv.name_hint != "main")


def main():
    print(f"tvm version: {tvm.__version__}")
    print()

    print("=" * 78)
    print("PART 1: unoptimized Relax IR for the diamond graph")
    print("=" * 78)
    print(DiamondModule)

    print("=" * 78)
    print("PART 2: after the real fusion pipeline (LegalizeOps -> "
          "AnnotateTIROpPattern -> FuseOps -> FuseTIR)")
    print("=" * 78)
    fused = run_real_fusion_pipeline(DiamondModule)
    print(fused)
    n = count_primfuncs(fused)
    print(f"-- PrimFunc(s) besides main: {n}")
    print(f"-- t1's own computation lives INSIDE that one fused PrimFunc's "
          f"own local buffers (sblock_alloc_buffer), not passed back out "
          f"through main's own parameters: {n == 1}")

    print()
    print("=" * 78)
    print("PART 3: same graph, but t1 is ALSO returned as a live output")
    print("=" * 78)
    fused_mo = run_real_fusion_pipeline(DiamondModuleMultiOutput)
    print(fused_mo)
    n_mo = count_primfuncs(fused_mo)
    print(f"-- PrimFunc(s) besides main once t1 escapes as an output: {n_mo}")

    print()
    print("=" * 78)
    print("PART 4: real LLVM compile + real execution, checked against NumPy")
    print("=" * 78)
    target = tvm.target.Target("llvm")
    ex = tvm.compile(fused, target=target)
    dev = tvm.runtime.cpu(0)
    vm = relax.VirtualMachine(ex, dev)

    rng = np.random.default_rng(0)
    a_np = rng.uniform(size=(6, 8)).astype("float32")
    b_np = rng.uniform(size=(6, 8)).astype("float32")
    c_np = rng.uniform(size=(6, 8)).astype("float32")
    a_tvm = tvm.runtime.tensor(a_np, dev)
    b_tvm = tvm.runtime.tensor(b_np, dev)
    c_tvm = tvm.runtime.tensor(c_np, dev)

    result = vm["main"](a_tvm, b_tvm, c_tvm)
    t1_ref = a_np + b_np
    ref = (t1_ref * c_np) + np.maximum(t1_ref, 0.0)
    max_err = float(np.max(np.abs(result.numpy() - ref)))
    print(f"max abs error vs. NumPy reference: {max_err}")
    print(f"bit-exact match: {max_err == 0.0}")


if __name__ == "__main__":
    main()
```

```bash
python3 "067_relax_the_real_successor_to_relay_and_its_own_fusion_boundary.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated TVM output):**

```text
tvm version: 0.26.0

==============================================================================
PART 1: unoptimized Relax IR for the diamond graph
==============================================================================
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(a: R.Tensor((6, 8), dtype="float32"), b: R.Tensor((6, 8), dtype="float32"), c: R.Tensor((6, 8), dtype="float32")) -> R.Tensor((6, 8), dtype="float32"):
        with R.dataflow():
            t1: R.Tensor((6, 8), dtype="float32") = R.add(a, b)
            t2: R.Tensor((6, 8), dtype="float32") = R.multiply(t1, c)
            t3: R.Tensor((6, 8), dtype="float32") = R.nn.relu(t1)
            out: R.Tensor((6, 8), dtype="float32") = R.add(t2, t3)
            R.output(out)
        return out
==============================================================================
PART 2: after the real fusion pipeline (LegalizeOps -> AnnotateTIROpPattern -> FuseOps -> FuseTIR)
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True, s_tir=True)
    def fused_add_multiply_relu_add(a: T.Buffer((T.int64(6), T.int64(8)), "float32"), b: T.Buffer((T.int64(6), T.int64(8)), "float32"), c: T.Buffer((T.int64(6), T.int64(8)), "float32"), T_add_intermediate_1: T.Buffer((T.int64(6), T.int64(8)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        T_add_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        T_multiply_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        compute_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(a[v_ax0, v_ax1], b[v_ax0, v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = a[v_ax0, v_ax1] + b[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_add_intermediate[v_ax0, v_ax1], c[v_ax0, v_ax1])
                T.writes(T_multiply_intermediate[v_ax0, v_ax1])
                T_multiply_intermediate[v_ax0, v_ax1] = T_add_intermediate[v_ax0, v_ax1] * c[v_ax0, v_ax1]
        for i0, i1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add_intermediate[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(T_add_intermediate[v_i0, v_i1], T.float32(0.0))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_multiply_intermediate[v_ax0, v_ax1], compute_intermediate[v_ax0, v_ax1])
                T.writes(T_add_intermediate_1[v_ax0, v_ax1])
                T_add_intermediate_1[v_ax0, v_ax1] = T_multiply_intermediate[v_ax0, v_ax1] + compute_intermediate[v_ax0, v_ax1]

    @R.function
    def main(a: R.Tensor((6, 8), dtype="float32"), b: R.Tensor((6, 8), dtype="float32"), c: R.Tensor((6, 8), dtype="float32")) -> R.Tensor((6, 8), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_add_multiply_relu_add, (a, b, c), out_ty=R.Tensor((6, 8), dtype="float32"))
            R.output(gv)
        return gv
-- PrimFunc(s) besides main: 1
-- t1's own computation lives INSIDE that one fused PrimFunc's own local buffers (sblock_alloc_buffer), not passed back out through main's own parameters: True

==============================================================================
PART 3: same graph, but t1 is ALSO returned as a live output
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True, s_tir=True)
    def add(a: T.Buffer((T.int64(6), T.int64(8)), "float32"), b: T.Buffer((T.int64(6), T.int64(8)), "float32"), T_add: T.Buffer((T.int64(6), T.int64(8)), "float32")):
        T.func_attr({"op_pattern": 0, "tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(a[v_ax0, v_ax1], b[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = a[v_ax0, v_ax1] + b[v_ax0, v_ax1]

    @T.prim_func(private=True, s_tir=True)
    def fused_multiply_relu_add(t1: T.Buffer((T.int64(6), T.int64(8)), "float32"), c: T.Buffer((T.int64(6), T.int64(8)), "float32"), T_add_intermediate: T.Buffer((T.int64(6), T.int64(8)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        T_multiply_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        compute_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(t1[v_ax0, v_ax1], c[v_ax0, v_ax1])
                T.writes(T_multiply_intermediate[v_ax0, v_ax1])
                T_multiply_intermediate[v_ax0, v_ax1] = t1[v_ax0, v_ax1] * c[v_ax0, v_ax1]
        for i0, i1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(t1[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(t1[v_i0, v_i1], T.float32(0.0))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_multiply_intermediate[v_ax0, v_ax1], compute_intermediate[v_ax0, v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = T_multiply_intermediate[v_ax0, v_ax1] + compute_intermediate[v_ax0, v_ax1]

    @R.function
    def main(a: R.Tensor((6, 8), dtype="float32"), b: R.Tensor((6, 8), dtype="float32"), c: R.Tensor((6, 8), dtype="float32")) -> R.Tuple(R.Tensor((6, 8), dtype="float32"), R.Tensor((6, 8), dtype="float32")):
        cls = Module
        with R.dataflow():
            t1 = R.call_tir(cls.add, (a, b), out_ty=R.Tensor((6, 8), dtype="float32"))
            gv = R.call_tir(cls.fused_multiply_relu_add, (t1, c), out_ty=R.Tensor((6, 8), dtype="float32"))
            R.output(t1, gv)
        return (gv, t1)
-- PrimFunc(s) besides main once t1 escapes as an output: 2

==============================================================================
PART 4: real LLVM compile + real execution, checked against NumPy
==============================================================================
max abs error vs. NumPy reference: 0.0
bit-exact match: True
```

**Output (device, aarch64 Linux VM -- real, live-generated TVM output):**

```text
tvm version: 0.26.0

==============================================================================
PART 1: unoptimized Relax IR for the diamond graph
==============================================================================
# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(a: R.Tensor((6, 8), dtype="float32"), b: R.Tensor((6, 8), dtype="float32"), c: R.Tensor((6, 8), dtype="float32")) -> R.Tensor((6, 8), dtype="float32"):
        with R.dataflow():
            t1: R.Tensor((6, 8), dtype="float32") = R.add(a, b)
            t2: R.Tensor((6, 8), dtype="float32") = R.multiply(t1, c)
            t3: R.Tensor((6, 8), dtype="float32") = R.nn.relu(t1)
            out: R.Tensor((6, 8), dtype="float32") = R.add(t2, t3)
            R.output(out)
        return out
==============================================================================
PART 2: after the real fusion pipeline (LegalizeOps -> AnnotateTIROpPattern -> FuseOps -> FuseTIR)
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True, s_tir=True)
    def fused_add_multiply_relu_add(a: T.Buffer((T.int64(6), T.int64(8)), "float32"), b: T.Buffer((T.int64(6), T.int64(8)), "float32"), c: T.Buffer((T.int64(6), T.int64(8)), "float32"), T_add_intermediate_1: T.Buffer((T.int64(6), T.int64(8)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        T_add_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        T_multiply_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        compute_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(a[v_ax0, v_ax1], b[v_ax0, v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = a[v_ax0, v_ax1] + b[v_ax0, v_ax1]
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_add_intermediate[v_ax0, v_ax1], c[v_ax0, v_ax1])
                T.writes(T_multiply_intermediate[v_ax0, v_ax1])
                T_multiply_intermediate[v_ax0, v_ax1] = T_add_intermediate[v_ax0, v_ax1] * c[v_ax0, v_ax1]
        for i0, i1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(T_add_intermediate[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(T_add_intermediate[v_i0, v_i1], T.float32(0.0))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add1"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_multiply_intermediate[v_ax0, v_ax1], compute_intermediate[v_ax0, v_ax1])
                T.writes(T_add_intermediate_1[v_ax0, v_ax1])
                T_add_intermediate_1[v_ax0, v_ax1] = T_multiply_intermediate[v_ax0, v_ax1] + compute_intermediate[v_ax0, v_ax1]

    @R.function
    def main(a: R.Tensor((6, 8), dtype="float32"), b: R.Tensor((6, 8), dtype="float32"), c: R.Tensor((6, 8), dtype="float32")) -> R.Tensor((6, 8), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.fused_add_multiply_relu_add, (a, b, c), out_ty=R.Tensor((6, 8), dtype="float32"))
            R.output(gv)
        return gv
-- PrimFunc(s) besides main: 1
-- t1's own computation lives INSIDE that one fused PrimFunc's own local buffers (sblock_alloc_buffer), not passed back out through main's own parameters: True

==============================================================================
PART 3: same graph, but t1 is ALSO returned as a live output
==============================================================================
# from tvm.script import ir as I
# from tvm.script import tirx as T
# from tvm.tirx.layout import Axis
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True, s_tir=True)
    def add(a: T.Buffer((T.int64(6), T.int64(8)), "float32"), b: T.Buffer((T.int64(6), T.int64(8)), "float32"), T_add: T.Buffer((T.int64(6), T.int64(8)), "float32")):
        T.func_attr({"op_pattern": 0, "tirx.noalias": True})
        # with T.sblock("root"):
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(a[v_ax0, v_ax1], b[v_ax0, v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = a[v_ax0, v_ax1] + b[v_ax0, v_ax1]

    @T.prim_func(private=True, s_tir=True)
    def fused_multiply_relu_add(t1: T.Buffer((T.int64(6), T.int64(8)), "float32"), c: T.Buffer((T.int64(6), T.int64(8)), "float32"), T_add_intermediate: T.Buffer((T.int64(6), T.int64(8)), "float32")):
        T.func_attr({"tirx.noalias": True})
        # with T.sblock("root"):
        T_multiply_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        compute_intermediate = T.sblock_alloc_buffer((T.int64(6), T.int64(8)))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_multiply"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(t1[v_ax0, v_ax1], c[v_ax0, v_ax1])
                T.writes(T_multiply_intermediate[v_ax0, v_ax1])
                T_multiply_intermediate[v_ax0, v_ax1] = t1[v_ax0, v_ax1] * c[v_ax0, v_ax1]
        for i0, i1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(t1[v_i0, v_i1])
                T.writes(compute_intermediate[v_i0, v_i1])
                compute_intermediate[v_i0, v_i1] = T.max(t1[v_i0, v_i1], T.float32(0.0))
        for ax0, ax1 in T.grid(T.int64(6), T.int64(8)):
            with T.sblock("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_multiply_intermediate[v_ax0, v_ax1], compute_intermediate[v_ax0, v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = T_multiply_intermediate[v_ax0, v_ax1] + compute_intermediate[v_ax0, v_ax1]

    @R.function
    def main(a: R.Tensor((6, 8), dtype="float32"), b: R.Tensor((6, 8), dtype="float32"), c: R.Tensor((6, 8), dtype="float32")) -> R.Tuple(R.Tensor((6, 8), dtype="float32"), R.Tensor((6, 8), dtype="float32")):
        cls = Module
        with R.dataflow():
            t1 = R.call_tir(cls.add, (a, b), out_ty=R.Tensor((6, 8), dtype="float32"))
            gv = R.call_tir(cls.fused_multiply_relu_add, (t1, c), out_ty=R.Tensor((6, 8), dtype="float32"))
            R.output(t1, gv)
        return (gv, t1)
-- PrimFunc(s) besides main once t1 escapes as an output: 2

==============================================================================
PART 4: real LLVM compile + real execution, checked against NumPy
==============================================================================
max abs error vs. NumPy reference: 1.1920928955078125e-07
bit-exact match: False
```

*Both machines agree on every structural finding (one fused PrimFunc for the diamond graph; two once t1 escapes as an output). They diverge in Part 4's own correctness check -- the cloud sandbox is bit-exact, the device lands one float32 ULP away -- a real, honestly reported cross-architecture floating-point divergence, discussed above.*


## 26.2 Relax: Relay's Real Successor, and Its Own Fusion Boundary

File 067 runs the exact same diamond graph File 063 (Chapter 25) built
in JAX -- `t1 = a + b` feeding two consumers, `t2 = t1 * c` and `t3 =
relu(t1)`, joining at `out = t2 + t3` -- through Relax, TVM's own
current graph-level IR.

A first version of this file's own Part 2 called only
`relax.transform.FuseOps()` and found it fused *nothing at all* -- not
even a plain, unshared linear chain with no sharing whatsoever. Reading
every name in `relax.transform` in turn found the real reason:
`FuseOps()` groups PrimFuncs by an `op_pattern` attribute (elementwise,
broadcast, injective, reduction, opaque) that nothing assigns by
default; `AnnotateTIROpPattern()` has to run first. The real pipeline --
`LegalizeOps -> AnnotateTIROpPattern -> FuseOps -> FuseTIR` -- is what
File 067's own Part 2 actually runs.

With the real pipeline in place, real Relax fuses the WHOLE diamond
graph -- including the shared `t1` -- into one PrimFunc, the same real
result Chapter 25 found for real XLA on the identical graph. Making
`t1` also a live output (Part 3, the same test File 063's own Part 3
ran) splits it back out into its own separate PrimFunc, again matching
XLA's own real behavior on the identical test. Two independent real
production compilers, inspected the same way, drew their fusion
boundary in the same place: not Chapter 13's own simple `consumers != 1`
rule, but "does every consumer of this value stay inside the group."

Where the two real systems diverge is HOW the fused kernel is written,
not WHERE the boundary falls. Real XLA's own `kind=kLoop` fusion (Ch25)
computed `t1` as a pure in-register SSA value, substituted directly into
both consumer expressions with no named intermediate at all. Real Relax's
own `FuseTIR()` instead keeps each original op's own named buffer
INSIDE the fused PrimFunc -- `T_add_intermediate`, `T_multiply_intermediate`,
`compute_intermediate`, allocated with `T.sblock_alloc_buffer()` -- still
never round-tripping through the outer function's own parameters or main
memory, but organized as a small pipeline of local buffers rather than a
single flattened expression. Same real boundary rule, two different
real lowering strategies for what happens once a value is inside it.

Part 4 builds and runs the fused module for real on both machines. The
cloud sandbox matches its own NumPy reference bit-for-bit. The device
does not: it lands one float32 ULP away (max abs error
`1.1920928955078125e-07`), a real, honest, cross-architecture
floating-point divergence in a fused multiply/add/relu/add chain that
Chapter 21's own float32-non-associativity finding already predicted
could happen somewhere in this book -- this is simply the first time it
showed up across ARCHITECTURES rather than across ELEMENT ORDERS.
Reported here plainly, the same as every other real machine-specific
finding since Chapter 22.

```bash
python3 "067_relax_the_real_successor_to_relay_and_its_own_fusion_boundary.py"
```

```python
#!/usr/bin/env python3
# Chapter 26, Section 26.3 (capstone): TVM's own real, current auto-tuning
# system -- MetaSchedule -- which the real, installed package confirms has
# replaced BOTH AutoTVM and Ansor (`tvm.autotvm` and `tvm.auto_scheduler`
# do not exist in this real 0.26.0 release; `tvm.s_tir.meta_schedule` does).
# Chapter 24's own citation of "TVM's own AutoTVM tuning-log practice" was
# accurate to the real system at the time that citation was written -- this
# section brings the citation current, the same honest update File 066's
# own header gave to Chapter 3's "Relay"/"Ansor" naming.
#
# OpenXLA's real determinism docs (Chapter 25) described XLA's own
# autotuner as: profile real candidates on real hardware, pick the fastest.
# TVM's own real MetaSchedule docs describe the same three-part shape by
# name: a CostModel (default XGBoost) ranks candidates cheaply, a
# SearchStrategy (default EvolutionarySearch) sends the promising ones to a
# Builder/Runner for REAL measurement, and a Database (default
# JSONDatabase) persists the winning "trace + measured run time" --
# TVM's own real name for exactly what Chapter 24's own `TuningCache`
# built by hand: `loopNestKey(nest) = serializeSchedule(schedule)` lines in
# a plain-text file.
#
# This sandbox has no GPU (same limitation as every chapter since Ch18),
# and a full EvolutionarySearch run is not a quick, small, reproducible
# demo the way Files 066/067's own experiments were -- so, matching Chapter
# 25.3's own honest choice when the same situation came up for XLA's
# autotuner, this file does not run a full auto-search. What it DOES do
# for real: build a real schedule (File 066's own split+vectorize),
# measure it for real on THIS machine, and commit that real measurement to
# a real MetaSchedule `Database` -- TVM's own actual `TuningRecord` and
# `JSONDatabase` classes, not a hand-rolled stand-in -- then load it back
# with a fresh `Database` object, the same MISS-then-HIT shape as Chapter
# 24.1's own `TuningCache` demonstration, using TVM's own real persistence
# format instead of this book's own.
#
# One real design difference surfaced while getting this file's own Part 2
# to work at all, and is worth naming rather than quietly fixing: this
# file's own FIRST attempt committed the SCHEDULED module (after split and
# vectorize) as the workload, and the real `JSONDatabase` constructor threw
# a real, internal consistency error reopening it. The reason, found by
# printing the record file's own real content: a `TuningRecord`'s `trace`
# field is a REPLAY LOG (`GetSBlock`, `GetLoops`, `Split`, `Vectorize`, in
# order), not a snapshot -- so the `workload` half of a record has to be
# the ORIGINAL, UNSCHEDULED module the trace starts from, not the already-
# scheduled result. A real, notable design difference from Chapter 24's own
# `TuningCache`, which stores a fully-specified `Schedule{tileSizePerLoop,
# loopOrder, unrollFactor}` struct directly: TVM's own real persistence
# format stores a starting point plus a sequence of edits, and replays the
# edits to reconstruct the winning schedule, rather than storing the
# winning schedule's own final shape. Part 2 below uses the corrected,
# real design -- the UNSCHEDULED module as the workload.

import time
import numpy as np
import tvm
from tvm import te, s_tir
from tvm.s_tir import meta_schedule as ms


def build_scheduled_module(n: int):
    a = te.placeholder((n,), name="A", dtype="float32")
    b = te.placeholder((n,), name="B", dtype="float32")
    c = te.compute((n,), lambda i: a[i] + b[i], name="C")
    unscheduled_mod = tvm.IRModule({"main": te.create_prim_func([a, b, c])})
    sch = s_tir.Schedule(unscheduled_mod)
    block_c = sch.get_sblock("C")
    (loop_i,) = sch.get_loops(block_c)
    _outer, inner = sch.split(loop_i, factors=[6, 8])
    sch.vectorize(inner)
    return sch, unscheduled_mod


def main():
    print(f"tvm version: {tvm.__version__}")
    print(f"real meta_schedule module: {ms}")
    print(f"real MetaSchedule Database classes available: "
          f"{[c for c in ('Database', 'JSONDatabase', 'TuningRecord') if hasattr(ms.database, c)]}")
    print()

    n = 48
    sch, unscheduled_mod = build_scheduled_module(n)

    print("=" * 78)
    print("PART 1: measure File 066's own real split+vectorize schedule, "
          "for real, on this machine")
    print("=" * 78)
    target = tvm.target.Target("llvm")
    built = tvm.compile(sch.mod, target=target)
    dev = tvm.runtime.cpu(0)
    rng = np.random.default_rng(0)
    a_tvm = tvm.runtime.tensor(rng.uniform(size=n).astype("float32"), dev)
    b_tvm = tvm.runtime.tensor(rng.uniform(size=n).astype("float32"), dev)
    c_tvm = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)

    reps = 200_000
    start = time.perf_counter()
    for _ in range(reps):
        built["main"](a_tvm, b_tvm, c_tvm)
    real_secs_per_call = (time.perf_counter() - start) / reps
    print(f"real measured time per call: {real_secs_per_call * 1e6:.4f} us "
          f"(best of 1 pass, {reps} reps)")

    print()
    print("=" * 78)
    print("PART 2: commit that real measurement to a real MetaSchedule "
          "JSONDatabase -- a real TuningRecord, not a hand-rolled one")
    print("=" * 78)
    ws_path = "/tmp/ch26_ms_database_workload.json"
    rec_path = "/tmp/ch26_ms_database_tuning_record.json"
    import os
    for p in (ws_path, rec_path):
        if os.path.exists(p):
            os.remove(p)

    database = ms.database.JSONDatabase(path_workload=ws_path,
                                         path_tuning_record=rec_path)
    # The workload is the ORIGINAL, UNSCHEDULED module -- the real starting
    # point sch.trace's own real replay log (GetSBlock/GetLoops/Split/
    # Vectorize) was recorded against. See this file's own header for the
    # real error this file's first draft hit by committing sch.mod (the
    # already-scheduled module) here instead.
    workload = database.commit_workload(unscheduled_mod)
    record = ms.database.TuningRecord(
        sch.trace, workload, run_secs=[real_secs_per_call], target=target,
    )
    database.commit_tuning_record(record)
    print(f"committed a real TuningRecord (run_secs={real_secs_per_call:.8f}) "
          f"to a real JSONDatabase at {rec_path}")

    print()
    print("=" * 78)
    print("PART 3: a FRESH Database object, loaded from the exact files "
          "the first one wrote -- Ch24.1's own MISS-then-HIT shape, "
          "TVM's own real persistence format")
    print("=" * 78)
    fresh_database = ms.database.JSONDatabase(path_workload=ws_path,
                                               path_tuning_record=rec_path)
    fresh_workload = fresh_database.commit_workload(unscheduled_mod)
    records = fresh_database.get_top_k(fresh_workload, top_k=1)
    print(f"fresh Database, same workload -> {len(records)} record(s) found")
    if records:
        recovered_secs = records[0].run_secs
        print(f"recovered run_secs: {recovered_secs}")
        traces_match = str(records[0].trace) == str(sch.trace)
        print(f"recovered trace matches the original schedule's own trace: "
              f"{traces_match}")


if __name__ == "__main__":
    main()
```

```bash
python3 "068_metaschedules_real_tuningrecord_database_ch24s_own_cache_idea_at_production_scale.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated TVM output):**

```text
tvm version: 0.26.0
real meta_schedule module: <module 'tvm.s_tir.meta_schedule' from '/usr/local/lib/python3.11/dist-packages/tvm/s_tir/meta_schedule/__init__.py'>
real MetaSchedule Database classes available: ['Database', 'JSONDatabase', 'TuningRecord']

==============================================================================
PART 1: measure File 066's own real split+vectorize schedule, for real, on this machine
==============================================================================
real measured time per call: 15.3638 us (best of 1 pass, 200000 reps)

==============================================================================
PART 2: commit that real measurement to a real MetaSchedule JSONDatabase -- a real TuningRecord, not a hand-rolled one
==============================================================================
committed a real TuningRecord (run_secs=0.00001536) to a real JSONDatabase at /tmp/ch26_ms_database_tuning_record.json

==============================================================================
PART 3: a FRESH Database object, loaded from the exact files the first one wrote -- Ch24.1's own MISS-then-HIT shape, TVM's own real persistence format
==============================================================================
fresh Database, same workload -> 1 record(s) found
recovered run_secs: (T.float32(1.5363752914999795e-05),)
recovered trace matches the original schedule's own trace: True
```

**Output (device, aarch64 Linux VM -- real, live-generated TVM output):**

```text
tvm version: 0.26.0
real meta_schedule module: <module 'tvm.s_tir.meta_schedule' from '/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/.local/lib/python3.10/site-packages/tvm/s_tir/meta_schedule/__init__.py'>
real MetaSchedule Database classes available: ['Database', 'JSONDatabase', 'TuningRecord']

==============================================================================
PART 1: measure File 066's own real split+vectorize schedule, for real, on this machine
==============================================================================
real measured time per call: 82.4305 us (best of 1 pass, 200000 reps)

==============================================================================
PART 2: commit that real measurement to a real MetaSchedule JSONDatabase -- a real TuningRecord, not a hand-rolled one
==============================================================================
committed a real TuningRecord (run_secs=0.00008243) to a real JSONDatabase at /tmp/ch26_ms_database_tuning_record.json

==============================================================================
PART 3: a FRESH Database object, loaded from the exact files the first one wrote -- Ch24.1's own MISS-then-HIT shape, TVM's own real persistence format
==============================================================================
fresh Database, same workload -> 1 record(s) found
recovered run_secs: (T.float32(8.2430530665005789e-05),)
recovered trace matches the original schedule's own trace: True
```

*Both machines commit and recover a real TuningRecord correctly. The measured run_secs figures differ, as every real timing figure in this book has since Chapter 22 -- real, machine-specific data, not a discrepancy to reconcile.*


## 26.3 MetaSchedule: the Real Successor to Both AutoTVM and Ansor

`tvm.autotvm` and `tvm.auto_scheduler` do not exist in this real,
installed release -- `tvm.s_tir.meta_schedule` does, and its own real
documentation confirms why: MetaSchedule is TVM's current auto-tuning
system, and it is built from the same three real pieces both of this
book's own citations already separately named. A `CostModel` (default
XGBoost) ranks candidates cheaply -- Chapter 22's own
`estimateScheduleCost()`, at production scale, with a learned model
standing in for Chapter 22's own two hand-derived linear ones. A
`SearchStrategy` (default EvolutionarySearch) sends the promising
candidates to a `Builder`/`Runner` for REAL measurement on real hardware
-- Chapter 23.3's own hybrid autotuner's own second pass. A `Database`
(default `JSONDatabase`) persists the winning "trace + measured run
time" for reuse -- Chapter 24's own `TuningCache`, TVM's own real name
for the same idea Chapter 24 built by hand and cited TVM's own AutoTVM
tuning-log practice as the precedent for. Ansor itself, TVM's own 2021
post on it explains, existed to fix AutoTVM's own worst limitation:
hand-written per-operator templates ("more than 15k lines of code for
these templates in the TVM code repository") replaced by automatic
search-space construction from general rules. MetaSchedule is the
system that inherited both lineages.

File 068 measures File 066's own real split+vectorize schedule for real
on each machine, then commits that real measurement to a real
`JSONDatabase` -- TVM's own actual `TuningRecord` class, not a
hand-rolled stand-in -- and loads it back with a fresh `Database`
object, the same MISS-then-HIT shape as Chapter 24.1's own `TuningCache`
demonstration.

Getting that commit to work at all surfaced a real, worth-naming design
difference. This file's own first attempt committed the SCHEDULED
module (after `split`/`vectorize`) as the workload, and the real
`JSONDatabase` constructor threw a real internal consistency error on
reopening it. Printing the record file's own real content showed why: a
`TuningRecord`'s own `trace` field is a REPLAY LOG --
`["GetSBlock",...]`, `["GetLoops",...]`, `["Split",...]`,
`["Vectorize",...]`, in order -- not a snapshot of the final schedule.
The `workload` half of a record has to be the ORIGINAL, UNSCHEDULED
module the trace starts from; committing the already-scheduled module
broke the replay's own consistency check. This is a real, substantive
difference from Chapter 24's own `TuningCache`, which stores a
fully-specified `Schedule` struct directly: TVM's own real persistence
format instead stores a starting point plus a sequence of edits, and
replays the edits to reconstruct the winning schedule on demand, rather
than storing the winning shape itself. File 068's own Part 2 uses the
corrected, real design.

```bash
python3 "068_metaschedules_real_tuningrecord_database_ch24s_own_cache_idea_at_production_scale.py"
```
