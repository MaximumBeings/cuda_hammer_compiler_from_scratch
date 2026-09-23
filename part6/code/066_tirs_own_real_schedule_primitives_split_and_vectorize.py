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
