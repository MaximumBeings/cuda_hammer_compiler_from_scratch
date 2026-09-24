#!/usr/bin/env python3
# Appendix F: From torch.compile and TVM's Python API to C++: A Rosetta
# Stone
# 105_the_same_diamond_graph_through_tvm_relaxs_own_real_fusion_pipeline.py
#
# Section F.1 -- the TVM half of this appendix's first side-by-side pair.
# Reuses Chapter 26's own File 067 completely unchanged (the same real
# Relax pipeline, LegalizeOps -> AnnotateTIROpPattern -> FuseOps -> FuseTIR,
# on Chapter 4's own diamond graph -- t1=a+b feeds t2=t1*c and t3=relu(t1);
# out=t2+t3), re-run fresh this session so its own real PrimFunc count sits
# directly next to File 106's own fresh CUDA Hammer run of the SAME graph
# shape through boundedReductionFusionPass() (Chapter 16.3). Part 1 builds
# the diamond graph directly in Relax and prints its unoptimized IR. Part 2
# lowers it through TVM's own real fusion pipeline (the real reason it has
# to be exactly these four passes, not just FuseOps() alone, is explained
# in Chapter 26's own prose: FuseOps() groups PrimFuncs by an op_pattern
# ATTRIBUTE nothing assigns without AnnotateTIROpPattern() running first).
# Part 3 asks the same escape-boundary question Chapter 25's own File 063
# asked of XLA: does making t1 ALSO a live output change what gets fused?
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
