#!/usr/bin/env python3
# Chapter 25, Section 25.2 (continued): the reduction half of the same
# question File 063 asked about elementwise sharing -- this time against
# Ch14's own reductionFusionPass() rule, which forces a Sum node's OUTPUT
# external unconditionally ("a Sum node's output is ALWAYS external, any
# consumer count" -- Ch14), not just when it is shared.
#
# Part 1 lowers an elementwise-into-reduction chain, structurally the same
# shape Ch14's own file 031/032 built (Add -> ReLU -> Sum), and prints
# XLA's real optimized HLO.
#
# Part 2 extends the chain with a REAL consumer of the reduction's scalar
# result (`out = sum(...) * 2.0`) to ask the sharper question: does that
# downstream use force the reduction's own result to materialize to memory
# the way Ch14's rule always does, or does XLA fuse straight through a
# reduction the same way File 063 found it fusing straight through a
# shared elementwise value?
#
# A caveat stated plainly, not discovered by accident: this sandbox has no
# physical GPU (the same honest limitation Ch18 onward has lived with since
# Chapter 18's own CUDA codegen), so every HLO dump in this book comes from
# XLA's CPU backend. XLA's real GPU backend compiles "always... to exactly
# one GPU kernel" per fusion (OpenXLA's own GPU architecture docs) and a
# full reduction on a GPU genuinely needs cross-thread-block synchronization
# that a CPU loop does not -- so XLA:GPU's own fusion boundary around a
# reduction may be stricter than what this file observes on XLA:CPU. That
# gap is named here explicitly rather than left implicit.

import jax
import jax.numpy as jnp


def reduce_chain(a, b):
    t1 = a + b
    t2 = jax.nn.relu(t1)
    total = jnp.sum(t2)
    return total


def reduce_then_use(a, b):
    t1 = a + b
    t2 = jax.nn.relu(t1)
    s = jnp.sum(t2)
    out = s * 2.0
    return out


def main():
    a = jnp.ones((6, 8), dtype=jnp.float32)
    b = jnp.ones((6, 8), dtype=jnp.float32)

    print(f"jax version: {jax.__version__}")
    print(f"real backend device: {jax.devices()[0]}")
    print()

    print("=" * 78)
    print("PART 1: optimized HLO for Add -> ReLU -> Sum (Ch14's own shape)")
    print("=" * 78)
    compiled1 = jax.jit(reduce_chain).lower(a, b).compile()
    text1 = compiled1.as_text()
    print(text1)
    print(f"-- fusion(...) instructions in ENTRY: {text1.count(' fusion(')}")
    print(f"-- the reduce(...) op is inside the SAME fused_computation body "
          f"as the add/relu (not its own materialized buffer): "
          f"{'reduce(' in text1}")

    print()
    print("=" * 78)
    print("PART 2: Sum's result now feeds a real downstream consumer (*2.0)")
    print("=" * 78)
    compiled2 = jax.jit(reduce_then_use).lower(a, b).compile()
    text2 = compiled2.as_text()
    print(text2)
    entry_fusion_count = text2.count(" fusion(")
    print(f"-- fusion(...) instructions in ENTRY: {entry_fusion_count}")
    print(f"-- the downstream multiply is fused into the SAME single kernel "
          f"as the reduce, not split into a second one: "
          f"{entry_fusion_count == 1}")


if __name__ == "__main__":
    main()
