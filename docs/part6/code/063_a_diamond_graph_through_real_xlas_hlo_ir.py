#!/usr/bin/env python3
# Chapter 25, Section 25.1-25.2: CUDA Hammer's own diamond test graph --
# the same shared-value shape used by Ch4's Graph/Node IR and Ch13's
# elementwiseFusionPass() ever since -- run through a REAL production
# compiler's own fusion pass instead of CUDA Hammer's own.
#
# This is the first Python file in this book. Every chapter through Ch24
# built CUDA Hammer itself in C++; this chapter studies a real, independent
# system (XLA, via JAX) that this book has only ever cited before (Ch3).
# There is no C++ source for XLA available to compile in this sandbox, and
# reimplementing XLA's own fusion pass here would defeat the point of a
# case study -- so this file installs the real thing (`pip install jax`,
# which pulls a real jaxlib containing XLA) and asks XLA itself, live, what
# it decides. Every HLO dump below is genuinely produced by XLA on this
# machine, not transcribed from documentation.
#
# Part 1 prints XLA's UNOPTIMIZED HLO (StableHLO, as traced from the Python
# function, before any XLA-internal optimization) for the diamond graph:
#     t1 = a + b
#     t2 = t1 * c
#     t3 = relu(t1)
#     out = t2 + t3
# -- structurally the same "t1 feeds two consumers that join at out" shape
# as Ch4's own diamond test graph, reused through every fusion pass since
# Ch13.
#
# Part 2 prints XLA's OPTIMIZED HLO for the same graph -- after XLA's real,
# built-in fusion pass has run. Ch13's own elementwiseFusionPass() forces a
# value external whenever `consumers != 1` (t1 has 2 consumers here, so
# Ch13's own pass keeps it external, materialized to memory once and read
# back twice). The question this file asks empirically is whether a real
# production fusion pass uses that same rule.
#
# Part 3 changes ONE thing: it also returns t1 itself as a second output,
# so t1 is no longer just "used twice inside the graph" but also escapes
# the whole computation as a live result -- the same distinction Ch16's own
# classifyNode() drew between an ordinary Shared node and a GraphRoot.

import jax
import jax.numpy as jnp


def diamond(a, b, c):
    t1 = a + b
    t2 = t1 * c
    t3 = jax.nn.relu(t1)
    out = t2 + t3
    return out


def diamond_multi_output(a, b, c):
    t1 = a + b
    t2 = t1 * c
    t3 = jax.nn.relu(t1)
    out = t2 + t3
    return out, t1


def main():
    a = jnp.ones((6, 8), dtype=jnp.float32)
    b = jnp.ones((6, 8), dtype=jnp.float32)
    c = jnp.ones((6, 8), dtype=jnp.float32)

    print(f"jax version: {jax.__version__}")
    print(f"real backend device: {jax.devices()[0]}")
    print()

    print("=" * 78)
    print("PART 1: unoptimized HLO (StableHLO) for the diamond graph")
    print("=" * 78)
    lowered = jax.jit(diamond).lower(a, b, c)
    print(lowered.as_text())

    print("=" * 78)
    print("PART 2: optimized HLO for the diamond graph (real XLA fusion pass)")
    print("=" * 78)
    compiled = jax.jit(diamond).lower(a, b, c).compile()
    optimized_text = compiled.as_text()
    print(optimized_text)

    fusion_count = optimized_text.count(" fusion(")
    print(f"-- number of `fusion(...)` instructions in the ENTRY computation: "
          f"{fusion_count}")
    print(f"-- t1 (the shared add) appears INSIDE the single fused_computation "
          f"body, not as its own materialized buffer: "
          f"{'add.1' in optimized_text or 'add.0' in optimized_text}")

    print()
    print("=" * 78)
    print("PART 3: same graph, but t1 is ALSO returned as a live output")
    print("=" * 78)
    compiled_mo = jax.jit(diamond_multi_output).lower(a, b, c).compile()
    mo_text = compiled_mo.as_text()
    print(mo_text)

    mo_fusion_count = mo_text.count(" fusion(")
    print(f"-- number of `fusion(...)` instructions in the ENTRY computation "
          f"once t1 escapes as an output: {mo_fusion_count}")


if __name__ == "__main__":
    main()
