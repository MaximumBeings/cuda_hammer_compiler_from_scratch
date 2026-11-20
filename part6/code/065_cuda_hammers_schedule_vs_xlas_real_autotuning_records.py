#!/usr/bin/env python3
# Chapter 25, Section 25.3 (capstone): CUDA Hammer's own Schedule/TuningCache
# machinery (Ch21, Ch24) laid directly next to the real record types XLA's
# own GPU autotuner uses in production.
#
# This sandbox has no physical GPU (same limitation named in File 064 and in
# every CUDA chapter since Ch18), so there is no real XLA GPU-autotuning run
# to execute here the way File 063/064 executed real XLA fusion. What CAN be
# done honestly is a direct structural comparison: this file is real,
# runnable Python -- not prose -- built from field names taken verbatim from
# XLA's own real, public source (openxla/xla, xla/autotuning.proto, and the
# OpenXLA determinism docs), laid next to this book's own real C++ types
# from Ch21 and Ch24, so the comparison is inspectable rather than asserted.
#
# Three real XLA facts anchor this comparison (all cited, none invented):
#   1. XLA's GPU autotuner is measurement-based: "During compilation, XLA's
#      autotuner profiles multiple candidate kernel implementations ... live
#      on the host's GPU to find the fastest algorithm" (OpenXLA
#      determinism docs) -- the same "rank cheaply, then measure for real"
#      shape as Ch23.3's own hybrid autotuner, at production scale.
#   2. A TritonGemmKey (xla/autotuning.proto) records real tuning knobs:
#      block_m, block_n, block_k, num_stages, num_warps, num_ctas -- a GPU
#      GEMM's own tile-shape-and-launch-configuration analogue of Ch21's own
#      Schedule{tileSizePerLoop, loopOrder, unrollFactor}.
#   3. AutotuningLog records context, not the cache key or a reuse guarantee.
#      Current AutotuneCacheKey uses a device-model description, canonical HLO
#      and a cache version. Persisted cache compatibility also has user-managed
#      version/invalidation requirements (OpenXLA persisted_autotuning docs).

from dataclasses import dataclass, fields


# --- CUDA Hammer's own Ch21 Schedule, transcribed field-for-field from the
# real C++ struct (docs/part5/code/051-053) -----------------------------
@dataclass
class CudaHammerSchedule:
    tile_size_per_loop: list   # one tile size per loop dimension
    loop_order: list           # a permutation of loop indices
    unroll_factor: int         # innermost-loop unroll factor


# --- XLA's real TritonGemmKey, transcribed field-for-field from
# xla/autotuning.proto (openxla/xla, fetched 2026) ------------------------
@dataclass
class XlaTritonGemmKey:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    num_ctas: int
    is_tma_allowed: bool


# --- CUDA Hammer's own Ch24.3 cache key -----------------------------------
def loop_nest_key_for_target(shape_str: str, target: str) -> str:
    return f"{shape_str}@{target}"


# --- XLA's real AutotuningLog target-context fields, transcribed from
# xla/autotuning.proto (the fields that accompany a logged result) -----
@dataclass
class XlaAutotuningLogContext:
    cudnn_version: str
    compute_capability: str
    device_pci_bus_id: str


def print_fields(label, instance):
    print(f"{label}:")
    for f in fields(instance):
        print(f"    {f.name} = {getattr(instance, f.name)!r}")


def main():
    print("=" * 78)
    print("PART 1: the tuning-knob record itself")
    print("=" * 78)
    ch_sched = CudaHammerSchedule(
        tile_size_per_loop=[6, 8], loop_order=[0, 1], unroll_factor=4
    )  # this book's own real Chapter 23.3 cloud-sandbox winner
    print_fields("CudaHammerSchedule (Ch21, this book's own struct)", ch_sched)
    print(f"    -- {len(fields(ch_sched))} fields")
    print()

    xla_key = XlaTritonGemmKey(
        block_m=128, block_n=128, block_k=32,
        num_stages=3, num_warps=4, num_ctas=1, is_tma_allowed=False,
    )
    print("NOTE: the field NAMES below are real (xla/autotuning.proto); the "
          "VALUES are representative placeholders, not a measured result -- "
          "this sandbox has no GPU for XLA's real autotuner to search on.")
    print_fields("XlaTritonGemmKey (real field names, xla/autotuning.proto)",
                 xla_key)
    print(f"    -- {len(fields(xla_key))} fields")
    print()
    print("-- both records answer the exact same question for their own "
          "system: how should ONE loop nest / GEMM be tiled and launched. "
          "CUDA Hammer's Schedule generalizes over ANY LoopNest shape (Ch21's "
          "own enumerateSchedules()); XLA's TritonGemmKey is specific to one "
          "matmul-shaped Triton kernel template -- CUDA Hammer's toy IR never "
          "grew a matmul op (Ch4's own scope, unchanged through Ch24).")

    print()
    print("=" * 78)
    print("PART 2: target context is not a complete cache key")
    print("=" * 78)
    ch_key = loop_nest_key_for_target("dim0:6,dim1:8", "aarch64")
    print(f"CUDA Hammer's own Ch24.3 key: {ch_key!r}")

    xla_ctx = XlaAutotuningLogContext(
        cudnn_version="9.x",
        compute_capability="sm_90",
        device_pci_bus_id="0000:00:00.0",
    )
    print("NOTE: again, real field names, placeholder values -- no GPU here "
          "to read real ones from.")
    print_fields("XLA's own real AutotuningLog context fields "
                 "(xla/autotuning.proto)", xla_ctx)
    print()
    print("-- AutotuningLog retains diagnostic context; these placeholder fields "
          "do not specify cache equality or guarantee exact-device reuse. "
          "Current XLA AutotuneCacheKey combines a device-model description, "
          "canonical HLO and a cache version. Persisted cache version separation "
          "and invalidation also require user care. CUDA Hammer's simple target "
          "string is an educational comparison, not XLA's full reuse policy.")


if __name__ == "__main__":
    main()
