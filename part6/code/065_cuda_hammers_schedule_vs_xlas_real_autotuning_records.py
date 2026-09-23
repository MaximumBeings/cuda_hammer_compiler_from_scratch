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
#   3. XLA persists autotuning results keyed by real target context --an
#      AutotuningLog carries cudnn_version, compute_capability, and
#      device_pci_bus_id alongside the winning config (xla/autotuning.proto)
#      -- the same (workload, target) shape Ch24.3's own
#      loopNestKeyForTarget() arrived at by direct measurement on two real
#      machines, not by reading XLA's source first.

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
# xla/autotuning.proto (the fields that accompany a persisted result) -----
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
    print("PART 2: the (workload, target) cache key shape")
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
    print("-- Ch24.3's own loopNestKeyForTarget() was built by direct "
          "measurement on two real machines (a cloud sandbox and a real "
          "device disagreeing on the fastest schedule for the identical "
          "LoopNest shape) BEFORE this chapter ever looked at XLA's own "
          "source. XLA's own AutotuningLog arrives at the same shape -- a "
          "winning config is only valid for the (workload, target) pair "
          "that produced it -- at production scale, with three real target "
          "fields where Ch24.3's own key used one string.")


if __name__ == "__main__":
    main()
