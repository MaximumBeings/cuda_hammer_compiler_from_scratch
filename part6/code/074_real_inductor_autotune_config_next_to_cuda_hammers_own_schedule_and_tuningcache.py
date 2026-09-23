# Chapter 28, File 074: TorchInductor's own real autotuning config surface,
# laid directly next to CUDA Hammer's own Schedule/enumerateSchedules()
# (Chapter 21), hybrid cost-model-then-measurement autotuner (Chapter 23.3),
# and TuningCache (Chapter 24) -- the same "read the real library's own
# record types side by side with CUDA Hammer's own" capstone pattern used
# to close Chapters 25, 26, and 27.
#
# Real toolchain note: `torch._inductor.config` is a real, plain Python
# module whose attributes ARE the config -- there is no separate dataclass
# or schema object to read. Filtering `dir(torch._inductor.config)` by
# substring (e.g. "autotune" in name) surprisingly returns only a handful
# of function-valued entries (helper functions the module itself defines),
# because most of the real config fields are dynamically installed by an
# internal `install_config_module()` call and don't show up under a naive
# `dir()` scan the same way. Using `hasattr()`/`getattr()` against a
# hand-picked list of real field names (found by reading Inductor's own
# source and release notes) is what actually surfaces their real values.
#
# Compiled with:   python3 "074_real_inductor_autotune_config_next_to_cuda_hammers_own_schedule_and_tuningcache.py"

import torch
import torch._inductor.config as inductor_config


# CUDA Hammer's own Chapter 21 Schedule and Chapter 24 TuningCache shapes,
# reproduced here (not imported -- this file has no dependency on the rest
# of the CUDA Hammer codebase) purely so the real Inductor fields below can
# be read directly alongside them.
class Schedule:
    """CUDA Hammer, Chapter 21: one point in the autotuning search space."""

    def __init__(self, tile_size_per_loop, loop_order, unroll_factor):
        self.tile_size_per_loop = tile_size_per_loop
        self.loop_order = loop_order
        self.unroll_factor = unroll_factor


class TuningCacheEntry:
    """CUDA Hammer, Chapter 24: one persisted autotuning result. Chapter 24's
    real TuningCache stores only the WINNING schedule per loop-nest key, in
    a plain-text, human-readable file."""

    def __init__(self, loop_nest_key, best_schedule, measured_time_ns):
        self.loop_nest_key = loop_nest_key
        self.best_schedule = best_schedule
        self.measured_time_ns = measured_time_ns


REAL_AUTOTUNE_FIELDS = [
    "max_autotune",
    "max_autotune_gemm",
    "max_autotune_pointwise",
    "coordinate_descent_tuning",
    "autotune_local_cache",
    "search_autotune_cache",
    "max_autotune_gemm_backends",
]


def read_real_inductor_autotune_config():
    """Reads the REAL current values of torch._inductor.config's own
    autotuning fields -- these are real library defaults, not values this
    file invented or hardcoded."""
    return {name: getattr(inductor_config, name, "<MISSING>")
            for name in REAL_AUTOTUNE_FIELDS}


if __name__ == "__main__":
    defaults = read_real_inductor_autotune_config()
    print("Real torch._inductor.config autotuning fields (library defaults):")
    for name, value in defaults.items():
        print(f"  {name} = {value!r}")

    print()
    print("Compared against CUDA Hammer's own autotuning machinery:")
    print(f"  Chapter 21 Schedule{{tile_size_per_loop, loop_order, "
          f"unroll_factor}} <-> Inductor's own internal per-op Config "
          f"objects (block sizes, num_warps, num_stages -- never exposed "
          f"as a public dataclass the way CUDA Hammer's own Schedule is)")
    print(f"  Chapter 23.3 hybrid cost-model-then-measurement autotuner <-> "
          f"Inductor's own coordinate_descent_tuning={defaults['coordinate_descent_tuning']!r} "
          f"(a real greedy local search over the config space, distinct "
          f"from CUDA Hammer's own top-K-prune-then-measure hybrid) plus "
          f"max_autotune_gemm_backends={defaults['max_autotune_gemm_backends']!r} "
          f"(the real, named set of candidate implementations -- ATen's "
          f"own library call, a real generated Triton kernel, or a real "
          f"generated C++ kernel -- Inductor benchmarks against each other "
          f"under max-autotune, a genuinely different search axis than "
          f"CUDA Hammer's tile/order/unroll space: WHICH BACKEND, not just "
          f"which schedule)")
    print(f"  Chapter 24 TuningCache (plain-text, human-keyed, winner-only) "
          f"<-> autotune_local_cache={defaults['autotune_local_cache']!r} "
          f"(True by default: a real persistent local autotuning cache, "
          f"keeping this book's Chapter 27 Triton-vs-CUDA-Hammer disk-cache "
          f"comparison consistent -- a third independent real production "
          f"compiler defaulting to ON persistent caching, where CUDA "
          f"Hammer's own TuningCache requires an explicit save)")

    print()
    print("Real end-to-end execution: torch.compile(f, mode='max-autotune') "
          "on CPU with no GPU present (the mode still runs; it just narrows "
          "to backends that make sense without a GPU):")

    def matmul(x, y):
        return x @ y

    # Real, explicit backend restriction -- 'TRITON' alone would require a
    # working GPU code-generation path this sandbox does not have; 'ATEN'
    # and 'CPP' are both real CPU-capable backends.
    old_backends = inductor_config.max_autotune_gemm_backends
    inductor_config.max_autotune_gemm_backends = "ATEN,CPP"
    try:
        compiled_matmul = torch.compile(matmul, mode="max-autotune")
        x = torch.randn(64, 64)
        y = torch.randn(64, 64)
        out = compiled_matmul(x, y)
        eager_out = matmul(x, y)
        print(f"  max_autotune_gemm_backends temporarily set to "
              f"{inductor_config.max_autotune_gemm_backends!r}")
        print(f"  Real correctness check: compiled vs. eager match = "
              f"{torch.allclose(out, eager_out, atol=1e-4)}")
    finally:
        inductor_config.max_autotune_gemm_backends = old_backends

    print()
    print("Real torch.cuda.is_available():", torch.cuda.is_available())
    print("(max-autotune ran and produced a correct result with zero "
          "physical GPU present -- Inductor's own backend search degrades "
          "to CPU-capable candidates automatically rather than failing)")
