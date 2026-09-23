# Chapter 27, File 071 (capstone): real Triton autotuning, laid directly next
# to CUDA Hammer's own real Schedule (Ch21), hybrid autotuner (Ch23), and
# TuningCache (Ch24) -- the same "no GPU here for a real autotuning RUN, so
# lay the real record types side by side" pattern Chapter 25.3 used for XLA's
# own TritonGemmKey/AutotuningLog and Chapter 26.3 used for MetaSchedule's own
# TuningRecord/Database. Every field and method name below is read directly
# from this sandbox's own real, installed `triton` package via
# `inspect.getsource()` -- cited from the real source, not fabricated -- but,
# like Chapter 25.3 and 26.3, no real search or real measurement runs here:
# Triton's own `Autotuner._bench()` ultimately needs
# `driver.active.get_benchmarker()`, which needs the same live GPU driver
# File 069 already found does not exist in this sandbox.
#
# Compiled with:   python3 "071_real_triton_autotune_config_search_next_to_cuda_hammers_own_schedule_and_tuningcache.py"

import inspect
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget

TARGET = GPUTarget(backend="cuda", arch=80, warp_size=32)


# Real triton.Config objects -- direct analogue of Chapter 21's own
# Schedule{tileSizePerLoop, loopOrder, unrollFactor}: a named bundle of
# tunable meta-parameters, not a value the kernel body computes with.
REAL_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=REAL_CONFIGS, key=["n_elements"])
@triton.jit
def tuned_vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


if __name__ == "__main__":
    autotuner = tuned_vector_add_kernel
    print(f"Real Autotuner object: {type(autotuner).__name__}")
    print(f"  self.configs: {len(autotuner.configs)} real triton.Config objects")
    for cfg in autotuner.configs:
        print(f"    {cfg}")
    print(f"  self.keys (cache-key argument names): {autotuner.keys}")
    print(f"  self.cache (in-process memo, empty until first real call): "
          f"{autotuner.cache!r}")
    print(f"  self.cache_results (persistent on-disk cache enabled): "
          f"{autotuner.cache_results}")
    print(f"  self.perf_model (cost-model-based pruning before measurement): "
          f"{autotuner.perf_model}")
    print(f"  self.configs_top_k (fraction/count kept after perf_model ranks): "
          f"{autotuner.configs_top_k}")

    print()
    print("Real Autotuner.run()'s own cache-key construction (Ch24's own "
          "loopNestKeyForTarget(), read directly from installed source):")
    run_src_lines = inspect.getsource(type(autotuner).run).splitlines()
    for line in run_src_lines[1:9]:
        print(f"  {line}")

    print()
    print("Real Autotuner.prune_configs()'s own cost-model step (Ch22/23's "
          "own estimateScheduleCost()-then-measure shape, read directly "
          "from installed source):")
    prune_src_lines = inspect.getsource(type(autotuner).prune_configs).splitlines()
    for line in prune_src_lines[8:19]:
        print(f"  {line}")

    print()
    print("Real Autotuner.check_disk_cache()'s own persistent-cache write "
          "(Ch24's own TuningCache, read directly from installed source):")
    disk_src_lines = inspect.getsource(type(autotuner).check_disk_cache).splitlines()
    for line in disk_src_lines[-9:]:
        print(f"  {line}")

    # No GPU to run the real search, but every candidate Config's own kwargs
    # still compile through the exact same real AOT path File 069/070 used --
    # confirming the search space is a search space of real, independently
    # compilable kernels, the same "every candidate is a real compiled
    # program" discipline Chapter 23.1 established for CUDA Hammer's own
    # `generateScheduledElementwiseFunction()`.
    print()
    print("Real AOT compilation of every candidate Config (no GPU needed, "
          "same technique as Files 069/070):")
    base_fn = tuned_vector_add_kernel.fn  # the underlying @triton.jit function
    for cfg in REAL_CONFIGS:
        block_size = cfg.kwargs["BLOCK_SIZE"]
        src = triton.compiler.ASTSource(
            fn=base_fn,
            signature={"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
                       "n_elements": "i32", "BLOCK_SIZE": "constexpr"},
            constexprs={"BLOCK_SIZE": block_size},
        )
        compiled = triton.compile(src, target=TARGET)
        print(f"  BLOCK_SIZE={block_size:5d} num_warps={cfg.num_warps} "
              f"num_stages={cfg.num_stages}: compiled OK, "
              f"ptx={len(compiled.asm['ptx'])} chars, "
              f"cubin={len(compiled.asm['cubin'])} bytes")
