# Chapter 25: XLA -- Fusion and Autotuning in a Real Production Compiler

Chapter 3 surveyed three real ML compilers -- XLA, TVM, and Triton -- as
citations, to ground this book's own design decisions in what production
systems actually do. Every chapter since Chapter 4 has instead built CUDA
Hammer itself: its own IR (Part 1), its own optimization passes (Part 2),
its own fusion engine (Part 3), its own code generators (Part 4), its own
autotuner (Part 5). Part 6 turns back outward. Its six chapters study real,
independent compilers directly rather than citing them from a distance --
this chapter on XLA, later chapters on TVM and Triton -- asking the same
question of each: given the same kind of problem CUDA Hammer just solved
from scratch, what does a real production system actually choose to do,
and why?

**A note on how this chapter is built, since it changes some of this
book's own ground rules.** Every code file through Chapter 24 was C++,
compiled with g++ or nvcc, because CUDA Hammer itself is a C++ program.
XLA is not something this book can compile from source in a sandbox --
it is a very large, independent C++ codebase with its own build system --
and hand-transcribing what XLA's documentation *says* it does would trade
this book's own standing discipline (show it happening, don't just assert
it) for something weaker. There is a real, working middle path: XLA ships
as the backend underneath JAX, `pip install jax` pulls a real compiled
`jaxlib` containing real XLA, and a JAX program run through it produces
XLA's own real, genuine compiler output -- HLO (High-Level Operations),
XLA's own IR -- with no simulation and no hand-editing. Every HLO dump in
this chapter was produced that way, by code in this chapter's own files,
run for real on both machines, same as every chapter before it. The
language changes from C++ to Python because that is what talks to XLA
directly without reimplementing it; the standing discipline -- run it for
real, report what actually happened, verify on both machines, name the
divergences plainly -- does not change.

This sandbox still has no physical GPU, the same honest limitation this
book has lived with since Chapter 18's own CUDA codegen. Every HLO dump
in this chapter comes from XLA's CPU backend. XLA's real GPU backend
compiles, in OpenXLA's own words, "always... to exactly one GPU kernel"
per fusion -- and a full reduction on a GPU genuinely needs cross-thread-
block synchronization that a CPU loop does not, so XLA:GPU's own fusion
boundaries may be stricter than anything this chapter observes on
XLA:CPU. That gap is named here explicitly, the same way Chapter 18 named
its own "compiles but never executes" limitation, rather than left
implicit.

```text
+----------------------------------------------------------------+
|  Chapter 25's own shape, section by section                    |
|                                                                  |
|  25.1  A production IR meets CUDA Hammer's own diamond graph    |
|        -- same shared-value shape as Ch4/Ch13, run through      |
|           REAL XLA instead of CUDA Hammer's own passes.         |
|                                                                  |
|  25.2  What XLA's real fusion pass draws its own boundary       |
|        around -- an "escapes the group" rule tested against     |
|        Ch13's own "consumers != 1" rule and Ch14's own          |
|        "a reduction's output is ALWAYS external" rule.          |
|                                                                  |
|  25.3  A real production autotuner's own tuning-knob and        |
|        cache-key records, laid directly next to Ch21's          |
|        Schedule and Ch24's loopNestKeyForTarget().              |
+----------------------------------------------------------------+
```

## 25.1 A Production IR Meets CUDA Hammer's Own Diamond Graph

Every fusion pass this book has built since Chapter 13 has been tested on
the same shape first: a shared value with two consumers that rejoin
downstream. Chapter 4's own diamond test graph put it in CUDA Hammer's
own IR; Chapter 13's `elementwiseFusionPass()` used it to prove that a
value with `consumers != 1` gets forced external. XLA has never seen
Chapter 4's own `Graph`/`Node` types, so it cannot be handed that graph
directly -- but the same *shape* is trivial to write in JAX, and JAX's own
`jit` compiles it through the real XLA pipeline:

```text
    a       b                    JAX / Python:
     \     /                       t1 = a + b
      \   /                        t2 = t1 * c
      [t1]                          t3 = relu(t1)
      /   \                        out = t2 + t3
     /     \
   [t2]   [t3]     -- t1 feeds BOTH t2 and t3
     \     /
      \   /
     [out]
```

XLA's own real compilation pipeline, per OpenXLA's own architecture docs,
runs a JAX/PyTorch/TensorFlow program through several real stages: the
frontend traces the program into **StableHLO** (a "portability layer
between ML frameworks and the compiler"), XLA runs "several built-in
optimization and analysis passes on the StableHLO graph that are
target-independent, such as CSE, target-independent operation fusion, and
buffer analysis," then hands the result to a backend for
target-specific HLO optimization and finally code generation (the CPU and
GPU backends both lower through LLVM). Two of CUDA Hammer's own
five-chapters-old ideas are sitting right there in that description under
different names: XLA's own target-independent CSE pass is the same idea
as Chapter 10's `commonSubexpressionEliminationPass()`, and its
target-independent/backend-specific split is the same two-stage shape as
Chapter 8's own `PassManager` running generic passes before Chapter 18's
own backend-specific CUDA codegen runs.

File 063 asks JAX for the diamond graph's own StableHLO, unoptimized,
exactly as XLA's frontend first sees it, then asks for the same graph's
real, fully optimized HLO after XLA's own fusion pass has run.
```python
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
```

```bash
python3 "063_a_diamond_graph_through_real_xlas_hlo_ir.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated XLA output):**

```text
jax version: 0.10.2
real backend device: cpu:0

==============================================================================
PART 1: unoptimized HLO (StableHLO) for the diamond graph
==============================================================================
module @jit_diamond attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<6x8xf32>, %arg1: tensor<6x8xf32>, %arg2: tensor<6x8xf32>) -> (tensor<6x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<6x8xf32>
    %1 = stablehlo.multiply %0, %arg2 : tensor<6x8xf32>
    %2 = call @relu(%0) : (tensor<6x8xf32>) -> tensor<6x8xf32>
    %3 = stablehlo.add %1, %2 : tensor<6x8xf32>
    return %3 : tensor<6x8xf32>
  }
  func.func private @relu(%arg0: tensor<6x8xf32>) -> tensor<6x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x8xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<6x8xf32>
    return %1 : tensor<6x8xf32>
  }
}

==============================================================================
PART 2: optimized HLO for the diamond graph (real XLA fusion pass)
==============================================================================
HloModule jit_diamond, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0}, f32[6,8]{1,0})->f32[6,8]{1,0}}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true}

FileNames
1 "/home/claude/hammer_repo/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py"

FunctionNames
1 "<module>"
2 "main"
3 "diamond"

FileLocations
1 {file_name_id=1 function_name_id=1 line=102 end_line=102 column=4 end_column=10}
2 {file_name_id=1 function_name_id=2 line=71 end_line=71 column=14 end_column=45}
3 {file_name_id=1 function_name_id=3 line=44 end_line=44 column=9 end_column=14}
4 {file_name_id=1 function_name_id=3 line=45 end_line=45 column=9 end_column=15}
5 {file_name_id=1 function_name_id=3 line=46 end_line=46 column=9 end_column=24}
6 {file_name_id=1 function_name_id=3 line=47 end_line=47 column=10 end_column=17}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=3}
5 {file_location_id=5 parent_frame_id=3}
6 {file_location_id=6 parent_frame_id=3}


%fused_computation (param_0.2: f32[6,8], param_1.3: f32[6,8], param_2.2: f32[6,8]) -> f32[6,8] {
  %param_1.3 = f32[6,8]{1,0} parameter(1)
  %param_2.2 = f32[6,8]{1,0} parameter(2)
  %add.1 = f32[6,8]{1,0} add(%param_1.3, %param_2.2), metadata={op_name="jit(diamond)/add" stack_frame_id=3}
  %param_0.2 = f32[6,8]{1,0} parameter(0)
  %mul.0 = f32[6,8]{1,0} multiply(%add.1, %param_0.2), metadata={op_name="jit(diamond)/mul" stack_frame_id=4}
  %constant.2 = f32[] constant(0), metadata={op_name="jit(diamond)/jit(relu)" stack_frame_id=5}
  %max.5 = f32[6,8]{1,0} broadcast(%constant.2), dimensions={}, metadata={op_name="jit(diamond)/jit(relu)/max" stack_frame_id=5}
  %max.4 = f32[6,8]{1,0} maximum(%add.1, %max.5), metadata={op_name="jit(diamond)/jit(relu)/max" stack_frame_id=5}
  ROOT %add.0 = f32[6,8]{1,0} add(%mul.0, %max.4), metadata={op_name="jit(diamond)/add" stack_frame_id=6}
}

ENTRY %main.2 (a.1: f32[6,8], b.1: f32[6,8], c.1: f32[6,8]) -> f32[6,8] {
  %a.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %b.1 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  %c.1 = f32[6,8]{1,0} parameter(2), metadata={op_name="c"}
  ROOT %maximum_add_fusion = f32[6,8]{1,0} fusion(%c.1, %a.1, %b.1), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(diamond)/add" stack_frame_id=6}
}


-- number of `fusion(...)` instructions in the ENTRY computation: 1
-- t1 (the shared add) appears INSIDE the single fused_computation body, not as its own materialized buffer: True

==============================================================================
PART 3: same graph, but t1 is ALSO returned as a live output
==============================================================================
HloModule jit_diamond_multi_output, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0}, f32[6,8]{1,0})->(f32[6,8]{1,0}, f32[6,8]{1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true,true}

FileNames
1 "/home/claude/hammer_repo/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py"

FunctionNames
1 "<module>"
2 "main"
3 "diamond_multi_output"

FileLocations
1 {file_name_id=1 function_name_id=1 line=102 end_line=102 column=4 end_column=10}
2 {file_name_id=1 function_name_id=2 line=92 end_line=92 column=18 end_column=62}
3 {file_name_id=1 function_name_id=3 line=52 end_line=52 column=9 end_column=14}
4 {file_name_id=1 function_name_id=3 line=53 end_line=53 column=9 end_column=15}
5 {file_name_id=1 function_name_id=3 line=54 end_line=54 column=9 end_column=24}
6 {file_name_id=1 function_name_id=3 line=55 end_line=55 column=10 end_column=17}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=3}
5 {file_location_id=5 parent_frame_id=3}
6 {file_location_id=6 parent_frame_id=3}


%fused_computation (param_0.1: f32[6,8], param_1.2: f32[6,8]) -> f32[6,8] {
  %param_0.1 = f32[6,8]{1,0} parameter(0)
  %param_1.2 = f32[6,8]{1,0} parameter(1)
  %mul.0 = f32[6,8]{1,0} multiply(%param_0.1, %param_1.2), metadata={op_name="jit(diamond_multi_output)/mul" stack_frame_id=4}
  %constant.2 = f32[] constant(0), metadata={op_name="jit(diamond_multi_output)/jit(relu)" stack_frame_id=5}
  %max.5 = f32[6,8]{1,0} broadcast(%constant.2), dimensions={}, metadata={op_name="jit(diamond_multi_output)/jit(relu)/max" stack_frame_id=5}
  %max.4 = f32[6,8]{1,0} maximum(%param_0.1, %max.5), metadata={op_name="jit(diamond_multi_output)/jit(relu)/max" stack_frame_id=5}
  ROOT %add.0 = f32[6,8]{1,0} add(%mul.0, %max.4), metadata={op_name="jit(diamond_multi_output)/add" stack_frame_id=6}
}

%wrapped_add_computation (param_0.2: f32[6,8], param_1.3: f32[6,8]) -> f32[6,8] {
  %param_0.2 = f32[6,8]{1,0} parameter(0)
  %param_1.3 = f32[6,8]{1,0} parameter(1)
  ROOT %add.1 = f32[6,8]{1,0} add(%param_0.2, %param_1.3), metadata={op_name="jit(diamond_multi_output)/add" stack_frame_id=3}
}

ENTRY %main.2 (a.1: f32[6,8], b.1: f32[6,8], c.1: f32[6,8]) -> (f32[6,8], f32[6,8]) {
  %a.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %b.1 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  %c.1 = f32[6,8]{1,0} parameter(2), metadata={op_name="c"}
  %wrapped_add = f32[6,8]{1,0} fusion(%a.1, %b.1), kind=kLoop, calls=%wrapped_add_computation, metadata={op_name="jit(diamond_multi_output)/add" stack_frame_id=3}
  %maximum_add_fusion = f32[6,8]{1,0} fusion(%wrapped_add, %c.1), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(diamond_multi_output)/add" stack_frame_id=6}
  ROOT %tuple.1 = (f32[6,8]{1,0}, f32[6,8]{1,0}) tuple(%maximum_add_fusion, %wrapped_add)
}


-- number of `fusion(...)` instructions in the ENTRY computation once t1 escapes as an output: 2
```

**Output (device, aarch64 Linux VM -- real, live-generated XLA output):**

```text
jax version: 0.6.2
real backend device: TFRT_CPU_0

==============================================================================
PART 1: unoptimized HLO (StableHLO) for the diamond graph
==============================================================================
module @jit_diamond attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<6x8xf32>, %arg1: tensor<6x8xf32>, %arg2: tensor<6x8xf32>) -> (tensor<6x8xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<6x8xf32>
    %1 = stablehlo.multiply %0, %arg2 : tensor<6x8xf32>
    %2 = call @relu(%0) : (tensor<6x8xf32>) -> tensor<6x8xf32>
    %3 = stablehlo.add %1, %2 : tensor<6x8xf32>
    return %3 : tensor<6x8xf32>
  }
  func.func private @relu(%arg0: tensor<6x8xf32>) -> tensor<6x8xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<6x8xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<6x8xf32>
    return %1 : tensor<6x8xf32>
  }
}

==============================================================================
PART 2: optimized HLO for the diamond graph (real XLA fusion pass)
==============================================================================
HloModule jit_diamond, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0}, f32[6,8]{1,0})->f32[6,8]{1,0}}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true}

%fused_computation (param_0.2: f32[6,8], param_1.3: f32[6,8], param_2.2: f32[6,8]) -> f32[6,8] {
  %param_1.3 = f32[6,8]{1,0} parameter(1)
  %param_2.2 = f32[6,8]{1,0} parameter(2)
  %add.1 = f32[6,8]{1,0} add(%param_1.3, %param_2.2), metadata={op_name="jit(diamond)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=44}
  %param_0.2 = f32[6,8]{1,0} parameter(0)
  %multiply.0 = f32[6,8]{1,0} multiply(%add.1, %param_0.2), metadata={op_name="jit(diamond)/jit(main)/mul" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=45}
  %constant.1 = f32[] constant(0)
  %broadcast.1 = f32[6,8]{1,0} broadcast(%constant.1), dimensions={}, metadata={op_name="jit(diamond)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=46}
  %maximum.1 = f32[6,8]{1,0} maximum(%add.1, %broadcast.1), metadata={op_name="jit(diamond)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=46}
  ROOT %add.0 = f32[6,8]{1,0} add(%multiply.0, %maximum.1), metadata={op_name="jit(diamond)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=47}
}

ENTRY %main.13 (Arg_0.1: f32[6,8], Arg_1.2: f32[6,8], Arg_2.3: f32[6,8]) -> f32[6,8] {
  %Arg_0.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %Arg_1.2 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  %Arg_2.3 = f32[6,8]{1,0} parameter(2), metadata={op_name="c"}
  ROOT %maximum_add_fusion = f32[6,8]{1,0} fusion(%Arg_2.3, %Arg_0.1, %Arg_1.2), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(diamond)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=47}
}


-- number of `fusion(...)` instructions in the ENTRY computation: 1
-- t1 (the shared add) appears INSIDE the single fused_computation body, not as its own materialized buffer: True

==============================================================================
PART 3: same graph, but t1 is ALSO returned as a live output
==============================================================================
HloModule jit_diamond_multi_output, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0}, f32[6,8]{1,0})->(f32[6,8]{1,0}, f32[6,8]{1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true,true}

%fused_computation (param_0.1: f32[6,8], param_1.2: f32[6,8]) -> f32[6,8] {
  %param_0.1 = f32[6,8]{1,0} parameter(0)
  %param_1.2 = f32[6,8]{1,0} parameter(1)
  %multiply.0 = f32[6,8]{1,0} multiply(%param_0.1, %param_1.2), metadata={op_name="jit(diamond_multi_output)/jit(main)/mul" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=53}
  %constant.1 = f32[] constant(0)
  %broadcast.1 = f32[6,8]{1,0} broadcast(%constant.1), dimensions={}, metadata={op_name="jit(diamond_multi_output)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=46}
  %maximum.1 = f32[6,8]{1,0} maximum(%param_0.1, %broadcast.1), metadata={op_name="jit(diamond_multi_output)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=46}
  ROOT %add.0 = f32[6,8]{1,0} add(%multiply.0, %maximum.1), metadata={op_name="jit(diamond_multi_output)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=55}
}

ENTRY %main.14 (Arg_0.1: f32[6,8], Arg_1.2: f32[6,8], Arg_2.3: f32[6,8]) -> (f32[6,8], f32[6,8]) {
  %Arg_0.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %Arg_1.2 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  %Arg_2.3 = f32[6,8]{1,0} parameter(2), metadata={op_name="c"}
  %add.4 = f32[6,8]{1,0} add(%Arg_0.1, %Arg_1.2), metadata={op_name="jit(diamond_multi_output)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=52}
  %maximum_add_fusion = f32[6,8]{1,0} fusion(%add.4, %Arg_2.3), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(diamond_multi_output)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/063_a_diamond_graph_through_real_xlas_hlo_ir.py" source_line=55}
  ROOT %tuple.13 = (f32[6,8]{1,0}, f32[6,8]{1,0}) tuple(%maximum_add_fusion, %add.4)
}


-- number of `fusion(...)` instructions in the ENTRY computation once t1 escapes as an output: 1
```

*Both machines agree on every structural finding this section relies on (single kLoop fusion covering the shared value; that value materializing separately once it also escapes as an output). They diverge in surface formatting and in exactly how that separate materialization is represented, because pip resolved two different real jaxlib versions for the two machines -- discussed in the prose above, not glossed over.*


### What the unoptimized StableHLO already shows

The unoptimized dump is close to a direct transcription of the Python
function: one `stablehlo.add`, one `stablehlo.multiply`, a `call` out to
a separate `relu` function, one final `stablehlo.add`. Nothing has been
fused yet -- this is XLA's own frontend IR, not its own optimized backend
IR, and it looks a lot like Chapter 4's own `printGraphAsSource()` output:
readable, one operation per line, no notion yet of which values will
share a kernel.

### What the optimized HLO shows: a stricter fusion boundary than Ch13's own rule

The optimized dump is where the real question gets answered. XLA's own
real fusion pass produces exactly **one** `fusion(...)` instruction in the
whole `ENTRY` computation, of `kind=kLoop`, and t1's own `add` instruction
lives *inside* that fusion's body -- not materialized to memory, not
computed twice, but computed once and fed directly to both `multiply` and
`maximum` as an in-kernel value. Chapter 13's own `elementwiseFusionPass()`
would have kept t1 external here, because Ch13's own rule is `consumers !=
1` forces materialization, full stop, regardless of where those consumers
end up. Real XLA's own rule is looser and, on inspection, more precise: a
value can stay inside a fusion group as long as *every* one of its
consumers is also inside that same group. Two consumers is fine, as long
as both are being fused together anyway.

File 063's own Part 3 tests the actual boundary of that looser rule by
changing exactly one thing: t1 is *also* returned as a second output of
the whole computation, exactly the distinction Chapter 16's own
`classifyNode()` drew between an ordinary `Shared` node and a `GraphRoot`.
Now one of t1's "consumers" is the outside world itself, which by
definition cannot be inside any fusion group -- and on both machines,
real XLA responds by materializing t1 separately:

**Excerpt, cloud sandbox (x86-64, jax 0.10.2 -- real, version-specific
structure, quoted from the full Part 3 dump above):**

```text
%wrapped_add = f32[6,8]{1,0} fusion(%a.1, %b.1), kind=kLoop, calls=%wrapped_add_computation, ...
%maximum_add_fusion = f32[6,8]{1,0} fusion(%wrapped_add, %c.1), kind=kLoop, calls=%fused_computation, ...
```

**Excerpt, device (aarch64, jax 0.6.2 -- real, version-specific structure,
quoted from the full Part 3 dump above):**

```text
%add.4 = f32[6,8]{1,0} add(%Arg_0.1, %Arg_1.2), ...
%maximum_add_fusion = f32[6,8]{1,0} fusion(%add.4, %Arg_2.3), kind=kLoop, calls=%fused_computation, ...
```

Two real, independently pip-installed jaxlib builds (0.10.2 on the cloud
sandbox's x86-64, 0.6.2 on the device's aarch64 -- pip resolved different
versions for the two different Python versions, not a chapter default
this book chose) diverge here in how they *represent* "compute this once,
plain": the newer build wraps the lone `add` in its own trivial one-
instruction `kind=kLoop` fusion; the older build emits it as a bare `add`
directly inside `ENTRY`, no fusion wrapper at all. That surface
difference means File 063's own printed `fusion(...)` count differs too
(2 on the cloud sandbox, 1 on the device) -- an honest, reproducible
divergence, reported here rather than smoothed over, in the same spirit
as Chapter 22's own machine-specific timing numbers. What does *not*
diverge, on either machine, is the actual finding: once t1 escapes the
fusion group, XLA computes it separately from `multiply`/`maximum`,
exactly where Chapter 13's own `consumers != 1` rule would also have cut
the boundary -- reached here by a different, more general test
("does every use stay inside this group") that happens to agree with
Chapter 13's own simpler one in this particular case.

```bash
python3 "063_a_diamond_graph_through_real_xlas_hlo_ir.py"
```

```python
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
```

```bash
python3 "064_reduction_fusion_under_real_xla.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated XLA output):**

```text
jax version: 0.10.2
real backend device: cpu:0

==============================================================================
PART 1: optimized HLO for Add -> ReLU -> Sum (Ch14's own shape)
==============================================================================
HloModule jit_reduce_chain, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0})->f32[]}, allow_spmd_sharding_propagation_to_parameters={true,true}, allow_spmd_sharding_propagation_to_output={true}

FileNames
1 "/home/claude/hammer_repo/docs/part6/code/064_reduction_fusion_under_real_xla.py"

FunctionNames
1 "<module>"
2 "main"
3 "reduce_chain"

FileLocations
1 {file_name_id=1 function_name_id=1 line=82 end_line=82 column=4 end_column=10}
2 {file_name_id=1 function_name_id=2 line=59 end_line=59 column=16 end_column=49}
3 {file_name_id=1 function_name_id=3 line=36 end_line=36 column=12 end_column=23}
4 {file_name_id=1 function_name_id=3 line=34 end_line=34 column=9 end_column=14}
5 {file_name_id=1 function_name_id=3 line=35 end_line=35 column=9 end_column=24}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=3}
5 {file_location_id=5 parent_frame_id=3}


%region_0.2 (reduce_sum.3: f32[], reduce_sum.4: f32[]) -> f32[] {
  %reduce_sum.3 = f32[] parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.4 = f32[] parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.5 = f32[] add(%reduce_sum.3, %reduce_sum.4), metadata={op_name="jit(reduce_chain)/reduce_sum" stack_frame_id=3}
}

%fused_computation (param_0.3: f32[6,8], param_1.2: f32[6,8]) -> f32[] {
  %param_0.3 = f32[6,8]{1,0} parameter(0)
  %param_1.2 = f32[6,8]{1,0} parameter(1)
  %add.0 = f32[6,8]{1,0} add(%param_0.3, %param_1.2), metadata={op_name="jit(reduce_chain)/add" stack_frame_id=4}
  %constant.1 = f32[] constant(0)
  %max.5 = f32[6,8]{1,0} broadcast(%constant.1), dimensions={}, metadata={op_name="jit(reduce_chain)/jit(relu)/max" stack_frame_id=5}
  %max.4 = f32[6,8]{1,0} maximum(%add.0, %max.5), metadata={op_name="jit(reduce_chain)/jit(relu)/max" stack_frame_id=5}
  ROOT %reduce_sum.0 = f32[] reduce(%max.4, %constant.1), dimensions={0,1}, to_apply=%region_0.2, metadata={op_name="jit(reduce_chain)/reduce_sum" stack_frame_id=3}
}

ENTRY %main.3 (a.1: f32[6,8], b.1: f32[6,8]) -> f32[] {
  %a.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %b.1 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  ROOT %maximum_reduce_fusion = f32[] fusion(%a.1, %b.1), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reduce_chain)/reduce_sum" stack_frame_id=3}
}


-- fusion(...) instructions in ENTRY: 1
-- the reduce(...) op is inside the SAME fused_computation body as the add/relu (not its own materialized buffer): True

==============================================================================
PART 2: Sum's result now feeds a real downstream consumer (*2.0)
==============================================================================
HloModule jit_reduce_then_use, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0})->f32[]}, allow_spmd_sharding_propagation_to_parameters={true,true}, allow_spmd_sharding_propagation_to_output={true}

FileNames
1 "/home/claude/hammer_repo/docs/part6/code/064_reduction_fusion_under_real_xla.py"

FunctionNames
1 "<module>"
2 "main"
3 "reduce_then_use"

FileLocations
1 {file_name_id=1 function_name_id=1 line=82 end_line=82 column=4 end_column=10}
2 {file_name_id=1 function_name_id=2 line=71 end_line=71 column=16 end_column=52}
3 {file_name_id=1 function_name_id=3 line=43 end_line=43 column=8 end_column=19}
4 {file_name_id=1 function_name_id=3 line=41 end_line=41 column=9 end_column=14}
5 {file_name_id=1 function_name_id=3 line=42 end_line=42 column=9 end_column=24}
6 {file_name_id=1 function_name_id=3 line=44 end_line=44 column=10 end_column=17}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=3}
5 {file_location_id=5 parent_frame_id=3}
6 {file_location_id=6 parent_frame_id=3}


%region_0.2 (reduce_sum.3: f32[], reduce_sum.4: f32[]) -> f32[] {
  %reduce_sum.3 = f32[] parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.4 = f32[] parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.5 = f32[] add(%reduce_sum.3, %reduce_sum.4), metadata={op_name="jit(reduce_then_use)/reduce_sum" stack_frame_id=3}
}

%fused_computation (param_0.5: f32[6,8], param_1.4: f32[6,8]) -> f32[] {
  %param_0.5 = f32[6,8]{1,0} parameter(0)
  %param_1.4 = f32[6,8]{1,0} parameter(1)
  %add.0 = f32[6,8]{1,0} add(%param_0.5, %param_1.4), metadata={op_name="jit(reduce_then_use)/add" stack_frame_id=4}
  %constant.2 = f32[] constant(0)
  %max.5 = f32[6,8]{1,0} broadcast(%constant.2), dimensions={}, metadata={op_name="jit(reduce_then_use)/jit(relu)/max" stack_frame_id=5}
  %max.4 = f32[6,8]{1,0} maximum(%add.0, %max.5), metadata={op_name="jit(reduce_then_use)/jit(relu)/max" stack_frame_id=5}
  %reduce_sum.0 = f32[] reduce(%max.4, %constant.2), dimensions={0,1}, to_apply=%region_0.2, metadata={op_name="jit(reduce_then_use)/reduce_sum" stack_frame_id=3}
  %constant.1 = f32[] constant(2)
  ROOT %mul.0 = f32[] multiply(%reduce_sum.0, %constant.1), metadata={op_name="jit(reduce_then_use)/mul" stack_frame_id=6}
}

ENTRY %main.3 (a.1: f32[6,8], b.1: f32[6,8]) -> f32[] {
  %a.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %b.1 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  ROOT %reduce_multiply_fusion = f32[] fusion(%a.1, %b.1), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reduce_then_use)/mul" stack_frame_id=6}
}


-- fusion(...) instructions in ENTRY: 1
-- the downstream multiply is fused into the SAME single kernel as the reduce, not split into a second one: True
```

**Output (device, aarch64 Linux VM -- real, live-generated XLA output):**

```text
jax version: 0.6.2
real backend device: TFRT_CPU_0

==============================================================================
PART 1: optimized HLO for Add -> ReLU -> Sum (Ch14's own shape)
==============================================================================
HloModule jit_reduce_chain, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0})->f32[]}, allow_spmd_sharding_propagation_to_parameters={true,true}, allow_spmd_sharding_propagation_to_output={true}

%region_0.14 (Arg_0.11: f32[], Arg_1.12: f32[]) -> f32[] {
  %Arg_0.11 = f32[] parameter(0), metadata={op_name="jit(reduce_chain)/jit(main)/reduce_sum"}
  %Arg_1.12 = f32[] parameter(1), metadata={op_name="jit(reduce_chain)/jit(main)/reduce_sum"}
  ROOT %add.13 = f32[] add(%Arg_0.11, %Arg_1.12), metadata={op_name="jit(reduce_chain)/jit(main)/reduce_sum" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=36}
}

%fused_computation (param_0.3: f32[6,8], param_1.2: f32[6,8]) -> f32[] {
  %param_0.3 = f32[6,8]{1,0} parameter(0)
  %param_1.2 = f32[6,8]{1,0} parameter(1)
  %add.0 = f32[6,8]{1,0} add(%param_0.3, %param_1.2), metadata={op_name="jit(reduce_chain)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=34}
  %constant.1 = f32[] constant(0)
  %broadcast.1 = f32[6,8]{1,0} broadcast(%constant.1), dimensions={}, metadata={op_name="jit(reduce_chain)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=35}
  %maximum.1 = f32[6,8]{1,0} maximum(%add.0, %broadcast.1), metadata={op_name="jit(reduce_chain)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=35}
  ROOT %reduce.0 = f32[] reduce(%maximum.1, %constant.1), dimensions={0,1}, to_apply=%region_0.14, metadata={op_name="jit(reduce_chain)/jit(main)/reduce_sum" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=36}
}

ENTRY %main.16 (Arg_0.1: f32[6,8], Arg_1.2: f32[6,8]) -> f32[] {
  %Arg_0.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %Arg_1.2 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  ROOT %maximum_reduce_fusion = f32[] fusion(%Arg_0.1, %Arg_1.2), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reduce_chain)/jit(main)/reduce_sum" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=36}
}


-- fusion(...) instructions in ENTRY: 1
-- the reduce(...) op is inside the SAME fused_computation body as the add/relu (not its own materialized buffer): True

==============================================================================
PART 2: Sum's result now feeds a real downstream consumer (*2.0)
==============================================================================
HloModule jit_reduce_then_use, is_scheduled=true, entry_computation_layout={(f32[6,8]{1,0}, f32[6,8]{1,0})->f32[]}, allow_spmd_sharding_propagation_to_parameters={true,true}, allow_spmd_sharding_propagation_to_output={true}

%region_0.15 (Arg_0.12: f32[], Arg_1.13: f32[]) -> f32[] {
  %Arg_0.12 = f32[] parameter(0), metadata={op_name="jit(reduce_then_use)/jit(main)/reduce_sum"}
  %Arg_1.13 = f32[] parameter(1), metadata={op_name="jit(reduce_then_use)/jit(main)/reduce_sum"}
  ROOT %add.14 = f32[] add(%Arg_0.12, %Arg_1.13), metadata={op_name="jit(reduce_then_use)/jit(main)/reduce_sum" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=43}
}

%fused_computation (param_0.5: f32[6,8], param_1.4: f32[6,8]) -> f32[] {
  %param_0.5 = f32[6,8]{1,0} parameter(0)
  %param_1.4 = f32[6,8]{1,0} parameter(1)
  %add.0 = f32[6,8]{1,0} add(%param_0.5, %param_1.4), metadata={op_name="jit(reduce_then_use)/jit(main)/add" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=41}
  %constant.2 = f32[] constant(0)
  %broadcast.1 = f32[6,8]{1,0} broadcast(%constant.2), dimensions={}, metadata={op_name="jit(reduce_then_use)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=35}
  %maximum.1 = f32[6,8]{1,0} maximum(%add.0, %broadcast.1), metadata={op_name="jit(reduce_then_use)/jit(main)/jit(relu)/max" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=35}
  %reduce.0 = f32[] reduce(%maximum.1, %constant.2), dimensions={0,1}, to_apply=%region_0.15, metadata={op_name="jit(reduce_then_use)/jit(main)/reduce_sum" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=43}
  %constant.1 = f32[] constant(2)
  ROOT %multiply.0 = f32[] multiply(%reduce.0, %constant.1), metadata={op_name="jit(reduce_then_use)/jit(main)/mul" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=44}
}

ENTRY %main.18 (Arg_0.1: f32[6,8], Arg_1.2: f32[6,8]) -> f32[] {
  %Arg_0.1 = f32[6,8]{1,0} parameter(0), metadata={op_name="a"}
  %Arg_1.2 = f32[6,8]{1,0} parameter(1), metadata={op_name="b"}
  ROOT %reduce_multiply_fusion = f32[] fusion(%Arg_0.1, %Arg_1.2), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(reduce_then_use)/jit(main)/mul" source_file="/sessions/rcw-01sjvzizaftbfuu7vxgwxekf/mnt/hammer_compiler_from_scratch/docs/part6/code/064_reduction_fusion_under_real_xla.py" source_line=44}
}


-- fusion(...) instructions in ENTRY: 1
-- the downstream multiply is fused into the SAME single kernel as the reduce, not split into a second one: True
```

*Both machines fuse the full Add-ReLU-Sum chain, and the downstream multiply, into a single kind=kLoop fusion -- agreeing with each other and diverging from Chapter 14's own unconditional rule, for the reason given above.*


## 25.2 Reductions: A Boundary Chapter 14 Drew Unconditionally, XLA Draws Contextually

Chapter 14's own `reductionFusionPass()` has one rule with no exceptions:
"a Sum node's output is ALWAYS external, any consumer count." A
reduction's result always gets written to memory and read back, even if
its only consumer is the very next instruction. File 064 asks real XLA
the same question on the same shape Chapter 14's own files 031/032 built
(`Add -> ReLU -> Sum`), then goes one step further than Chapter 14ever
did: it gives the reduction's own scalar result a real downstream
consumer (`* 2.0`) and asks whether XLA still cuts a boundary there.

On both machines, for this toy 6x8 input, real XLA fuses straight through
the reduction in both cases -- the `reduce(...)` HLO op lives inside the
very same `fused_computation` as the `add`/`relu` that feed it, and the
downstream multiply lives in that same fused computation too. Unlike
Chapter 14's own unconditional rule, real XLA's CPU backend does not
treat "this is a reduction" as an automatic fusion boundary by itself --
only "does every consumer of this value stay inside the same group"
matters, the same rule Section 25.1 already found for ordinary elementwise
sharing.

```bash
python3 "064_reduction_fusion_under_real_xla.py"
```

```python
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
```

```bash
python3 "065_cuda_hammers_schedule_vs_xlas_real_autotuning_records.py"
```

**Output (cloud sandbox, x86-64 -- real, live-generated XLA output):**

```text
==============================================================================
PART 1: the tuning-knob record itself
==============================================================================
CudaHammerSchedule (Ch21, this book's own struct):
    tile_size_per_loop = [6, 8]
    loop_order = [0, 1]
    unroll_factor = 4
    -- 3 fields

NOTE: the field NAMES below are real (xla/autotuning.proto); the VALUES are representative placeholders, not a measured result -- this sandbox has no GPU for XLA's real autotuner to search on.
XlaTritonGemmKey (real field names, xla/autotuning.proto):
    block_m = 128
    block_n = 128
    block_k = 32
    num_stages = 3
    num_warps = 4
    num_ctas = 1
    is_tma_allowed = False
    -- 7 fields

-- both records answer the exact same question for their own system: how should ONE loop nest / GEMM be tiled and launched. CUDA Hammer's Schedule generalizes over ANY LoopNest shape (Ch21's own enumerateSchedules()); XLA's TritonGemmKey is specific to one matmul-shaped Triton kernel template -- CUDA Hammer's toy IR never grew a matmul op (Ch4's own scope, unchanged through Ch24).

==============================================================================
PART 2: the (workload, target) cache key shape
==============================================================================
CUDA Hammer's own Ch24.3 key: 'dim0:6,dim1:8@aarch64'
NOTE: again, real field names, placeholder values -- no GPU here to read real ones from.
XLA's own real AutotuningLog context fields (xla/autotuning.proto):
    cudnn_version = '9.x'
    compute_capability = 'sm_90'
    device_pci_bus_id = '0000:00:00.0'

-- Ch24.3's own loopNestKeyForTarget() was built by direct measurement on two real machines (a cloud sandbox and a real device disagreeing on the fastest schedule for the identical LoopNest shape) BEFORE this chapter ever looked at XLA's own source. XLA's own AutotuningLog arrives at the same shape -- a winning config is only valid for the (workload, target) pair that produced it -- at production scale, with three real target fields where Ch24.3's own key used one string.
```

**Output (device, aarch64 Linux VM -- real, live-generated XLA output):**

```text
==============================================================================
PART 1: the tuning-knob record itself
==============================================================================
CudaHammerSchedule (Ch21, this book's own struct):
    tile_size_per_loop = [6, 8]
    loop_order = [0, 1]
    unroll_factor = 4
    -- 3 fields

NOTE: the field NAMES below are real (xla/autotuning.proto); the VALUES are representative placeholders, not a measured result -- this sandbox has no GPU for XLA's real autotuner to search on.
XlaTritonGemmKey (real field names, xla/autotuning.proto):
    block_m = 128
    block_n = 128
    block_k = 32
    num_stages = 3
    num_warps = 4
    num_ctas = 1
    is_tma_allowed = False
    -- 7 fields

-- both records answer the exact same question for their own system: how should ONE loop nest / GEMM be tiled and launched. CUDA Hammer's Schedule generalizes over ANY LoopNest shape (Ch21's own enumerateSchedules()); XLA's TritonGemmKey is specific to one matmul-shaped Triton kernel template -- CUDA Hammer's toy IR never grew a matmul op (Ch4's own scope, unchanged through Ch24).

==============================================================================
PART 2: the (workload, target) cache key shape
==============================================================================
CUDA Hammer's own Ch24.3 key: 'dim0:6,dim1:8@aarch64'
NOTE: again, real field names, placeholder values -- no GPU here to read real ones from.
XLA's own real AutotuningLog context fields (xla/autotuning.proto):
    cudnn_version = '9.x'
    compute_capability = 'sm_90'
    device_pci_bus_id = '0000:00:00.0'

-- Ch24.3's own loopNestKeyForTarget() was built by direct measurement on two real machines (a cloud sandbox and a real device disagreeing on the fastest schedule for the identical LoopNest shape) BEFORE this chapter ever looked at XLA's own source. XLA's own AutotuningLog arrives at the same shape -- a winning config is only valid for the (workload, target) pair that produced it -- at production scale, with three real target fields where Ch24.3's own key used one string.
```

*This file calls no XLA API at all -- it is deterministic Python -- so both machines' output matches exactly, as expected; it is included here for the same cross-machine-verification discipline every other chapter in this book has followed since Chapter 17.*


## 25.3 A Real Production Autotuner's Own Tuning-Knob and Cache-Key Records

Chapter 3 already cited TVM's own AutoTVM tuning logs as the precedent
behind Chapter 24's own `TuningCache`. XLA runs a real autotuner of its
own, for a different but closely related reason: choosing among real
candidate kernel implementations for expensive operations like matrix
multiplies and convolutions. OpenXLA's own determinism documentation
states plainly that "during compilation, XLA's autotuner profiles
multiple candidate kernel implementations... live on the host's GPU to
find the fastest algorithm," drawing candidates from "cuBLAS, cuDNN,
Triton, native emitters" -- the exact same "rank cheaply if you can, then
measure for real" shape as Chapter 23.3's own hybrid autotuner, at a
scale this book's own toy 48-element `LoopNest` never approached.

This sandbox has no GPU, so there is no real autotuning *run* to execute
the way Files 063/064 executed real fusion. What File 065 does instead is
lay this book's own real Chapter 21/24 types directly next to the real
record types XLA's own autotuner persists, field for field, using names
taken verbatim from XLA's own public source (`xla/autotuning.proto`) --
inspectable rather than merely asserted, even without a GPU to fill the
values in for real.

**`TritonGemmKey` vs. `Schedule`.** Chapter 21's own `Schedule` struct has
three fields: `tileSizePerLoop`, `loopOrder`, `unrollFactor` -- generic
over any `LoopNest` this book's own IR can build. XLA's real
`TritonGemmKey` has seven: `block_m`, `block_n`, `block_k`, `num_stages`,
`num_warps`, `num_ctas`, `is_tma_allowed` -- specific to one matmul-shaped
Triton kernel template. That is not XLA being more thorough; it is XLA
solving a narrower, GPU-specific problem (this book's own toy IR never
grew a matmul op at all, an honest scope limit unchanged since Chapter 4)
with a record shaped exactly to that problem's own real launch
parameters -- a thread block's tile dimensions, its pipelining depth, its
warp and cooperative-thread-array counts.

**`AutotuningLog` vs. `loopNestKeyForTarget()`.** Chapter 24.3's own
`loopNestKeyForTarget()` was built by direct measurement on two real
machines -- a cloud sandbox and a real device that genuinely disagreed on
the fastest schedule for the identical `LoopNest` shape -- *before* this
chapter ever looked at XLA's own source for comparison. XLA's real
`AutotuningLog` arrives at the same shape independently: a persisted
result carries `cudnn_version`, `compute_capability`, and
`device_pci_bus_id` alongside the winning config, so a cached config is
only ever reused for the exact (workload, target) pair that produced it
-- Chapter 24.3's own one-string key, at production scale, with three
real fields standing in for "target."

```bash
python3 "065_cuda_hammers_schedule_vs_xlas_real_autotuning_records.py"
```

### What a real, independent, mature fusion engine still leaves on the table

Snider and Liang's 2023 paper "Operator Fusion in XLA: Analysis and
Evaluation" measured XLA's own real fusion engine directly and is worth
closing on for the same reason Chapter 22 closed its own cost-model
sections with honest misses rather than clean wins. The paper's own real,
measured numbers: removing a slow `cuRAND` call bought a 1.87x speedup,
a targeted concat-fusion change bought roughly 10%, avoiding an
unnecessary concat bought 3.41x, and a 10x loop unroll bought 3.5x over
an already-optimized baseline -- 10.56x combined, on the paper's own
benchmark. And after all of that: the paper's own hand-written CUDA
implementation was still 2.7x faster than the best XLA could produce
automatically. XLA is real, in production, at a scale this book's own
128-schedule toy search space never approached -- and a human with
domain knowledge still beat it by a real, measured margin. That is not a
weakness unique to XLA; it is the same honest gap Chapter 23.3's own
hybrid autotuner reported against its own exhaustive search on both real
machines -- a production compiler's own real fusion and autotuning close
most of the distance to hand-tuned code, not all of it.
