# Second systems audit — CUDA Hammer HEAD69a3b19

Read-only audit. Repository unchanged; all new experiments and generated artifacts are under /tmp/hammer-second-systems. No GPU connection/provisioning or remote commands used.

## Confirmed findings

### S2-SYS-01 [P1] Check compiler status and refuse stale shared libraries before JIT measurement

Primary locations:
- /Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part5/code/057_real_codegen_for_a_scheduled_loop_tile_size_and_unroll_factor.cpp:76–89 (pclose status discarded; success inferred from empty compiler output).
- /Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part5/code/059_the_autotuner_pruning_with_a_cost_model_then_measuring_for_real.cpp:186–199 and271–276 (helper result ignored, old fixed-name .so loaded regardless).
- /Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part5/code/062_caching_across_machines_why_the_key_needs_a_target_too.cpp:267–269 (same unconditional compilation/load pattern).
- Shared helper also appears in Chapter20 sources048–050 and Part5 sources058/060/061; corresponding embedded listings inherit it.

Impact: a failed candidate compilation can be timed and correctness-checked using the previous shared object at the same path. If the prior schedule computes the same workload correctly, its output check passes while the tuner associates measurements with a candidate that was not compiled. This undermines the chapter's fresh-code measurement/cache claims. Empty output is not process success, and a diagnostic is not necessarily failure either.

Repro: /tmp/hammer-second-systems/jit_failure.cpp includes the unmodified Chapter23.1 source. It first compiles a library returning37, then demonstrates both ordinary compiler-error diagnostics followed by the caller's unconditional stale load, and a compiler process that exits1 silently. Recorded output in jit_failure.out:

    normal compile error; helper success=false; following caller's unconditional dlopen still runs stale function=37
    compiler exit1, empty stderr; helper returned success=true; stale function=37

Command: clang++ -std=c++17 -O2 /tmp/hammer-second-systems/jit_failure.cpp -o /tmp/hammer-second-systems/jit_failure && /tmp/hammer-second-systems/jit_failure

Correction/pass criteria: inspect pclose/WIFEXITED/WEXITSTATUS; reject a failed build before dlopen/timing/cache insertion; compile to a fresh candidate artifact and atomically promote only successful builds. Both failed-compilation repro cases must be rejected and never call or time the old function. Add a regression with a preexisting artifact, not merely an absent-file test.

Primary documentation: Apple's pclose manual states it returns the child command's wait status, not its output: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/pclose.3.html . POSIX/OpenGroup URLs returned403 during this audit; the Apple primary manual was successfully read.

### S2-SYS-02 [P1] Serialize floating-point literals without rounding valid constants to zero

Primary locations:
- /Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part4/code/041_lowering_a_whole_fused_graph.cpp:500–502 and533–535.
- CPU scalar helper040:562; vector program helpers045:620,046:675,047:708/741.
- CUDA generators042:552,043:515,044:543/578/651.

Impact: C++17 floating std::to_string uses fixed six-decimal formatting. Finite float1e-7 becomes0.000000f in generated source. This is not FMA/reduction reassociation or a stated broadcast limitation: the constant itself changes before arithmetic. The issue affects serialized fixture inputs and actual Graph Const nodes in scalar/SIMD/CUDA standalone generated programs. JIT function calls receiving existing float buffers are not intrinsically affected; the finding is specifically literal-producing paths.

Repro: /tmp/hammer-second-systems/float_literals.cpp includes unmodified Chapter17.3 source, builds a legal one-element graph with inputa=1e8 and constantc=1e-7, and generates/compiles the program. Both operands and the expected result are finite. Recorded float_literals.out:

    interpreter out: 10
    serialized constant: {0.000000f}
    a: 100000000.000000
    c: 0.000000
    out: 0.000000

Commands:

    clang++ -std=c++17 -O2 /tmp/hammer-second-systems/float_literals.cpp -o /tmp/hammer-second-systems/float_literals
    /tmp/hammer-second-systems/float_literals
    clang++ -std=c++17 -O2 /tmp/hammer-second-systems/float_generated.cpp -o /tmp/hammer-second-systems/float_generated
    /tmp/hammer-second-systems/float_generated

Correction/pass criteria: use locale-independent round-trip float formatting (hexfloat or to_chars/max_digits10 scientific formatting with a valid float suffix), shared across scalar/SIMD/CUDA emitters. Add small-magnitude and precision-sensitive constants; generated graph must retainc's float bits and return10 in this case. Six-decimal stdout serialization used for program-to-interpreter comparisons should likewise be reviewed for lost precision.

Primary source: WG21 P2587R3 documents the pre-C++26 sprintf-based floating to_string behavior and explicitly shows1e-7 rounding to zero: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2587r3.html . The book targets C++17, so later C++26 changes do not cure this implementation.

## Coverage and non-findings

Reviewed Chapter17–24 source/emitter/JIT/schedule/cache contracts and their key explanatory claims; inspected AppendixA/C/D/E/F sources/tool claims and cross-cutting validation runners/evidence. Fresh local compile/run reproductions above use native Apple Clang arm64 C++17. Existing GPU evidence was inspected as saved local logs only; no fresh GPU run was attempted. No exhaustive new host suite or external LLVM/MLIR/Polly/TVM/Valgrind run is claimed.

Prior corrections that remain appropriately scoped: FMA and vector-reduction rounding are explicitly relaxed; direct numeric comparison rejectsNaN; Chapter17 openly excludes interior-singleton broadcast indexing; architecture-only cache identity is disclosed as teaching scope; native macOS missing/proc is distinguished from missing SIMD; SIMD shape fallback is not claimed to be CPU-ISA dispatch; profiling counters are distinguished from physical PMU measurements. These explicit limitations are not reclassified as new bugs. Existing A100 four-fixture numeric results and12direct sanitizer reports remain valid for their concrete cases; they do not test new literal values or compiler-failure recovery.

Root owns the additional empty-input validation coverage finding (0/0 or no-source passes); not duplicated here. No additional confirmed accuracy defect identified in assigned appendices during this bounded pass. Main gaps: no hardware/SSE/pre-AVX CPU execution, no independent rerun of historical tool-specific appendix transcripts, and no exhaustive fuzzing of arbitrary schedules/graphs.
