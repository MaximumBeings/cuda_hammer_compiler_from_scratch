# Independent second foundations audit — CUDA Hammer 69a3b19

Read-only audit of current Chapters 1–16 and copied numerical/shape/counting patterns. No repository file was edited. No GPU was connected, provisioned, or used. Artifacts: `/tmp/hammer-second-foundations/`.

## Five confirmed findings

### F2-01 — P1: The published constant-folding example writes outside its temporary array

**Source:** `/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part0/code/004_constant_folding_pass.cpp`, **153–157**, with sparse ids retained at **197**. Embedded copy: `docs/part0/02-what-a-compiler-pipeline-looks-like.md:679` and `:723`.

The array is allocated with `temps(ir.size(), 0)`, but folding removes instructions without renumbering `destTemp`. The very first existing sample, `a + (2 * 3)`, starts with t0=2*3 and t1=a+t0. Folding removes t0, leaving one instruction that writes t1. Thus `temps[1]` writes beyond a one-element allocation. This is not a contrived malformed IR or a merely unsupported input: it occurs in the example's unmodified main.

**Reproduction:** compile `/tmp/hammer-second-foundations/folded_temp_oob.cpp` (an unchanged copy of the published source) with `clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all`; run it. Saved stderr/stdout: `folded_temp_oob.txt` records **AddressSanitizer: heap-buffer-overflow, WRITE of size 4**, originating in interpretIr. Normal runs happened to print matching answers despite this undefined behavior.

**Correction:** either use an id-keyed temporary map/checked storage sized for the largest id, or compact all surviving definition/use/result ids consistently. Test original samples with ASan as well as folding away early/middle definitions. The sibling pre-fold interpreter in `003:238` also assumes dense ids, but lower() itself emits dense ids there; the demonstrated defect is the folded consumer in 004.

**Primary reference:** [Clang AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html).

### F2-02 — P2: maxChainLength does not cap the number of steps in a fused body

**Source:** `/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part3/code/037_a_cost_based_boundary_capping_how_large_one_fused_kernel_can_grow.cpp`, **344–374**. Claim: `docs/part3/16-fusion-boundaries.md:639`, “a cap on how many steps one FusedElementwise body may contain.” Related copied policy: `docs/part3/code/038_putting_it_together_a_full_fusion_report_through_chapter_8s_pass_manager.cpp:330–336`.

computeChainDepths uses the maximum predecessor depth rather than total fused-subgraph size. It cuts long chains but misses branching trees. An eight-input balanced Add tree has seven Add operations, all internal producers single-consumer; boundedFusionPass(g,3) emits a single **seven-step** body. No invalid graph, shared-producer complication or nonpositive cap is required.

**Reproduction:** `fusion_cap.cpp`; `fusion_cap.txt`: `requested cap=3, largest fused body=7 steps`, exit 42 denoting reproduced violation.

**Correction:** count distinct operations in the candidate group and partition with a real size bound, then assert every emitted body's size. Alternatively describe/rename the policy as a path-depth heuristic and remove its hard body-size guarantee. Add branching-tree and merge cases, not only straight-chain tests. A separate input-validation issue is that cap=0 reaches a modulo-by-zero expression at 374; validate positive policy inputs too.

### F2-03 — P2: The size-overflow fix does not extend to loop totals or logical byte totals

**Sources:**
- `/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part3/code/034_tiling_a_loop_splitting_one_loop_into_two_correctly_at_the_boundary.cpp`, **42–44**, unchecked `total *= l.extent`; copies in `033:368–370` and `035:353–355`.
- `/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part3/code/026_measuring_fusions_effect_on_cuda_hammers_own_graph.cpp`, **206–215**, unchecked aggregate addition and double-to-long-long byte conversion. Same conversion pattern in `029:391–398`, `032:428–435`, `038:520–527`.

The new numElements check is correct locally but does not make subsequent arithmetic safe. A LoopNest with extents LLONG_MAX and 2 overflows. For the byte case, a single ReLU on shape [2147483647,2147483647] has an individually representable element count; its logical input/output count also fits long long, but multiplying by four bytes through double produces approximately 3.68935e19, which cannot be converted to long long. These analytical functions do not allocate the described tensor, so an allocation failure does not protect them.

**Reproductions:** `loop_product_overflow.cpp/.txt` and `byte_count_overflow.cpp/.txt`. UBSan explicitly reports signed integer overflow and out-of-range float-to-integer conversion, respectively.

**Correction:** use checked integer addition/multiplication throughout shape → loop → traffic accounting. Reject an unrepresentable total with an explicit exception, or choose a documented wider representation. Avoid the lossy double detour for integer byte counts. Test representable operands whose aggregate/product is not representable.

**Primary reference:** [Clang UBSan checks](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html).

### F2-04 — P2: The first compiler's lexer and arithmetic can execute undefined behavior instead of reporting invalid arithmetic

**Source:** `/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part0/code/003_parsing_lowering_and_ir.cpp`, **48**, unchecked decimal accumulation, and **224–229**, unchecked signed arithmetic/division. Copies in `004:54` and `004:149–150`.

An otherwise syntactically ordinary integer literal `2147483648` overflows while being lexed, rather than producing an out-of-range diagnostic. Likewise applyOp has no handling for zero divisors, INT_MIN/-1, or overflowing signed operations; constant folding calls the same function on compile-time values. The language explanation does not establish a checked, wrapping, or explicitly undefined arithmetic contract, so describing the parser/folder as handling ordinary integer expressions hides this boundary.

**Reproduction:** `integer_literal_overflow.cpp/.txt`; UBSan reports `2147483640 + 8 cannot be represented in type int` at source line 48. Division/other arithmetic branches were inspected; the saved executed counterexample is the literal overflow, not every arithmetic category.

**Correction:** range-check literal parsing and define the language's arithmetic semantics. For this teaching compiler, checked arithmetic with explicit diagnostics is straightforward; reject zero division and overflow before executing host C++ operations. Reuse that helper in folding and runtime interpretation, with tests for both paths.

### F2-05 — P2: Later copies of inferShapes bypass the earlier negative-dimension rejection

**Source:** `/Users/oluwaseyiawoga/Documents/Resume_2026/cuda_hammer_compiler_from_scratch/docs/part3/code/033_from_shapes_to_loop_nests_giving_fused_nodes_real_iteration_bounds.cpp`, **187–188**. Equivalent direct assignments appear in 026:181, 029:327, 030:220, 031:178, 032:180, 035:180, 036:181, and 038:191.

Chapter 6's current 009/010 helpers reject nonpositive dimensions, but these later copies directly assign `declaredShapes.at(id)` without that check. Calling the copied 033 inference function with a normal Input and declared shape [-2,3] returns that invalid shape successfully. In some full example paths numElements later rejects it, but that does not make inferShapes itself enforce its advertised invariant; different callers and standalone shape propagation bypass that later check.

**Reproduction:** `copied_negative_shape.cpp/.txt`: `inferShapes accepted [-2, 3]`, exit 0. This is a missing rejection, not a request to implement dynamic or zero-sized shapes, which the book explicitly excludes.

**Correction:** centralize validated static shape construction or propagate the same checks into every inference entry point before assigning leaf shapes. Apply the regression to each copy rather than testing only Chapter 6's canonical version.

## Coverage, evidence, and limits

- Independently re-read the relevant current parser/IR/pass/fusion/shape/counting implementations and explanatory contracts; checked later source-copy patterns where the same code recurs. Relevant source boundary cases above were executed directly against commit 69a3b19.
- Freshly compiled and ran **all 38 original Part 0–3 standalone examples** with Apple Clang, C++17, AddressSanitizer, UBSan, and float-cast-overflow instrumentation. **37 pass; 004 fails ASan.** Full outputs and summary are in `/tmp/hammer-second-foundations/baseline-sanitized/`. This materially differs from the earlier unsanitized all-pass result.
- `/tmp/hammer-second-foundations/reproduce.py` reconstructs all saved targeted counterexamples; `sanitize_all.py` reproduces the complete sanitized baseline campaign. Logs are actual local macOS/Apple Clang observations, not invented expected output or GPU results.
- Known, explicitly disclosed limitations were not promoted to new defects: representative-uniform-scalar evaluators rather than general tensor indexing; logical byte models rather than measured HBM traffic; same-loop-extent candidate checks without full dependence analysis; conservative edge-count fusion; lack of explicit output Values; trusted single-output graph construction; omitted NaN/infinity text serialization. The later general broadcasting backend is being independently reviewed by the systems agent.
- An additional exploratory duplicate-debug-name printer example is saved, but not ranked among these five findings: whether Graph source serialization promises uniqueness needs an explicit API-contract decision. Finite signed/scientific constants and earlier strict zero-rewrite fixes were inspected; this pass does not assert exhaustive floating-point correctness.
- No exhaustive proof, fuzzing campaign, concurrency audit, GPU kernel execution, or cross-platform sanitizer reproduction was performed. These results do not justify an all-clear. Repository status remained clean during the read-only audit.
