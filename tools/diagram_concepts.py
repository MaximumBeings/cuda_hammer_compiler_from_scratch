"""Hand-authored learning diagrams; these depict contracts, not measurements."""
# Each row lists three section diagrams: title | three connected stages | boundary.
ROWS = {
1: [
('Eager execution materializes intermediates','input array|operation A -> buffer|operation B -> output','Each materialized stage adds logical array reads and writes.'),
('Launch overhead is a per-launch cost','host submission|kernel launch|device work','A fixed launch-cost model is illustrative; real timings require a GPU.'),
('Fusion removes intermediate traffic','input load|local operation chain|final output store','Logical traffic savings are not a measured speedup.')],
2: [
('A frontend makes structure explicit','source text|parser|syntax tree','Parsing establishes syntax; semantic validation is a separate step.'),
('Lowering changes representation','syntax tree|linear instructions|named intermediate values','The representation must preserve the program semantics.'),
('Passes operate before code generation','validated IR|transformation passes|generated code','Execute and compare with a reference after lowering.')],
3: [
('XLA optimizes a high-level graph','framework operation graph|HLO transformations|backend executable','Actual fusion choices depend on backend and compiler version.'),
('TVM separates computation from scheduling','tensor computation|schedule decisions|lowered implementation','Scheduling APIs evolve; the examples target documented versions.'),
('Triton exposes blocks of values','program instance|vector-shaped tile|masked memory operations','A block-level program is lowered to GPU threads and instructions.')],
4: [
('A DAG can represent shared work','input values|shared producer|multiple consumers','Sharing is explicit; a tree would duplicate the producer.'),
('Nodes carry operations and dependencies','operands|operation and shape|result value','Shape and type contracts are part of graph validity.'),
('Topological order respects dependencies','earlier operands|new node|later consumers','Public mutation can still require explicit cycle validation.')],
5: [
('Text describes a graph','input declarations|operation declarations|output declaration','The supported grammar is deliberately small.'),
('Names resolve to existing values','tokenized statement|symbol-table lookup|new graph node','A resolved operand is a dependency, not copied computation.'),
('Reject invalid names early','unknown or duplicate name|diagnostic|no executable graph','Do not continue code generation after a frontend error.')],
6: [
('Graph validation checks contracts','operation arity|operand compatibility|valid graph','Validation is necessary before interpretation and lowering.'),
('Shape inference follows dependencies','known input shapes|operator shape rule|inferred output shape','Broadcasting must follow the implementation\'s actual supported rules.'),
('A mismatch should stop compilation','incompatible operands|shape diagnostic|reject graph','A failed shape check is not a runtime result.')],
7: [
('Readable IR reveals structure','graph in memory|named node listing|inspect dependencies','A printer is a debugging interface, not a correctness proof.'),
('Round trips test the text interface','graph|print then parse|equivalent graph','Compare graph semantics rather than incidental pointer identity.'),
('DOT turns dependencies into edges','operation nodes|operand edges|rendered DAG','Rendering requires Graphviz; DOT output alone is source text.')],
8: [
('Analysis observes; transforms modify','input graph|analysis or transform|report or changed graph','Keep each pass\'s mutation contract explicit.'),
('Canonical names reduce incidental differences','original names|deterministic renaming|stable printed IR','Renaming does not change tensor arithmetic.'),
('Validation brackets transformations','validate input|run pass|validate output','Structural validity alone does not prove semantic equivalence.')],
9: [
('Fold known values before execution','constant operands|evaluate operation|constant result','The folding arithmetic must match the declared numerical contract.'),
('Keep only values needed by outputs','declared outputs|backward reachability|retained nodes','Side effects would require additional liveness rules.'),
('Pass order can expose more work','constant folding|dead-code elimination|smaller reachable graph','Measure graph changes and compare outputs independently.')],
10: [
('Reuse structurally identical expressions','operation and operands|structural key|existing result','Only merge operations whose semantics and attributes agree.'),
('A key defines the equivalence being used','opcode|operand identities|shape and attributes','Floating-point operand reordering is not automatically valid.'),
('Combine local optimizations','fold constants|merge repeated work|remove dead nodes','Repeat only to an explicitly defined fixed point or pass budget.')],
11: [
('Identity rewrites need numerical assumptions','expression x + 0|declared arithmetic rules|candidate replacement x','Signed zero and exceptional values can distinguish rewrites.'),
('Algebra is not all floating-point semantics','expression x * 0|NaN and infinity cases|guard or restrict rewrite','Real-number identities may fail under IEEE floating point.'),
('A complete pass pipeline retains a contract','input graph|fold / CSE / simplify / DCE|checked output','Tests must include boundary values as well as ordinary inputs.')],
12: [
('Arithmetic intensity compares work and traffic','operation count|modeled bytes moved|FLOPs per byte','Logical array traffic is not necessarily measured DRAM traffic.'),
('Roofline is a performance bound','peak compute|bandwidth times intensity|minimum of the two bounds','Peak specifications and simplified models are not benchmark results.'),
('Fusion changes the modeled traffic','materialized intermediates|fused local values|fewer array transfers','Register pressure and launch details can still affect runtime.')],
13: [
('A fused node contains a local program','external operands|ordered local steps|one output','Local step references must preserve dependencies.'),
('A shared value constrains duplication','shared producer|multiple consumers|reuse or deliberate recomputation','One consumer cannot discard work still needed by another.'),
('Count what the fusion pass changes','original graph|fusion transformation|new launches and byte model','Keep analytical savings separate from device measurements.')],
14: [
('Reduction changes the output shape','elementwise producer|sum across an axis|smaller result','A reduction is not an elementwise expression substitution.'),
('A reduction creates a dependency barrier','many input elements|completed aggregate|downstream consumer','Cross-thread execution needs a valid synchronization strategy.'),
('Fuse the producer into reduction work','load input|local producer then accumulate|store aggregate','Reduction order can change floating-point rounding.')],
15: [
('Lower tensor work to iteration bounds','output shape|loop extents|per-element body','Each output coordinate must be visited as intended.'),
('Tile boundaries need a tail guard','outer tile|inner index|bounded valid element','The last tile may be shorter than the nominal tile size.'),
('Loop fusion and operator fusion differ','separate loop bodies|combined iteration traversal|shared loop structure','A shared loop does not by itself remove every temporary array.')],
16: [
('Not every boundary has the same cause','shape boundary|shared-value boundary|backend constraint','Classify the reason before deciding whether to fuse.'),
('Larger fusion is not always better','candidate fused group|resource or cost limit|accept or split','Cost thresholds are policies; validate them on target hardware.'),
('Explain the final partition','validated graph|fusion decisions|kernel-group report','A report should identify why each boundary remains.')],
17: [
('An array interpreter is the reference path','input arrays|per-element evaluation|reference output','Compare values, not just an aggregate checksum.'),
('Generate loops from the same IR','loop bounds|emitted scalar statements|compiled host function','Generated text still needs compilation and execution.'),
('Lower every reachable graph group','fused graph|scheduled functions|complete output','Intermediate allocation and lifetime remain explicit.')],
18: [
('Map output elements to CUDA threads','flat thread index|bounds guard|local fused arithmetic','Compilation does not establish device execution correctness.'),
('A reduction needs race-free accumulation','zero-initialized output|per-thread contributions|atomic accumulation','Atomics avoid a data race but do not fix summation order.'),
('Validate the full generated GPU program','emit CUDA|compile for target|execute against reference','Check API calls, launch errors and completed device results.')],
19: [
('SIMD processes multiple elements per instruction','vector-width block|lane-wise operations|scalar tail','ISA-specific code requires compatible hardware or dispatch.'),
('FMA changes the rounding contract','multiply and add|optional fused instruction|one-rounding result','Fused and unfused arithmetic need not be bit-identical.'),
('Choose a supported execution path','target capability|specialized or scalar code|compared output','Global ISA flags can undermine a baseline fallback.')],
20: [
('JIT crosses a compiler and loader boundary','generated source|shared-library compiler|loaded function','Check compilation, symbol lookup and library lifetime.'),
('A module can export several graph functions','function definitions|one shared library|named entry points','C linkage controls exported symbol naming; ABI details still matter.'),
('Invoke generated code through a checked interface','validated inputs|function pointer call|reference comparison','Only trusted generated source should enter this example toolchain.')],
21: [
('Tile size defines candidate loop partitions','iteration extent|candidate tile sizes|guarded tiled loops','Candidates must preserve coverage, including tails.'),
('Loop order changes memory traversal','same index space|different nesting order|different access stride','Equal arithmetic does not imply equal locality.'),
('A schedule combines several decisions','tile choice|loop order|unroll factor','Search-space size grows as decisions are combined.')],
22: [
('An analytical model estimates a cost','tile and unroll|loop-overhead assumptions|predicted score','A score is not a measured elapsed time.'),
('Stride models approximate locality','array layout|iteration order|memory-access pattern','Caches and hardware prefetching complicate the prediction.'),
('Compare predictions with actual measurements','candidate schedules|compile and measure|rank observed results','Control inputs, warm-up and noise before drawing conclusions.')],
23: [
('Scheduled source must remain correct','schedule parameters|generated tiled loop|validated output','Reject schedules that change coverage or arithmetic semantics.'),
('Search the combined schedule space','tile / order / unroll|candidate code|correctness filter','Only correct candidates should be benchmarked.'),
('Hybrid tuning prunes then measures','model ranking|retained candidates|measured winner','Pruning can discard the true fastest candidate.')],
24: [
('Persist a verified tuning result','workload key|selected schedule|cache entry','A cached result needs a validity policy and version.'),
('A key must capture relevant assumptions','graph and shape|compiler and options|target identity','An incomplete key can reuse an unsuitable schedule.'),
('Cross-machine reuse requires compatibility','saved target metadata|current environment|accept or retune','Architecture name alone does not identify a complete machine.')],
25: [
('Inspect the actual production IR','JAX computation|lowered HLO|backend optimization','HLO shape and fusion decisions are version- and target-dependent.'),
('Reduction fusion depends on backend rules','producer computation|reduction boundary|optimized executable','Do not infer GPU behavior from a CPU-only compiler run.'),
('Production tuning records differ from schedules','workload and target|candidate implementation|recorded selection','Inspect real records before claiming a measured result.')],
26: [
('TIR separates a computation from its schedule','tensor program|split and vectorize|lowered implementation','Use the pinned TVM API version in the example.'),
('Graph and loop IR have different responsibilities','Relax graph|fusion and lowering|TIR functions','A graph fusion group still needs a valid low-level implementation.'),
('A tuning database stores contextual results','workload representation|target and measurements|tuning record','A database lookup is valid only under its matching rules.')],
27: [
('Triton lowers a block-level program','tile operations|TTIR and GPU lowering|machine code','Offline compilation is distinct from running on the device.'),
('Language structure affects fusion opportunities','one program body|local tile values|one kernel result','A compiler does not arbitrarily merge independent launches.'),
('Autotuning selects among declared candidates','configuration list|correctness and timing|selected configuration','An offline configuration listing is not measured tuning.')],
28: [
('Capture and code generation are distinct stages','Python execution|Dynamo graph capture|Inductor backend','Backend behavior depends on graph, target and version.'),
('A graph break splits the captured region','captured operations|unsupported boundary|next captured region','A capture boundary is different from a kernel fusion choice.'),
('Inspect tuning policy and generated code','compiler configuration|candidate implementations|selected code','An enabled option alone does not prove which path executed.')],
29: [
('Online softmax carries stable partial state','block scores|running max and normalizer|rescaled output accumulator','Compare with a numerically independent dense reference.'),
('Masked tiles must guard every tail load','query tile|key/value tail masks|bounded memory access','Aligned test shapes cannot expose all tail errors.'),
('Attention dispatch uses available implementations','framework call|backend eligibility|selected attention kernel','A high-level call alone does not prove a particular backend ran.')],
30: [
('Affine quantization needs a representable zero','calibration range including zero|scale and zero point|quantized values','Handle one-sided, constant and invalid ranges explicitly.'),
('Quantized products need sufficient accumulation width','narrow integer inputs|wider dot accumulator|rescaled output','Accumulator overflow bounds depend on the reduction length.'),
('Container storage and kernel arithmetic differ','quantized tensor representation|supported operator|actual execution path','Backend-specific support must be checked for the pinned version.')],
31: [
('Calibration parameters define preprocessing','pixel values|brightness / contrast / normalization|float features','Parameter units and clipping choices affect the result.'),
('Shifted inputs make a fixed stencil explicit','pre-shifted image planes|weighted elementwise sum|filtered output','Preparing shifted inputs is part of end-to-end cost.'),
('Validate a fused preprocessing pipeline','original graph|fused SIMD implementation|reference comparison','Kernel-only timings can omit preparation and allocation.')],
32: [
('Mean-centering is one normalization component','token embedding|subtract feature mean|centered vector','Centering alone is not full LayerNorm.'),
('Pooling produces a bag representation','token embeddings|average across tokens|small linear classifier','This model omits token order and attention.'),
('Compile the supported classifier graph','static graph|fusion and SIMD lowering|checked class scores','This is a small fixed classifier, not a full transformer compiler.')],
33: [
('The heat stencil depends on neighboring values','left / center / right|FTCS update|next-time value','Boundary conditions and the stability parameter matter.'),
('Unrolling time creates a fixed-depth graph','time step one|repeated stencil chain|fixed final step','A finite unrolled graph is not a general time loop.'),
('Validate fusion across stencil stages','reference evolution|fused implementation|error and boundary checks','Temporary preparation and graph construction also have costs.')],
34: [
('A put payoff can use a maximum','strike minus asset price|max with zero|terminal payoff','The payoff is one part of the pricing model.'),
('Backward induction prices a European put','terminal nodes|discounted expected continuation|root price','Risk-neutral parameters require a valid binomial model.'),
('American exercise adds a local decision','continuation value|compare immediate exercise|backward maximum','Numerical agreement is not a market valuation guarantee.')],
}
APPENDICES = {
'A':('Toolchain capability is explicit','host compiler|optional SIMD and CUDA|record installed capabilities','Compilation and device execution are different validation steps.'),
'B':('Reason from contracts to consequences','identify assumptions|trace the example|check the worked answer','A passed example is not a universal proof.'),
'C':('Compiler infrastructure provides reusable layers','LLVM IR|MLIR dialect lowering|backend libraries','External tools remain optional and version-dependent.'),
'D':('Affine loops can expose scheduling freedom','iteration domain|dependence analysis|legal transformed schedule','Non-affine control flow and side effects limit applicability.'),
'E':('Measure the resource the claim concerns','controlled workload|counters and elapsed time|qualified conclusion','Instrumentation overhead and target conditions belong in the record.'),
'F':('Map concepts across interfaces','framework graph|fusion and schedule concepts|C++ teaching implementation','Related concepts do not imply identical implementation behavior.'),
}
