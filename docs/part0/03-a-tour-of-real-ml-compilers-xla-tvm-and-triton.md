# 3. A Tour of Real ML Compilers: XLA, TVM, and Triton

**What you will understand:** that the vocabulary Chapter 2 built from a toy arithmetic-expression compiler -- IR, pass, lowering, codegen -- is not a simplification invented for this book. Three real, production ML compilers (XLA, TVM, Triton) are built from exactly those same four moves, under their own names, and this chapter maps each one's real, current, cited documentation onto that vocabulary before CUDA Hammer starts building its own version of it in Part 1.

**What you need to know first:** Chapter 2's vocabulary (lexer/parser/AST are not needed again here, but IR, pass, and codegen are used throughout). No prior exposure to XLA, TVM, or Triton is assumed.

---

Every fact this chapter states about XLA, TVM, or Triton is cited to that project's own real, current documentation -- this chapter's own honesty discipline, stated once in [Getting Started](../getting-started.md) and followed here rather than repeated on every paragraph. That also means this chapter's `code/` directory is empty: there is nothing to compile, because the claims here are about what three *other* projects' own real source and documentation say, not about anything this book measures itself. Chapter 4 returns to genuinely compiled, genuinely run C++ when CUDA Hammer's own graph IR starts.

```text
CHAPTER 2'S VOCABULARY:   source --> AST --> lower --> IR --> passes --> codegen

XLA     : StableHLO graph --> HLO --> HLO passes (fusion, CSE, DCE, ...) --> LLVM --> machine code
TVM     : Relay graph --> TensorIR (TIR) --> schedule primitives (split, fuse, reorder, ...) --> codegen
Triton  : Python kernel --> Triton IR --> block-level scheduling (automatic) --> PTX / target ISA

each row below uses different names for the same four moves: build an IR,
transform it with passes or schedules, then generate real target code.
```

## 3.1 XLA: Fusing at the HLO Level

### Intuition

Picture a factory with two stations on its line. The first station applies the same general-purpose tools to every product that passes through, regardless of which specific machine will assemble it later -- deduplicating parts, discarding anything unused, planning where boxes will sit in the warehouse. The second station is specialized per destination: a product headed to one assembly line gets fittings that product headed to a different line would never need. XLA's own compilation pipeline has exactly this two-station shape, and a further, final station turns the result into real machine code.

```text
Phase 1 (target-independent, runs the same way for any backend):
  StableHLO --> [CSE, target-independent operation fusion, buffer analysis] --> HLO

Phase 2 (backend-specific, GPU shown here):
  HLO --> [further HLO passes: more fusion, pattern-match to optimized library calls] --> optimized HLO

Phase 3 (codegen):
  optimized HLO --> [LLVM: low-level IR, optimization, and code generation] --> machine code
```

### Background

HLO ("High Level Operations") is XLA's own IR -- in Chapter 2's vocabulary, the thing a real ML framework's graph gets lowered *into*, the same role File 003's linear three-address code played for a parsed arithmetic expression. [XLA's own architecture documentation](https://openxla.org/xla/architecture) describes StableHLO, the versioned operation set frontends target, as something that "provides a portability layer between ML frameworks and the compiler" -- multiple frameworks (JAX, PyTorch/XLA, TensorFlow) can all lower into the same StableHLO dialect, which XLA then converts into its internal HLO representation. From there, [XLA's HLO passes documentation](https://openxla.org/xla/hlo_passes) states plainly that "a single HLO Pass can be comprised of one or many compiler optimizations and transformations, and XLA provides several hundred such passes" -- Chapter 2's single, simple `foldConstants()` pass is the same *kind* of thing as each one of those several hundred, just enormously smaller in scope. The documentation names several hardware-independent examples directly: an algebraic simplifier ("a grab bag of simplifications, optimizations, and canonicalizations"), constant folding, dead code elimination, and rematerialization (selectively recomputing a value instead of keeping it live, trading compute for memory).

The specific pass this chapter cares most about is fusion, and [XLA's GPU architecture documentation](https://openxla.org/xla/gpu_architecture) is unambiguous about its importance: "Fusion is XLA's single most important optimization, which groups multiple operations (e.g. addition into exponentiation into matmul) to a single kernel." The reason is exactly Chapter 1's own argument, now confirmed from XLA's own side: "since many GPU workloads tend to be memory-bound, fusion dramatically speeds up the execution by avoiding the writing of intermediate tensors to HBM and then reading them back, and instead passes them around in either registers or shared memory." XLA states this as a hard constraint on what counts as a fusion at all: "no intermediate storage inside the fusion is materialized in HBM," and "a fusion is always compiled to exactly one GPU kernel" -- which is precisely Chapter 1's FUSED diagram (one box, one kernel launch, intermediates held in registers) confirmed as a real production compiler's own design rule, not this book's own simplification.

!!! warning "[COMMON TRAP] Assuming XLA's fusion is a single pass"
    The phrase "operator fusion" suggests one transformation, but XLA's own architecture documentation describes fusion happening at *two* separate points in the pipeline: target-independent fusion during Phase 1 (the same for every backend) and further, backend-specific fusion during Phase 2, where "backends may perform operation fusions beneficial for their architecture." A fusion decision that helps a GPU backend (grouping ops to keep values in fast on-chip memory instead of round-tripping through HBM) is not automatically the same decision a CPU backend would make, because a CPU backend has a different memory hierarchy to reason about -- which is exactly why XLA structures fusion as (at least) two separate passes at two separate pipeline stages instead of one.

## 3.2 TVM: Separating "What" From "How" With Explicit Schedules

### Intuition

A recipe states what a dish is: these ingredients, combined in this way, produce this result. It says nothing about *how* a kitchen actually executes it -- whether one cook does every step in sequence, or three cooks divide the prep work, or the oven preheats while the vegetables are being chopped. Many different kitchen workflows produce the identical dish; some are simply faster. TVM's own design separates a tensor computation's definition from a *schedule* -- a chosen sequence of transformations describing how to actually execute that same, unchanged computation -- in exactly this way.

```text
WHAT (the computation, defined once, in TensorIR):
  C[i, j] = sum_k A[i, k] * B[k, j]              -- one fixed mathematical definition

HOW (a schedule -- one of many valid choices for the SAME computation above):
  schedule 1: split i, bind to GPU threads, k innermost   --> one real kernel
  schedule 2: tile i and j, vectorize the k loop           --> a different real kernel
  schedule 3: reorder k outermost, no tiling at all        --> a third real kernel

all three schedules compute the identical C -- only their real, measured
speed on real hardware differs, which is exactly what CUDA Hammer's own
Part 5 autotuner will search over.
```

### Background

[TVM's own TensorIR documentation](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html) states that "TensorIR is one of the core abstractions in the Apache TVM stack, used to represent and optimize primitive tensor functions" -- TIR is TVM's own IR for a single tensor computation, the same role HLO plays for XLA and CUDA Hammer's own linear IR will play starting in Chapter 4. What TIR adds beyond a plain IR is a *schedule*: a separate, explicit object recording which transformations have been applied to that IR's loop structure. [TVM's own schedule-primitives documentation](https://tvm.apache.org/docs/v0.12.0/how_to/work_with_schedules/schedule_primitives.html) lists the concrete primitives a schedule is built from -- among them, `split` ("can split a specified axis into two axes by factor"), `tile` ("help you execute the computation tile by tile over two axes"), `reorder` ("can reorder the axes in the specified order"), `bind` ("can bind a specified axis with a thread axis, often used in gpu programming"), and `fuse` ("can fuse two consecutive axes of one computation"). Each of these is, in Chapter 2's vocabulary, a pass -- a transformation applied to an IR after it already exists, never touching source text again -- just one that transforms *loop structure* specifically rather than arithmetic expressions the way Chapter 2's own `foldConstants()` did.

TVM's own documentation also distinguishes two different ways a schedule actually gets chosen: DLight, described as rule-based scheduling, and MetaSchedule, described as search-based auto-tuning. That distinction -- a fast, hand-written heuristic versus a slower, measurement-driven search over real alternatives -- is exactly the distinction CUDA Hammer's own Part 5 (Chapter 22, "Cost Models vs. Measurement-Based Autotuning" specifically) builds from scratch, under different names, once CUDA Hammer has its own schedule-like choices (tile sizes, loop orders) to search over.

!!! warning "[COMMON TRAP] Confusing TVM's `fuse` schedule primitive with XLA's operator fusion"
    Both projects use the English word "fuse," and both use it for something that reduces overhead -- but they are not the same operation, applied at different levels of the same pipeline. TVM's `fuse` primitive, per its own documentation, "can fuse two consecutive axes of one computation": it collapses two *loop axes inside a single, already-defined tensor operation* into one axis, a purely structural change to that one operation's own loop nest. XLA's operator fusion groups *multiple separate operations* (an add, an exponentiation, a matmul) into a single kernel, eliminating the HBM round-trip between them. TVM has a separate concept for that second kind of fusion too (operator-level fusion across a Relay graph, distinct from the axis-level `fuse` schedule primitive named here) -- the trap is assuming a shared name means a shared operation, when the two "fuse" operations here apply to entirely different objects (one loop nest's own axes, versus a graph of otherwise-separate operations).

## 3.3 Triton: Programming at the Block Level, Not the Thread Level

### Intuition

Giving driving directions to a single driver is one kind of instruction: turn left here, merge there, one vehicle following one route. Giving a route to a convoy's own dispatcher is a different kind of instruction entirely: the dispatcher receives the destination and decides internally how the convoy's individual vehicles space themselves, merge lanes, and pass through checkpoints -- details the original route never had to specify. CUDA's own programming model is the first kind: a programmer writes the code *one thread* executes, and the hardware replicates that same scalar program across every thread in a block. Triton's programming model is the second kind.

```text
CUDA   -- "Scalar Program, Blocked Threads":
  programmer writes:    one thread's-worth of code, operating on a single acc[i]
  hardware replicates:  that same scalar program across every thread in the block

Triton -- "Blocked Program, Scalar Threads":
  programmer writes:    one block's-worth of code, operating on a whole acc[MB, NB]
  compiler figures out: how that block-level array maps onto real threads,
                         real shared memory, and real coalesced memory loads
```

### Background

[Triton's own programming-guide introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) states this contrast in exactly those terms: Triton is "Blocked Program, Scalar Threads," while CUDA is "Scalar Program, Blocked Threads." Concretely, a Triton kernel's own code operates on a multi-dimensional block directly -- an accumulator shaped `acc[MB, NB]`, not a single scalar per thread -- and the documentation states this block-structured approach can "lead to block-structured iteration spaces that offer programmers more flexibility than existing DSLs when implementing sparse operations." What that higher-level program buys the programmer is a long list of optimizations the Triton compiler now owns instead of the programmer: the same documentation states Triton "manages to apply a broad range of interesting optimization automatically (e.g., automatic coalescing, thread swizzling, pre-fetching, automatic vectorization, tensor core-aware instruction selection, shared memory allocation/synchronization, asynchronous copy scheduling)," achieved through "block-level data-flow analysis, a technique for scheduling iteration blocks statically." In Chapter 2's vocabulary, that scheduling step is Triton's own codegen stage -- the same stage File 003's `interpretIr()` occupied for CUDA Hammer's tiny toy language, just lowering to a real target ISA (PTX, for NVIDIA GPUs) instead of interpreting in place.

!!! warning "[COMMON TRAP] Assuming block-level programming means never thinking about tiling at all"
    Triton automating "automatic coalescing," "shared memory allocation/synchronization," and the rest of that list (per its own documentation, quoted above) does not mean tile sizes themselves disappear from the programmer's job. A real Triton kernel still declares its own block dimensions explicitly (constants conventionally named `BLOCK_M`, `BLOCK_N`, and similar in Triton's own published kernels) -- what the compiler automates is *how* a chosen block maps onto real hardware threads and memory, not *what size* that block should be. Choosing that size well is still an open, measurable, hardware-dependent decision, which is exactly the kind of decision CUDA Hammer's own Part 5 autotuner is built to search over rather than guess at.

## Chapter Summary

Three real, production ML compilers -- XLA, TVM, and Triton -- were shown here to be built from exactly the four moves Chapter 2's toy arithmetic compiler already introduced: build an IR, transform it with passes, generate real target code. XLA lowers a portable StableHLO graph into its own HLO IR and runs it through "several hundred" passes (per XLA's own documentation), including a target-independent fusion stage and a further backend-specific fusion stage, because XLA's own documentation calls fusion "XLA's single most important optimization" for exactly the memory-traffic reason Chapter 1 already quantified from scratch. TVM separates a tensor computation's fixed definition (TensorIR) from a schedule -- an explicit, chosen sequence of primitives like `split`, `tile`, `reorder`, and `fuse` -- and offers both a rule-based scheduler (DLight) and a search-based autotuner (MetaSchedule) for choosing one, foreshadowing the exact distinction CUDA Hammer's own Part 5 builds. Triton inverts CUDA's own programming model from "Scalar Program, Blocked Threads" to "Blocked Program, Scalar Threads" (Triton's own stated phrasing), letting its compiler automate coalescing, shared-memory management, and synchronization that a CUDA programmer would otherwise write by hand -- while still leaving the block-size decision itself to the programmer or an autotuner. Every fact in this chapter was cited to that project's own real, current documentation, never asserted from memory, closing out Part 0's vocabulary-building before Part 1 starts building CUDA Hammer's own graph IR from nothing.

## Self-Check Questions

1. XLA's fusion happens in (at least) two separate phases rather than one. Why might a fusion decision that benefits a GPU backend differ from one that benefits a CPU backend, given the "no intermediate storage materialized in HBM" constraint quoted in Section 3.1?
2. TVM's `fuse` schedule primitive and XLA's operator fusion are both called "fusion," but they operate on different objects. What is each one actually fusing?
3. In Chapter 2's own vocabulary (IR, pass, lowering, codegen), what role does TVM's schedule -- the sequence of `split`/`tile`/`reorder`/`fuse` primitives -- play?
4. TVM offers both DLight (rule-based scheduling) and MetaSchedule (search-based autotuning). Which two chapters, by number, does CUDA Hammer's own table of contents devote to exactly this same distinction, and in which Part?
5. Explain, in your own words, why Triton's "Blocked Program, Scalar Threads" model means the *compiler*, not the programmer, is responsible for automatic memory coalescing.
6. Per the Section 3.3 [COMMON TRAP], what decision does a Triton kernel's own author still have to make, even though intra-block scheduling is automated?
7. Name two hardware-independent HLO passes this chapter cites from XLA's own documentation, and match each to the general vocabulary term (pass) Chapter 2 introduced.
8. Why does this chapter's own `code/` directory contain no compiled `.cpp` files, unlike Chapters 1 and 2 -- what does Getting Started's own honesty discipline say this chapter's claims are checked against instead?

## Where We Go Next

Part 0 is done: Chapter 1 quantified two real, independent costs eager execution pays; Chapter 2 built the general vocabulary (IR, pass, lowering, codegen) from a genuinely working toy compiler; Chapter 3 confirmed that same vocabulary is exactly what three real, production ML compilers are built from, under their own names. Part 1 starts CUDA Hammer itself: Chapter 4 designs the actual graph-shaped IR CUDA Hammer will represent tensors and operations with -- not an arithmetic expression's linear IR anymore, but a real computation graph, genuinely built and genuinely printed in C++, the first piece of CUDA Hammer's own compiler that every later chapter builds on.

## Worked Solutions

1. HBM (high-bandwidth memory) round trips are specifically a GPU memory-hierarchy concern -- a GPU has a real, distinct gap between HBM and on-chip registers/shared memory, which is exactly what XLA's own GPU fusion rule ("no intermediate storage inside the fusion is materialized in HBM") is written to avoid crossing. A CPU backend has a different memory hierarchy (cache levels, not a separate HBM/on-chip split of the same kind), so the specific fusion boundary that helps a GPU kernel avoid an HBM round trip is not necessarily the same boundary that helps a CPU kernel avoid a cache miss -- which is why XLA's own architecture documentation describes backend-specific fusion as a separate stage from the target-independent fusion every backend shares.
2. TVM's `fuse` primitive fuses two consecutive *loop axes* belonging to one already-defined tensor operation -- a structural change to that one operation's own iteration space. XLA's operator fusion fuses multiple *separate operations* (per its own example, "addition into exponentiation into matmul") into a single compiled kernel, eliminating the memory round trip between them. One fuses axes within an operation; the other fuses operations within a graph.
3. A schedule is a pass (or, since it is built from several composable primitives applied in sequence, a sequence of passes) -- it transforms an already-lowered IR (TensorIR) without ever returning to source text, exactly matching Chapter 2's own definition of what a pass is. The IR's meaning (the computation TensorIR represents) stays fixed; only its structure changes, the same correctness bar Chapter 2's `foldConstants()` pass had to clear.
4. Chapter 22, "Cost Models vs. Measurement-Based Autotuning," in Part 5 (Autotuning), per CUDA Hammer's own table of contents.
5. In CUDA's model, the programmer writes code for a single thread and the hardware replicates it across every thread in a block -- each thread's memory access is whatever that thread's own code computes, with no single vantage point that sees the whole block's memory pattern at once. In Triton's model, the programmer's code already describes a whole block's operation on a block-shaped array (`acc[MB, NB]`), so the compiler has full visibility into every memory access the entire block will make before generating any code -- exactly the vantage point needed to decide how to group those accesses into coalesced loads, which is not a decision any single thread's own code could make on its own.
6. The block's own size -- the tile dimensions (conventionally named constants like `BLOCK_M`, `BLOCK_N` in Triton's own published kernels) are still chosen by the kernel's author or by an autotuner, even though the compiler automates how that chosen block maps onto real threads and memory once the size is fixed.
7. The algebraic simplifier ("a grab bag of simplifications, optimizations, and canonicalizations," per XLA's own hlo_passes documentation) and constant folding are both hardware-independent HLO passes cited in Section 3.1; both are passes in Chapter 2's vocabulary -- transformations applied purely to an already-lowered IR, never touching source text -- and constant folding specifically is the exact same operation, under the exact same name, as Chapter 2's own `foldConstants()`, just operating on XLA's HLO graph instead of a linear three-address-code IR.
8. This chapter's claims are about what three other real projects' own documentation and source state, not about anything this book measures or computes itself -- per Getting Started's own fifth honesty-discipline bullet, "real facts this book states about other compilers... are cited to that project's own real, current documentation or source, not asserted from memory." There is nothing here for this book's own toolchain to compile, because nothing here is a claim this book is making about its own code.

---

**Sources cited in this chapter:**

- OpenXLA Project, ["XLA architecture"](https://openxla.org/xla/architecture) -- the three-phase compilation pipeline (target-independent optimization, backend-specific optimization, LLVM-based codegen) and StableHLO's role as "a portability layer between ML frameworks and the compiler," used in Section 3.1.
- OpenXLA Project, ["HLO Passes"](https://openxla.org/xla/hlo_passes) -- "a single HLO Pass can be comprised of one or many compiler optimizations and transformations, and XLA provides several hundred such passes," and the algebraic simplifier / constant folding / dead code elimination / rematerialization examples, used in Section 3.1 and Self-Check Question 7.
- OpenXLA Project, ["XLA:GPU Architecture Overview"](https://openxla.org/xla/gpu_architecture) -- "Fusion is XLA's single most important optimization," the HBM-avoidance rationale, and the "a fusion is always compiled to exactly one GPU kernel" constraint, used in Section 3.1.
- Apache TVM, ["TensorIR"](https://tvm.apache.org/docs/deep_dive/tensor_ir/index.html) -- "TensorIR is one of the core abstractions in the Apache TVM stack, used to represent and optimize primitive tensor functions," and the DLight/MetaSchedule distinction, used in Section 3.2.
- Apache TVM, ["Schedule Primitives in TVM"](https://tvm.apache.org/docs/v0.12.0/how_to/work_with_schedules/schedule_primitives.html) -- the `split`, `tile`, `reorder`, `bind`, and `fuse` primitive descriptions, used in Section 3.2 and its [COMMON TRAP].
- Triton, ["Introduction"](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) -- "Blocked Program, Scalar Threads" versus CUDA's "Scalar Program, Blocked Threads," the automatic-optimization list, and "block-level data-flow analysis," used in Section 3.3.
