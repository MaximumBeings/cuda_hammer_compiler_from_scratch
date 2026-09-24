#!/usr/bin/env python3
# Appendix F: From torch.compile and TVM's Python API to C++: A Rosetta
# Stone
# 107_the_same_autotune_and_cache_problem_through_tvms_own_real_metaschedule.py
#
# Section F.2 (capstone) -- the TVM half of this appendix's second
# side-by-side pair. Reuses Chapter 26's own File 068 completely unchanged
# (TVM's own real, current auto-tuning system -- MetaSchedule, which the
# real, installed package confirms has replaced BOTH AutoTVM and Ansor:
# `tvm.autotvm` and `tvm.auto_scheduler` do not exist in this real 0.26.0
# release; `tvm.s_tir.meta_schedule` does), re-run fresh this session so its
# own real MISS-then-HIT numbers sit directly next to File 108's own fresh
# CUDA Hammer run of the matching-size problem through Chapter 23.3's real
# hybrid autotuner and Chapter 24.1's real `TuningCache`.
#
# OpenXLA's real determinism docs (Chapter 25) described XLA's own
# autotuner as: profile real candidates on real hardware, pick the fastest.
# TVM's own real MetaSchedule docs describe the same three-part shape by
# name: a CostModel (default XGBoost) ranks candidates cheaply, a
# SearchStrategy (default EvolutionarySearch) sends the promising ones to a
# Builder/Runner for REAL measurement, and a Database (default
# JSONDatabase) persists the winning "trace + measured run time" --
# TVM's own real name for exactly what Chapter 24's own `TuningCache`
# built by hand: `loopNestKey(nest) = serializeSchedule(schedule)` lines in
# a plain-text file.
#
# This sandbox has no GPU (same limitation as every chapter since Ch18),
# and a full EvolutionarySearch run is not a quick, small, reproducible
# demo the way Files 066/067's own experiments were -- so, matching Chapter
# 25.3's own honest choice when the same situation came up for XLA's
# autotuner, this file does not run a full auto-search. What it DOES do
# for real: build a real schedule (File 066's own split+vectorize),
# measure it for real on THIS machine, and commit that real measurement to
# a real MetaSchedule `Database` -- TVM's own actual `TuningRecord` and
# `JSONDatabase` classes, not a hand-rolled stand-in -- then load it back
# with a fresh `Database` object, the same MISS-then-HIT shape as Chapter
# 24.1's own `TuningCache` demonstration, using TVM's own real persistence
# format instead of this book's own.
#
# One real design difference surfaced while getting this file's own Part 2
# to work at all, and is worth naming rather than quietly fixing: this
# file's own FIRST attempt committed the SCHEDULED module (after split and
# vectorize) as the workload, and the real `JSONDatabase` constructor threw
# a real, internal consistency error reopening it. The reason, found by
# printing the record file's own real content: a `TuningRecord`'s `trace`
# field is a REPLAY LOG (`GetSBlock`, `GetLoops`, `Split`, `Vectorize`, in
# order), not a snapshot -- so the `workload` half of a record has to be
# the ORIGINAL, UNSCHEDULED module the trace starts from, not the already-
# scheduled result. A real, notable design difference from Chapter 24's own
# `TuningCache`, which stores a fully-specified `Schedule{tileSizePerLoop,
# loopOrder, unrollFactor}` struct directly: TVM's own real persistence
# format stores a starting point plus a sequence of edits, and replays the
# edits to reconstruct the winning schedule, rather than storing the
# winning schedule's own final shape. Part 2 below uses the corrected,
# real design -- the UNSCHEDULED module as the workload.

import time
import numpy as np
import tvm
from tvm import te, s_tir
from tvm.s_tir import meta_schedule as ms


def build_scheduled_module(n: int):
    a = te.placeholder((n,), name="A", dtype="float32")
    b = te.placeholder((n,), name="B", dtype="float32")
    c = te.compute((n,), lambda i: a[i] + b[i], name="C")
    unscheduled_mod = tvm.IRModule({"main": te.create_prim_func([a, b, c])})
    sch = s_tir.Schedule(unscheduled_mod)
    block_c = sch.get_sblock("C")
    (loop_i,) = sch.get_loops(block_c)
    _outer, inner = sch.split(loop_i, factors=[6, 8])
    sch.vectorize(inner)
    return sch, unscheduled_mod


def main():
    print(f"tvm version: {tvm.__version__}")
    print(f"real meta_schedule module: {ms}")
    print(f"real MetaSchedule Database classes available: "
          f"{[c for c in ('Database', 'JSONDatabase', 'TuningRecord') if hasattr(ms.database, c)]}")
    print()

    n = 48
    sch, unscheduled_mod = build_scheduled_module(n)

    print("=" * 78)
    print("PART 1: measure File 066's own real split+vectorize schedule, "
          "for real, on this machine")
    print("=" * 78)
    target = tvm.target.Target("llvm")
    built = tvm.compile(sch.mod, target=target)
    dev = tvm.runtime.cpu(0)
    rng = np.random.default_rng(0)
    a_tvm = tvm.runtime.tensor(rng.uniform(size=n).astype("float32"), dev)
    b_tvm = tvm.runtime.tensor(rng.uniform(size=n).astype("float32"), dev)
    c_tvm = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)

    reps = 200_000
    start = time.perf_counter()
    for _ in range(reps):
        built["main"](a_tvm, b_tvm, c_tvm)
    real_secs_per_call = (time.perf_counter() - start) / reps
    print(f"real measured time per call: {real_secs_per_call * 1e6:.4f} us "
          f"(best of 1 pass, {reps} reps)")

    print()
    print("=" * 78)
    print("PART 2: commit that real measurement to a real MetaSchedule "
          "JSONDatabase -- a real TuningRecord, not a hand-rolled one")
    print("=" * 78)
    ws_path = "/tmp/ch26_ms_database_workload.json"
    rec_path = "/tmp/ch26_ms_database_tuning_record.json"
    import os
    for p in (ws_path, rec_path):
        if os.path.exists(p):
            os.remove(p)

    database = ms.database.JSONDatabase(path_workload=ws_path,
                                         path_tuning_record=rec_path)
    # The workload is the ORIGINAL, UNSCHEDULED module -- the real starting
    # point sch.trace's own real replay log (GetSBlock/GetLoops/Split/
    # Vectorize) was recorded against. See this file's own header for the
    # real error this file's first draft hit by committing sch.mod (the
    # already-scheduled module) here instead.
    workload = database.commit_workload(unscheduled_mod)
    record = ms.database.TuningRecord(
        sch.trace, workload, run_secs=[real_secs_per_call], target=target,
    )
    database.commit_tuning_record(record)
    print(f"committed a real TuningRecord (run_secs={real_secs_per_call:.8f}) "
          f"to a real JSONDatabase at {rec_path}")

    print()
    print("=" * 78)
    print("PART 3: a FRESH Database object, loaded from the exact files "
          "the first one wrote -- Ch24.1's own MISS-then-HIT shape, "
          "TVM's own real persistence format")
    print("=" * 78)
    fresh_database = ms.database.JSONDatabase(path_workload=ws_path,
                                               path_tuning_record=rec_path)
    fresh_workload = fresh_database.commit_workload(unscheduled_mod)
    records = fresh_database.get_top_k(fresh_workload, top_k=1)
    print(f"fresh Database, same workload -> {len(records)} record(s) found")
    if records:
        recovered_secs = records[0].run_secs
        print(f"recovered run_secs: {recovered_secs}")
        traces_match = str(records[0].trace) == str(sch.trace)
        print(f"recovered trace matches the original schedule's own trace: "
              f"{traces_match}")


if __name__ == "__main__":
    main()
