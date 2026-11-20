# CUDA Hammer: a tensor compiler in C++

Source for the [published book](https://maximumbeings.github.io/cuda_hammer_compiler_from_scratch/).

## Build and verify the documentation

```sh
python3 -m pip install -r requirements-docs.txt
python3 tools/build_diagrams.py
mkdocs build --strict
python3 tools/verify_book.py --site site
python3 tests/validation_coverage.py
python3 tools/validate_host.py --jobs 3
python3 tools/validate_host.py --jobs 3 --sanitize --log-dir validation/host-sanitized
```

The sources in `docs/part*/code/` and `docs/appendix/code/` are standalone lessons. Their listings are embedded in the corresponding Markdown pages; change both together. Host examples use C++17, with the additional compile/link flags documented in each lesson. JIT examples need a working compiler at runtime and POSIX dynamic loading.

`validation/accuracy-fixes/` records the technical review, fresh test results, dependency limitations and GPU validation scripts. Compiler acceptance, CPU-reference agreement and actual GPU execution are different forms of evidence. Historical output is identified separately from fresh measurements.

## GPU correctness checks

```sh
python3 tools/validate_gpu.py --arch sm_80 --sanitize
python3 validation/accuracy-fixes/gpu_case_validation.py
compute-sanitizer --tool memcheck python3 validation/accuracy-fixes/gpu_case_validation.py
```

The first runner needs a Linux NVIDIA GPU, CUDA toolkit and Compute Sanitizer. The Python attention/dot checks additionally need compatible PyTorch, Triton and NumPy. `sm_80` matches the tested A100; `sm_86` is appropriate for the A10; select the architecture matching your actual GPU. A skipped or unavailable device test is not a pass. Large framework case-study dependencies are not required for building the documentation.

## Publishing

After reviewing changes and passing the applicable checks:

```sh
mkdocs gh-deploy --strict
```

The diagram builder is idempotent. All illustrations share a 760-unit deep-blue canvas, accessible labels and responsive full-width embeds. They describe contracts and data flow, not fabricated performance measurements.

Validation coverage is defined by `tools/book-manifest.json`. Update it deliberately when adding or removing a lesson; validators fail on missing, extra or empty input sets. This prevents a missing checkout from being reported as a successful zero-test run.
