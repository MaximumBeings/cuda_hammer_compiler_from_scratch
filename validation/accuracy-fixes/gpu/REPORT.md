# Chapter18 GPU validation

Hardware: NVIDIA A100-PCIE-40GB,40960MiB, compute capability8.0; driver580.105.08. Linuxx86_64, CUDA12.8.93, ComputeSanitizer2025.1.0.0. Targetsm80 used for both outer generator drivers and generated CUDA programs.

Final command: `python3 tools/validate_gpu.py --arch sm_80 --sanitize` — PASS.

-042 elementwise output:22,66,132,220,90,182,296,432,190,330,492,676; matches interpreter.
-043 atomicAdd reduction:11; matches interpreter.
-044 uncapped graph:y=132; matches interpreter.
-044 capped graph:y=132; matches interpreter.

Each of the four generated executables ran under memcheck/racecheck/synccheck directly:12processes, all recorded exit0 and zero errors/hazards. Three generator driver builds and numeric runs also recorded exit0.

Wrapper tracing through popen was not instrumented on this installation; those reports are explicitly superseded in superseded-wrapper-attempt, not passing evidence. Direct executable logs named hammer_ch18_* are the passing sanitizer evidence. macOS AppleDouble metadata was excluded from source discovery. Compiler output decoding tolerates diagnostic bytes; logs retain process exit status. No kernel numerical bug was found in these fixtures.

Limits: concrete fixtures only; no shape-general proof, performance claim or A10 test. No instance work remains; all text logs and metadata were downloaded locally. Binary build artifacts were deliberately not copied into the repository.
