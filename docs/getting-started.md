# Getting Started

This book builds CUDA Hammer entirely in C++17, with three real, distinct backends by the time Part 4 is done: a plain host C++ interpreter for CUDA Hammer's own IR (useful for testing a pass before trusting its output), a vectorized CPU code generator (real AVX2/FMA or NEON, whichever the machine actually has), and a CUDA C++ code generator that emits real kernels for `nvcc`. No external compiler framework -- no LLVM, no MLIR -- is used to build CUDA Hammer itself; Appendix C covers what a real production compiler gets by building on one of those instead, once CUDA Hammer's own from-scratch version exists to compare it against.

## 1 — Choose the machine and scope

The host lessons run on Linux or macOS with a C++17 compiler. Use **Linux with a supported NVIDIA GPU** for the CUDA lessons and Triton device examples. Apple Silicon runs the NEON CPU backend; current CUDA toolkits do not support macOS. Windows readers should use a supported Linux environment; the JIT examples rely on POSIX process handling and `dlopen`.

| Work | Required | Optional |
| --- | --- | --- |
| Read/build this website | Python 3 and `requirements-docs.txt` | No CUDA or compiler needed |
| Standalone host lessons | C++17 compiler; POSIX host | Graphviz for viewing DOT output |
| SIMD CPU generation | AVX2 **and** FMA on x86, or NEON on AArch64 | No NVIDIA driver needed |
| Generated CUDA programs | Linux, NVIDIA driver, CUDA toolkit, compatible GPU | Compute Sanitizer for diagnostic runs |
| Framework case studies | Separate Python environment and the chapter's compatible framework versions | LLVM/MLIR, Polly and CUTLASS only for their appendix comparisons |

## 2 — Install host tools and obtain the book

On Ubuntu 22.04/24.04:

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang git python3 python3-venv python3-pip graphviz
```

On macOS, install Apple's command-line tools and complete their installer:

```bash
xcode-select --install
```

Use an installed Python 3 interpreter (for example from python.org). Apple `g++` is normally a Clang frontend; check `clang++ --version`. Both compiler families can build the host lessons. On Linux, `lscpu` reports CPU features; the x86 SIMD lessons need both `avx2` and `fma`, not just an x86 architecture label. Apple Silicon uses the ARM path.

```bash
git clone https://github.com/MaximumBeings/cuda_hammer_compiler_from_scratch.git
cd cuda_hammer_compiler_from_scratch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs build --strict
python tools/verify_book.py --site site
python tools/verify_site_links.py --site site
```

Use `python -m pip` and `python -m mkdocs` from the same environment. Website packages are distinct from the optional numerical frameworks. To preview, run `python -m mkdocs serve` and open the local address it prints.

## 3 — Compile your first host lesson

From the repository root:

```bash
clang++ -std=c++17 -Wall -Wextra -O2 \
  docs/part0/code/001_eager_vs_fused_memory_traffic.cpp -o /tmp/hammer-first
/tmp/hammer-first
python tools/validate_host.py --jobs 3 --log-dir /tmp/hammer-host-check
```

Compare the lesson's calculated array traffic with its displayed explanation. A compiler error stops this step; do not continue to later backends until this works. JIT lessons also execute `g++` at runtime, so that command must be on `PATH` even if their driver was built with Clang. On Linux their compile recipes additionally link `-ldl`; the host runner supplies that flag automatically.

## 4 — Install the NVIDIA driver and CUDA toolkit on Linux

This concrete recipe targets **Ubuntu 22.04/24.04, x86_64** and the book's **CUDA 12.8** baseline. Other architectures/distributions need their own packages. First run `nvidia-smi` and `nvcc --version`: a provisioned GPU image may already contain both. Keep a working provider driver. `nvidia-smi`'s CUDA version describes driver capability; it does not establish that `nvcc` is installed.

For a fresh supported Ubuntu machine, enable NVIDIA's repository:

```bash
. /etc/os-release
case "$VERSION_ID" in
  22.04) hammer_distro=ubuntu2204 ;;
  24.04) hammer_distro=ubuntu2404 ;;
  *) echo 'Use the NVIDIA guide for this Ubuntu version'; exit 1 ;;
esac
test "$(uname -m)" = x86_64 || exit 1
sudo apt-get install -y wget ca-certificates
wget "https://developer.download.nvidia.com/compute/cuda/repos/${hammer_distro}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/hammer-cuda-keyring.deb
sudo dpkg -i /tmp/hammer-cuda-keyring.deb
sudo apt-get update
```

**Only if a driver is missing**, install the supported driver for the GPU. For the A100/H100/B200 families, NVIDIA's open-module driver packages support these architectures; older GPUs require a different choice. The following driver installation needs matching kernel headers and a reboot. Complete Secure Boot module enrollment if prompted; on a managed image use its provider's driver procedure.

```bash
sudo apt-get install -y linux-headers-"$(uname -r)"
sudo apt-get install -y nvidia-open
sudo reboot
```

Reconnect and confirm `nvidia-smi` works before installing the toolkit:

```bash
sudo apt-get install -y cuda-toolkit-12-8
export PATH=/usr/local/cuda-12.8/bin:"$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
nvcc --version
compute-sanitizer --version
nvidia-smi
```

Save the two exports in your shell startup file for later sessions. The versioned toolkit package does not replace the driver. CUDA 12.8 GA lists Linux driver 570.26 or newer as its corresponding driver baseline; the recorded A100 used 580.105.08. This setup does not rely on older-driver minor-version compatibility. A PyTorch CUDA wheel alone does not supply the standalone `nvcc` toolchain required by the book's generators.

Sources, checked November 20, 2026: [CUDA 12.8 installation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/), [Ubuntu driver installation](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/ubuntu.html), [CUDA 12.8 driver table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html), [macOS CUDA support boundary](https://docs.nvidia.com/cuda/archive/10.2/cuda-toolkit-release-notes/index.html).

## 5 — Select the GPU target and run the generated programs

| GPU | Architecture argument |
| --- | --- |
| A100 | `sm_80` |
| A10 | `sm_86` |
| H100/H200 | `sm_90` |
| B200 | `sm_100` (CUDA 12.8 or newer) |

These are architecture selections, not claims that this revision was executed on every device. See [Validation Status](validation-status.md) for actual evidence. The CUDA drivers generate and compile their own child programs; compiling only the driver is insufficient.

```bash
python tools/validate_gpu.py --arch sm_80 --sanitize --log-dir /tmp/hammer-gpu-check
```

Use a **new empty output directory** for each run. The three drivers use fixed `/tmp/hammer_ch18_*` child paths, so run one GPU validation process at a time on a host. No-device/skipped executions and missing sanitizers are failures of this acceptance run, not device passes. Compiling CUDA for a supported target does not require that target GPU; executing the result does.

## 6 — Add optional case-study dependencies separately

The compiler itself does not depend on PyTorch, JAX, TVM, Triton, LLVM or CUTLASS. Install only the dependencies for the chapter you are studying, in a separate environment. The earlier Linux/A100 record used Python 3.12.3, NumPy 1.26.4, PyTorch 2.7.0, Triton 3.3.0 and JAX 0.6.0. TVM was not installed in that validation. Those recorded versions are not a claim that every historical framework transcript uses that same API.

For the local NumPy-based acceptance/calibration regressions:

```bash
python -m pip install numpy==1.26.4
python tests/cases_regressions.py
```

For actual framework execution, follow the chapter's explicit version and the project's installation guide: [PyTorch](https://pytorch.org/get-started/locally/), [JAX](https://docs.jax.dev/en/latest/installation.html), [Triton](https://triton-lang.org/main/getting-started/installation.html), [TVM](https://tvm.apache.org/docs/install/index.html). TVM examples using `s_tir` require that namespace; a wheel exposing only older `tir` APIs is not interchangeable. Do not infer execution of these stacks from passing the NumPy/AST adapter tests.

After installing compatible GPU framework packages, the additional numerical checks are:

```bash
python validation/accuracy-fixes/gpu_case_validation.py
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python validation/accuracy-fixes/gpu_case_validation.py
```

## 7 — Troubleshoot the first failing step

| Symptom | Check before proceeding |
| --- | --- |
| `g++`/`clang++` missing | Install host tools; JIT subprocesses require `g++` on `PATH`. |
| `Illegal instruction` | Check AVX2 **and** FMA availability; a compiler accepting intrinsics does not prove CPU support. |
| `nvidia-smi` fails | Driver/kernel/module or GPU access is unresolved; installing Python libraries will not repair it. |
| `nvcc` missing | Install/select the toolkit and restore its `bin` directory on `PATH`. |
| Unsupported GPU architecture | Select the actual target and a toolkit that supports it; `sm_100` requires CUDA 12.8+. |
| Shared CUDA library missing | Check the selected toolkit's `lib64` and dynamic-loader configuration. |
| Sanitizer missing or denied | Check `compute-sanitizer --version` and GPU access; do not classify an unexecuted diagnostic as clean. |
| Framework import/API failure | Check that environment's versions against the chapter, rather than mixing installations. |

## Historical authoring context

The original cloud/device transcripts below predate the later A100 baseline. They explain the provenance of the historical outputs, not the current state of every example. The latest CUDA-generator corrections still need a fresh device run; the current host sanitizer checks do not replace it.

## The original edition’s validation discipline

This book's own two authoring machines are, between them, real and different: a cloud sandbox (x86_64, real AVX2/AVX512F/FMA, real `g++`, real `clang++`, and a real, complete `nvcc` toolchain) and a connected device (aarch64, real NEON, real `g++`, **no** `nvcc`, **no** physical GPU on either machine). That split is treated as a feature, not a limitation to route around -- it means CUDA Hammer's CPU backend gets to be genuinely, fully honest in a way its GPU backend structurally cannot:

- **CUDA Hammer's IR, its pass manager, and its host-side interpreter** (no vector intrinsics, no CUDA) are plain C++. They are genuinely compiled and genuinely run wherever they're touched, with exact output locked into the page and re-verified by a fresh recompile-and-rerun before publication.
- **CUDA Hammer's vectorized CPU backend** (Chapter 19 onward) is genuinely compiled *and genuinely run*, for real, on two real, different architectures: AVX2/FMA intrinsics on the cloud sandbox, NEON intrinsics on the connected device. A claim about what fused, vectorized code actually computes is never simulated in this book -- it is measured directly, compared with a scalar reference under the stated numerical tolerance, on real hardware, on two real instruction sets.
- **CUDA Hammer's CUDA backend** (Chapter 18 onward) is genuinely compiled with `nvcc` for a real architecture (`-arch=sm_80` unless a chapter says otherwise) and checked instead by a **host-side reference implementation**: ordinary C++ that computes the exact same result the generated kernel's own loop nest would, checked independently. This is the one honest limitation this book carries, stated once here rather than repeated on every page: neither authoring machine has a physical NVIDIA GPU, so the historical checks establish compiler acceptance and host-reference behavior, not device execution. A CPU reference is an expected-result oracle; actual GPU output, memory safety and race behavior require device tests. Current validation results are recorded separately on the Validation Status page.
- **Autotuning search and cost models** (Part 5) are run for real against whichever backend a chapter is tuning -- the CPU backend's genuine on-hardware measurements are real, measured numbers; a CUDA-kernel cost model is a real, stated formula rather than a fabricated measured number, exactly like the sibling books' own standing rule against inventing timing data.
- **Real facts this book states about other compilers** -- an XLA fusion rule, a TVM scheduling primitive, a Triton language feature -- are cited to that project's own real, current documentation or source, not asserted from memory.

Every chapter states which of the above applies to each piece of its own code, so nothing is left for a reader to guess about how a claim was actually established.

## Compile-line conventions

- Plain C++ host files (`.cpp`): `g++ -std=c++17 -Wall -Wextra -O2 file.cpp -o binary`
- AVX2/FMA CPU-codegen files: `g++ -std=c++17 -O3 -mavx2 -mfma file.cpp -o binary` (x86_64 only)
- NEON CPU-codegen files: `g++ -std=c++17 -O3 file.cpp -o binary` (NEON is baseline on AArch64, no extra flag needed)
- CUDA-generated files (`.cu`): `nvcc -arch=sm_80 file.cu -o binary`
- A JIT-backend file that compiles and loads generated code at runtime shows its exact `dlopen`/linking recipe inline where it's introduced.

## Prerequisites

This book assumes working knowledge of C++ (classes, templates, smart pointers, RAII) and enough familiarity with how a tensor operation like matrix multiplication or a convolution is actually computed to follow why a particular fusion or tiling choice helps. No prior compiler-construction experience is assumed -- Part 0 builds the general vocabulary (IR, pass, lowering, codegen) from nothing before Part 1 ever starts building CUDA Hammer itself. No prior CUDA experience is assumed either; Chapter 18 introduces exactly the subset of the CUDA programming model CUDA Hammer's own generated kernels need, in place.
