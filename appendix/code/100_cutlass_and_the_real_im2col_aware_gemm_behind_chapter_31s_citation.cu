// Appendix C.4 capstone -- Chapter 31.2 cited NVIDIA CUTLASS's own docs
// for the real im2col-based implicit-GEMM technique behind convolution
// libraries ("constructs the convolution matrix explicitly via... im2col").
// This file goes one level deeper: it compiles a real CUTLASS GEMM
// template instantiation (CUTLASS 4.8.0, commit 0b55a2f, the classic
// SIMT cutlass::gemm::device::Gemm API) against CUTLASS's own real
// headers, and names the exact real header where CUTLASS's modern
// hardware (Hopper/Blackwell TMA) copy engine has an explicit,
// filename-level im2col concept of its own:
//   include/cute/atom/copy_traits_sm90_im2col.hpp
//   include/cute/atom/copy_traits_sm100_im2col.hpp
// -- direct, current evidence that im2col is still how real production
// GEMM/convolution infrastructure names this idea, not a detail specific
// to this book's own Chapter 31.2 implementation.
//
// No physical GPU is available in this environment (same honest
// limitation as every CUDA-linked file since Chapter 18), so this file
// compiles and instantiates the real CUTLASS kernel template but only
// reports the real cudaGetDeviceCount() result rather than claiming
// a run.
#include <cstdio>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"

using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80
>;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount: err=%s, deviceCount=%d\n",
           cudaGetErrorString(err), deviceCount);

    const int M = 4, N = 4, K = 4;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    if (deviceCount == 0) {
        printf("No CUDA device detected in this environment -- compiling and\n");
        printf("instantiating the real CUTLASS Gemm template above (this is the\n");
        printf("real thing CUTLASS's build would have checked at kernel-launch\n");
        printf("time) is as far as this file honestly goes, the same real\n");
        printf("compile-but-no-claimed-run split as Chapter 18's CUDA backend\n");
        printf("and Appendix A.3's vector-add smoke test.\n");
        return 0;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, M * N * sizeof(float)));

    CutlassGemm::TensorRefA refA(dA, K);
    CutlassGemm::TensorRefB refB(dB, N);
    CutlassGemm::TensorRefC refC(dC, N);
    CutlassGemm::TensorRefD refD(dC, N);

    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        problem_size,
        refA, refB, refC, refD,
        {1.0f, 0.0f}
    );

    cutlass::Status status = gemm_op.can_implement(args);
    printf("can_implement: %s\n", cutlass::cutlassGetStatusString(status));

    status = gemm_op.initialize(args);
    printf("initialize: %s\n", cutlass::cutlassGetStatusString(status));

    status = gemm_op();
    printf("run: %s\n", cutlass::cutlassGetStatusString(status));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
