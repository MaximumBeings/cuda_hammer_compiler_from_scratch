// Appendix A: Installation and Setup -- Building CUDA Hammer's Toolchain
// 095_verifying_an_optional_cuda_toolkit_install.cu
//
// Section A.3 -- Chapter 18 onward needs a real CUDA toolkit (`nvcc`)
// to compile CUDA Hammer's own generated .cu programs, but never needs
// a physical GPU: getting-started.md already states this book's own
// honest limitation directly ("neither authoring machine has a
// physical NVIDIA GPU... a generated CUDA kernel's compiled
// correctness is verified for real, but its runtime result is
// verified by [a] host-side reference implementation, not by
// executing the kernel itself"). This section is a minimal, standalone
// smoke test a reader can run BEFORE Chapter 18 to confirm their own
// `nvcc` install works at all, using the exact same honest split:
// real compile, no claimed run.
//
// CUDA-linked (.cu) -- per this book's own standing cross-machine
// rule, verified on the cloud sandbox ONLY (the device has no nvcc,
// confirmed directly in TOC.md's own toolchain notes).
//
// Compile: nvcc -std=c++17 -arch=sm_80 095_verifying_an_optional_cuda_toolkit_install.cu -o 095_driver
// Run:     ./095_driver   (only prints the compile-time report below --
//                          the kernel itself is launched, per CUDA's
//                          own async launch model, but this program
//                          never claims its numeric result is real,
//                          since no physical GPU exists here to run it on)

#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) { \
    printf("CUDA_CHECK failed: %s (%s:%d): %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_e)); } } while (0)

// A trivial real elementwise add kernel -- not from CUDA Hammer's own
// codegen (Chapter 18 generates this shape directly from the IR), just
// enough real CUDA C++ to exercise a real compile end-to-end.
__global__ void vectorAddKernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

int main() {
    printf("=== Appendix A.3: verifying an optional CUDA toolkit install ===\n\n");

    int deviceCount = 0;
    cudaError_t countErr = cudaGetDeviceCount(&deviceCount);
    printf("real cudaGetDeviceCount(): %s, deviceCount=%d\n",
           cudaGetErrorString(countErr), deviceCount);
    printf("(this book's own two authoring machines both report 0 here -- no physical\n");
    printf("GPU on either side, stated directly in getting-started.md; this is expected,\n");
    printf("not an install failure)\n\n");

    printf("real nvcc compile of this file already succeeded -- reaching this printf at\n");
    printf("all confirms a working CUDA toolkit: a real .cu file with a real __global__\n");
    printf("kernel, real host<->device pointer types, and a real CUDA_CHECK macro all\n");
    printf("compiled through nvcc's real two-stage (device+host) pipeline.\n\n");

    if (deviceCount > 0) {
        printf("a real physical GPU WAS detected on this machine -- launching the real\n");
        printf("kernel and checking its real output:\n\n");
        const int n = 8;
        float hA[n], hB[n], hOut[n];
        for (int i = 0; i < n; ++i) { hA[i] = static_cast<float>(i); hB[i] = static_cast<float>(i) * 2.0f; }
        float *dA, *dB, *dOut;
        CUDA_CHECK(cudaMalloc(&dA, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice));
        vectorAddKernel<<<1, n>>>(dA, dB, dOut, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hOut, dOut, n * sizeof(float), cudaMemcpyDeviceToHost));
        bool allOk = true;
        for (int i = 0; i < n; ++i) if (hOut[i] != hA[i] + hB[i]) allOk = false;
        printf("real kernel output matches host reference exactly: %s\n", allOk ? "yes" : "NO -- MISMATCH");
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dOut));
    } else {
        printf("no physical GPU on this machine (deviceCount=0) -- the same honest\n");
        printf("limitation this book's own CUDA sections (Chapter 18 onward) have carried\n");
        printf("since Chapter 18: this file's own compiled correctness is confirmed (it\n");
        printf("built), its runtime result is not claimed here, the same split every\n");
        printf("Chapter 18+ .cu file in this book already follows.\n");
    }

    return 0;
}
