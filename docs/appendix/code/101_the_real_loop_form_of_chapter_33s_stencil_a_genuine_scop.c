/* Appendix D -- the real loop CUDA Hammer's own IR has never been able
 * to express. Chapter 33.2's own honest boundary named this directly:
 * "CUDA Hammer's IR has never had a loop/control-flow construct...
 * unrolls a FIXED, compile-time-known step count directly into the
 * host driver." This file writes that same update rule --
 *
 *   w[t+1][i] = r*w[t][i-1] + (1-2r)*w[t][i] + r*w[t][i+1]
 *
 * -- with r=0.25 (Chapter 33's own real stability condition and
 * formula) -- as an ACTUAL nested for-loop, not a host-unrolled chain.
 * `stencil()` is a real static control part (SCoP): every loop bound
 * and every array subscript is an affine function of the surrounding
 * loop indices, exactly what a polyhedral compiler like Polly (part of
 * this book's own LLVM 18.1.3 toolchain since Appendix C) analyzes and
 * transforms directly.
 */
#include <stdio.h>
#define N 10
#define STEPS 3

void stencil(double w[STEPS + 1][N], double r) {
    int t, i;
    for (t = 0; t < STEPS; t++) {
        for (i = 1; i < N - 1; i++) {
            w[t + 1][i] = r * w[t][i - 1] + (1.0 - 2.0 * r) * w[t][i] + r * w[t][i + 1];
        }
        w[t + 1][0] = w[t][0];
        w[t + 1][N - 1] = w[t][N - 1];
    }
}

int main() {
    double w[STEPS + 1][N];
    int i;
    for (i = 0; i < N; i++) w[0][i] = 0.0;
    w[0][N / 2] = 100.0;

    stencil(w, 0.25);

    for (i = 0; i < N; i++) printf("%.4f ", w[STEPS][i]);
    printf("\n");
    return 0;
}
