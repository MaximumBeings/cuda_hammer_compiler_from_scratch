/* Appendix D.3 -- a second, independent real SCoP, chosen to contrast
 * directly with File 101's stencil: every iteration of this loop reads
 * and writes completely independent elements, with no loop-carried
 * dependence at all. Used to show what Polly's real dependence analysis
 * concludes is SAFE to parallelize, laid next to File 101's stencil
 * (reused unchanged), where the same real analysis correctly finds the
 * opposite answer for the outer, time-carried loop.
 */
#define N 100

void axpy(double a[N], double b[N], double s) {
    int i;
    for (i = 0; i < N; i++) {
        a[i] = a[i] + s * b[i];
    }
}
