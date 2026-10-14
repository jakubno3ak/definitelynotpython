#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "include/ops.h"
#include "include/matrix.h"

#define N 1024
#define RUNS 5
// Relative tolerance: backends sum in a different order, and float rounding
// makes (a + b) + c != a + (b + c). An absolute 1e-4 is below float precision
// for values around ~264 (one float step there is ~3e-5).
#define REL_TOLERANCE 1e-5

int verify_results(Matrix C_ref, Matrix C_test, const char* name) {
    double max_rel = 0.0;
    for (int i = 0; i < C_ref.rows * C_ref.columns; i++) {
        double rel = fabs(C_ref.data[i] - C_test.data[i]) / fmax(fabs(C_ref.data[i]), 1e-6);
        if (rel > max_rel) max_rel = rel;
    }
    int ok = max_rel <= REL_TOLERANCE;
    printf("Verification %s for %s (max relative error: %.1e)\n", ok ? "PASSED" : "FAILED", name, max_rel);
    return ok;
}

// Warm-up run (thread pools, page faults), then best of RUNS.
double time_backend(Matrix A, Matrix B, Matrix *C) {
    *C = ops_mat_mul(A, B);
    double best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        double start = omp_get_wtime();
        Matrix tmp = ops_mat_mul(A, B);
        double t = omp_get_wtime() - start;
        free_matrix(tmp);
        if (t < best) best = t;
    }
    return best;
}

int main() {
    printf("=== BENCHMARK START (Matrix Size: %dx%d, best of %d runs) ===\n", N, N, RUNS);
    printf("Threads available: %d\n", omp_get_max_threads());

    Matrix A = create_matrix(N, N);
    Matrix B = create_matrix(N, N);
    
    initialize_weights(A);
    initialize_weights(B);

    // --- 1. NAIVE ---
    printf("\nRunning NAIVE backend...\n");
    ops_set_backend(BACKEND_NAIVE);
    Matrix C_naive;
    double time_naive = time_backend(A, B, &C_naive);
    printf("NAIVE Time: %.4f seconds\n", time_naive);

    // --- 2. OPENMP ---
    printf("\nRunning OPENMP backend...\n");
    ops_set_backend(BACKEND_OMP);
    Matrix C_omp;
    double time_omp = time_backend(A, B, &C_omp);
    printf("OPENMP Time: %.4f seconds\n", time_omp);
    verify_results(C_naive, C_omp, "OPENMP");

    // --- 3. BLAS ---
    printf("\nRunning BLAS backend...\n");
    ops_set_backend(BACKEND_BLAS);
    Matrix C_blas;
    double time_blas = time_backend(A, B, &C_blas);
    printf("BLAS Time: %.4f seconds\n", time_blas);
    verify_results(C_naive, C_blas, "BLAS");

    // --- SUMMARY ---
    printf("\n------------------------------------------------\n");
    printf("FINAL RESULTS SUMMARY (Reference: Naive)\n");
    printf("OPENMP Speedup: %.2fx\n", time_naive / time_omp);
    printf("BLAS   Speedup: %.2fx\n", time_naive / time_blas);
    printf("BLAS vs OPENMP: %.2fx faster\n", time_omp / time_blas);
    printf("------------------------------------------------\n");

    free_matrix(A); free_matrix(B); 
    free_matrix(C_naive); free_matrix(C_omp); free_matrix(C_blas);
    
    return 0;
}