#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include "include/ops.h"
#include "include/matrix.h"

#define N 1024
#define TOLERANCE 1e-4

int verify_results(Matrix C_ref, Matrix C_test, const char* name) {
    for (int i = 0; i < C_ref.rows * C_ref.columns; i++) {
        if (fabs(C_ref.data[i] - C_test.data[i]) > TOLERANCE) {
            printf("Verification FAILED for %s at index %d (Ref: %f, Test: %f)\n", name, i, C_ref.data[i], C_test.data[i]);
            return 0;
        }
    }
    printf("Verification PASSED for %s\n", name);
    return 1;
}

int main() {
    printf("=== BENCHMARK START (Matrix Size: %dx%d) ===\n", N, N);
    printf("Threads available: %d\n", omp_get_max_threads());

    Matrix A = create_matrix(N, N);
    Matrix B = create_matrix(N, N);
    
    initialize_weights(A);
    initialize_weights(B);

    double start, end;
    
    // --- 1. NAIVE ---
    printf("\nRunning NAIVE backend...\n");
    ops_set_backend(BACKEND_NAIVE);
    start = omp_get_wtime();
    Matrix C_naive = ops_mat_mul(A, B);
    end = omp_get_wtime();
    double time_naive = end - start;
    printf("NAIVE Time: %.4f seconds\n", time_naive);

    // --- 2. OPENMP ---
    printf("\nRunning OPENMP backend...\n");
    ops_set_backend(BACKEND_OMP);
    start = omp_get_wtime();
    Matrix C_omp = ops_mat_mul(A, B);
    end = omp_get_wtime();
    double time_omp = end - start;
    printf("OPENMP Time: %.4f seconds\n", time_omp);
    verify_results(C_naive, C_omp, "OPENMP");

    // --- 3. BLAS ---
    printf("\nRunning BLAS backend...\n");
    ops_set_backend(BACKEND_BLAS);
    start = omp_get_wtime();
    Matrix C_blas = ops_mat_mul(A, B);
    end = omp_get_wtime();
    double time_blas = end - start;
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