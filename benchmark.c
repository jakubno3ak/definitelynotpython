#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "include/ops.h"
#include "include/matrix.h"

#define N 1024

int main() {
    printf("=== BENCHMARK START (Matrix Size: %dx%d) ===\n", N, N);
    printf("Threads available: %d\n", omp_get_max_threads());

    Matrix A = create_matrix(N, N);
    Matrix B = create_matrix(N, N);
    
    initialize_weights(A);
    initialize_weights(B);

    double start, end;
    
    printf("\nRunning NAIVE backend...\n");
    ops_set_backend(BACKEND_NAIVE);
    
    start = omp_get_wtime();
    Matrix C1 = ops_mat_mul(A, B);
    end = omp_get_wtime();
    
    double time_naive = end - start;
    printf("NAIVE Time: %.4f seconds\n", time_naive);

    printf("\nRunning OPENMP backend...\n");
    ops_set_backend(BACKEND_OMP);
    
    start = omp_get_wtime();
    Matrix C2 = ops_mat_mul(A, B);
    end = omp_get_wtime();
    
    double time_omp = end - start;
    printf("OPENMP Time: %.4f seconds\n", time_omp);

    printf("\n------------------------------------------------\n");
    printf("Speedup: %.2fx faster\n", time_naive / time_omp);
    printf("------------------------------------------------\n");

    free_matrix(A); free_matrix(B); free_matrix(C1); free_matrix(C2);
    return 0;
}