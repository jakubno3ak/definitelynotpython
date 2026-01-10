#include "../include/ops.h"
#include <stdio.h>

Matrix mat_mul_naive(Matrix A, Matrix B);
Matrix mat_mul_omp(Matrix A, Matrix B);

static ComputeBackend CURRENT_BACKEND = BACKEND_NAIVE;

void ops_set_backend(ComputeBackend backend) {
  CURRENT_BACKEND = backend;
  printf("--- Backend switched to: %s ---\n", ops_get_backend_name());
}

char *ops_get_backend_name() {
  switch (CURRENT_BACKEND) {
  case BACKEND_NAIVE:
    return "Naive (Single Thread)";
  case BACKEND_OMP:
    return "OpenMP (Multi Thread)";
  case BACKEND_CUDA:
    return "CUDA (GPU)";
  case BACKEND_BLAS:
    return "BLAS (Optimized)";
  default:
    return "Unknown";
  }
}

Matrix ops_mat_mul(Matrix A, Matrix B) {
  switch (CURRENT_BACKEND) {
  case BACKEND_OMP:
    return mat_mul_omp(A, B);

  case BACKEND_CUDA:
    fprintf(stderr, "CUDA not implemented yet. Fallback to Naive.\n");
    return mat_mul_naive(A, B);

  case BACKEND_NAIVE:
  default:
    return mat_mul_naive(A, B);
  }
}
