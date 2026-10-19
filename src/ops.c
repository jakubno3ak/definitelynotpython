#include "ops.h"
#include "ops_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

  case BACKEND_BLAS:
    return mat_mul_blas(A, B);

  case BACKEND_CUDA:
    fprintf(stderr, "CUDA not implemented yet. Fallback to Naive.\n");
    return mat_mul_naive(A, B);

  case BACKEND_NAIVE:
  default:
    return mat_mul_naive(A, B);
  }
}

void ops_matvec(Matrix out, Matrix W, Matrix x) {
  if (W.columns != mat_size(x) || W.rows != mat_size(out)) {
    fprintf(stderr, "Shape mismatch in matvec: W (%dx%d), x %d, out %d\n", W.rows, W.columns,
            mat_size(x), mat_size(out));
    exit(1);
  }
  switch (CURRENT_BACKEND) {
  case BACKEND_OMP:
    matvec_omp(out, W, x);
    break;
  default: // BLAS and CUDA matvec not implemented yet
    matvec_naive(out, W, x);
  }
}

void ops_add_bias(Matrix *m, Matrix *bias) {
  if (CURRENT_BACKEND == BACKEND_NAIVE) {
    add_bias_naive(m, bias);
  } else {
    // For OMP and BLAS (since BLAS doesn't have a simple "add bias" for matrices)
    add_bias_omp(m, bias);
  }
}

void ops_relu(Matrix *m) {
  if (CURRENT_BACKEND == BACKEND_NAIVE)
    relu_naive(m);
  else
    relu_omp(m);
}

void ops_relu_prime(Matrix *m) {
  if (CURRENT_BACKEND == BACKEND_NAIVE)
    relu_prime_naive(m);
  else
    relu_prime_omp(m);
}

void ops_sigmoid(Matrix *m) {
  if (CURRENT_BACKEND == BACKEND_NAIVE)
    sigmoid_naive(m);
  else
    sigmoid_omp(m);
}

void ops_sigmoid_prime(Matrix *m) {
  if (CURRENT_BACKEND == BACKEND_NAIVE)
    sigmoid_prime_naive(m);
  else
    sigmoid_prime_omp(m);
}

void ops_softmax(Matrix *m) {
  if (CURRENT_BACKEND == BACKEND_NAIVE)
    softmax_naive(m);
  else
    softmax_omp(m);
}
