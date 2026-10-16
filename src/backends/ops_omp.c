#include "matrix.h"
#include "ops_kernels.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

Matrix mat_mul_omp(Matrix A, Matrix B) {
  if (A.columns != B.rows) {
    fprintf(stderr, "Shape mismatch in OMP MatMul\n");
    exit(1);
  }

  Matrix C = mat_create(A.rows, B.columns);

#pragma omp parallel for
  // i-k-j order: C[i][:] += A[i][k] * B[k][:]
  // B and C are walked row by row (sequential memory) instead of B by columns
  for (int i = 0; i < A.rows; i++) {
    for (int k = 0; k < A.columns; k++) {
      float a = A.data[i * A.columns + k];
      for (int j = 0; j < B.columns; j++) {
        C.data[i * C.columns + j] += a * B.data[k * B.columns + j];
      }
    }
  }
  return C;
}

void relu_omp(Matrix *m) { APPLY_KERNEL_OMP(m, KERNEL_RELU); }

void relu_prime_omp(Matrix *m) { APPLY_KERNEL_OMP(m, KERNEL_RELU_PRIME); }

void sigmoid_omp(Matrix *m) { APPLY_KERNEL_OMP(m, KERNEL_SIGMOID); }

void sigmoid_prime_omp(Matrix *m) { APPLY_KERNEL_OMP(m, KERNEL_SIGMOID_PRIME); }

void softmax_omp(Matrix *m) {
#pragma omp parallel for
  for (int i = 0; i < m->rows; i++) {
    kernel_softmax_row(&m->data[i * m->columns], m->columns);
  }
}

void add_bias_omp(Matrix *m, Matrix *bias) {
#pragma omp parallel for
  for (int i = 0; i < m->rows; i++) {
    kernel_add_bias(&m->data[i * m->columns], bias->data, m->columns);
  }
}
