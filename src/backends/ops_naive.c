#include "matrix.h"
#include "ops_kernels.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// O(n^3)
Matrix mat_mul_naive(Matrix A, Matrix B) {
  if (A.columns != B.rows) {
    fprintf(stderr, "Shape mismatch in Naive MatMul: (%dx%d) vs (%dx%d)\n", A.rows, A.columns,
            B.rows, B.columns);
    exit(1);
  }

  Matrix C = mat_create(A.rows, B.columns);

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

void matvec_naive(Matrix out, Matrix W, Matrix x) {
  for (int i = 0; i < W.rows; i++) {
    float sum = 0.0f;
    for (int j = 0; j < W.columns; j++)
      sum += W.data[i * W.columns + j] * x.data[j];
    out.data[i] = sum;
  }
}

void relu_naive(Matrix *m) { APPLY_KERNEL_NAIVE(m, KERNEL_RELU); }

void relu_prime_naive(Matrix *m) { APPLY_KERNEL_NAIVE(m, KERNEL_RELU_PRIME); }

void sigmoid_naive(Matrix *m) { APPLY_KERNEL_NAIVE(m, KERNEL_SIGMOID); }

void sigmoid_prime_naive(Matrix *m) { APPLY_KERNEL_NAIVE(m, KERNEL_SIGMOID_PRIME); }

void softmax_naive(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    kernel_softmax_row(&m->data[i * m->columns], m->columns);
  }
}

void add_bias_naive(Matrix *m, Matrix *bias) {
  for (int i = 0; i < m->rows; i++) {
    kernel_add_bias(&m->data[i * m->columns], bias->data, m->columns);
  }
}