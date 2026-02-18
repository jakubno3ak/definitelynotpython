#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../include/matrix.h"
#include "../../include/ops_kernels.h"

// O(n^3)
Matrix mat_mul_naive(Matrix A, Matrix B) {
  if (A.columns != B.rows) {
    fprintf(stderr, "Shape mismatch in Naive MatMul: (%dx%d) vs (%dx%d)\n",
            A.rows, A.columns, B.rows, B.columns);
    exit(1);
  }

  Matrix C = create_matrix(A.rows, B.columns);

  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < B.columns; j++) {
      float sum = 0.0f;
      for (int k = 0; k < A.columns; k++) {
        sum += A.data[i * A.columns + k] * B.data[k * B.columns + j];
      }
      C.data[i * C.columns + j] = sum;
    }
  }
  return C;
}

void relu_naive(Matrix *m) {
    APPLY_KERNEL_NAIVE(m, KERNEL_RELU);
}

void relu_prime_naive(Matrix *m) {
    APPLY_KERNEL_NAIVE(m, KERNEL_RELU_PRIME);
}

void sigmoid_naive(Matrix *m) {
    APPLY_KERNEL_NAIVE(m, KERNEL_SIGMOID);
}

void sigmoid_prime_naive(Matrix *m) {
    APPLY_KERNEL_NAIVE(m, KERNEL_SIGMOID_PRIME);
}

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