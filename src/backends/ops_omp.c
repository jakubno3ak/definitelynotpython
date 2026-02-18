#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <math.h>
#include "../../include/matrix.h"
#include "../../include/ops_kernels.h"

Matrix mat_mul_omp(Matrix A, Matrix B) {
  if (A.columns != B.rows) {
    fprintf(stderr, "Shape mismatch in OMP MatMul\n");
    exit(1);
  }

  Matrix C = create_matrix(A.rows, B.columns);

#pragma omp parallel for
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


void relu_omp(Matrix *m) {
    APPLY_KERNEL_OMP(m, KERNEL_RELU);
}

void relu_prime_omp(Matrix *m) {
    APPLY_KERNEL_OMP(m, KERNEL_RELU_PRIME);
}

void sigmoid_omp(Matrix *m) {
    APPLY_KERNEL_OMP(m, KERNEL_SIGMOID);
}

void sigmoid_prime_omp(Matrix *m) {
    APPLY_KERNEL_OMP(m, KERNEL_SIGMOID_PRIME);
}

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
