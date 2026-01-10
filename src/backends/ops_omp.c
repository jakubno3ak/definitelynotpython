#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../include/matrix.h"

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