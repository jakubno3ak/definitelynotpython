#include "../../include/matrix.h"
#include <stdio.h>
#include <stdlib.h>

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