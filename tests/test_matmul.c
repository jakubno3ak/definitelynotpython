#include "check.h"
#include "matrix.h"
#include "ops.h"
#include <stdlib.h>

static void check_close(Matrix ref, Matrix got) {
  CHECK(ref.rows == got.rows && ref.columns == got.columns);
  for (int i = 0; i < get_size(ref); i++)
    CHECK_NEAR(got.data[i], ref.data[i], 1e-5f * fmaxf(1.0f, fabsf(ref.data[i])));
}

int main(void) {
  // The 2x2 example from note 01
  Matrix A = create_matrix(2, 2), B = create_matrix(2, 2);
  float a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8}, expected[] = {19, 22, 43, 50};
  for (int i = 0; i < 4; i++) {
    A.data[i] = a[i];
    B.data[i] = b[i];
  }
  ComputeBackend backends[] = {BACKEND_NAIVE, BACKEND_OMP, BACKEND_BLAS};
  for (int k = 0; k < 3; k++) {
    ops_set_backend(backends[k]);
    Matrix C = ops_mat_mul(A, B);
    for (int i = 0; i < 4; i++)
      CHECK(C.data[i] == expected[i]);
    free_matrix(C);
  }

  // Non-square random matrices: every backend must match naive
  srand(42);
  Matrix X = create_matrix(64, 48), Y = create_matrix(48, 32);
  initialize_weights(X);
  initialize_weights(Y);
  ops_set_backend(BACKEND_NAIVE);
  Matrix ref = ops_mat_mul(X, Y);
  for (int k = 1; k < 3; k++) {
    ops_set_backend(backends[k]);
    Matrix got = ops_mat_mul(X, Y);
    check_close(ref, got);
    free_matrix(got);
  }

  free_matrix(A);
  free_matrix(B);
  free_matrix(X);
  free_matrix(Y);
  free_matrix(ref);
  return report("test_matmul");
}
