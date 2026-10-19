#include "check.h"
#include "matrix.h"
#include "ops.h"
#include <stdlib.h>

static void check_close(Matrix ref, Matrix got) {
  CHECK(ref.rows == got.rows && ref.columns == got.columns);
  for (int i = 0; i < mat_size(ref); i++)
    CHECK_NEAR(got.data[i], ref.data[i], 1e-5f * fmaxf(1.0f, fabsf(ref.data[i])));
}

int main(void) {
  // The 2x2 example from note 01
  Matrix A = mat_create(2, 2), B = mat_create(2, 2);
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
    mat_free(C);
  }

  // Non-square random matrices: every backend must match naive
  srand(42);
  Matrix X = mat_create(64, 48), Y = mat_create(48, 32);
  mat_fill_random(X);
  mat_fill_random(Y);
  ops_set_backend(BACKEND_NAIVE);
  Matrix ref = ops_mat_mul(X, Y);
  for (int k = 1; k < 3; k++) {
    ops_set_backend(backends[k]);
    Matrix got = ops_mat_mul(X, Y);
    check_close(ref, got);
    mat_free(got);
  }

  // matvec: the 2x3 example, W = [1 2 3; 4 5 6], x = [1 0 2] -> [7 16]
  Matrix W = mat_create(2, 3), x = mat_create(1, 3), out = mat_create(1, 2);
  float w[] = {1, 2, 3, 4, 5, 6}, xv[] = {1, 0, 2};
  for (int i = 0; i < 6; i++)
    W.data[i] = w[i];
  for (int i = 0; i < 3; i++)
    x.data[i] = xv[i];
  ops_matvec(out, W, x);
  CHECK(out.data[0] == 7.0f && out.data[1] == 16.0f);
  ops_set_backend(BACKEND_OMP);
  ops_matvec(out, W, x);
  CHECK(out.data[0] == 7.0f && out.data[1] == 16.0f);
  ops_set_backend(BACKEND_NAIVE);
  mat_free(W);
  mat_free(x);
  mat_free(out);

  mat_free(A);
  mat_free(B);
  mat_free(X);
  mat_free(Y);
  mat_free(ref);
  return report("test_matmul");
}
