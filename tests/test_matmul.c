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

  // linear (X · Wᵀ): 2 tokens x 3 neurons by hand, in every backend
  Matrix Xl = mat_create(2, 4), Wl = mat_create(3, 4), Yl = mat_create(2, 3);
  float xl[] = {1, 0, 2, 1, 0, 1, 1, 0}, wl[] = {1, 2, 3, 0, 0, 1, 0, 1, 2, 0, 1, 1};
  float yl[] = {7, 1, 5, 5, 1, 1};
  for (int i = 0; i < 8; i++)
    Xl.data[i] = xl[i];
  for (int i = 0; i < 12; i++)
    Wl.data[i] = wl[i];
  for (int k = 0; k < 3; k++) {
    ops_set_backend(backends[k]);
    ops_linear(Yl, Xl, Wl);
    for (int i = 0; i < 6; i++)
      CHECK(Yl.data[i] == yl[i]);
  }

  // linear on 5 random tokens: each row == that token alone, OpenMP and BLAS == naive
  Matrix T = mat_create(5, 48), ref_t = mat_create(5, 64), got_t = mat_create(5, 64);
  Matrix ref_one = mat_create(1, 64);
  mat_fill_random(T);
  ops_set_backend(BACKEND_NAIVE);
  ops_linear(ref_t, T, X);
  for (int t = 0; t < 5; t++) {
    ops_linear(ref_one, mat_view(T.data + t * 48, 1, 48), X);
    check_close(ref_one, mat_view(ref_t.data + t * 64, 1, 64));
  }
  for (int k = 1; k < 3; k++) {
    ops_set_backend(backends[k]);
    ops_linear(got_t, T, X);
    check_close(ref_t, got_t);
  }

  // linear on 1 token (decode path, BLAS uses sgemv): OpenMP and BLAS == naive
  Matrix one = mat_view(T.data, 1, 48), got_one = mat_view(got_t.data, 1, 64);
  ops_set_backend(BACKEND_NAIVE);
  ops_linear(ref_one, one, X);
  for (int k = 1; k < 3; k++) {
    ops_set_backend(backends[k]);
    ops_linear(got_one, one, X);
    check_close(ref_one, got_one);
  }
  ops_set_backend(BACKEND_NAIVE);
  mat_free(Xl);
  mat_free(Wl);
  mat_free(Yl);
  mat_free(T);
  mat_free(ref_t);
  mat_free(got_t);
  mat_free(ref_one);

  mat_free(A);
  mat_free(B);
  mat_free(X);
  mat_free(Y);
  mat_free(ref);
  return report("test_matmul");
}
