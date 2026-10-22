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
  for (int k = 1; k < 3; k++) {
    ops_set_backend(backends[k]);
    ops_matvec(out, W, x);
    CHECK(out.data[0] == 7.0f && out.data[1] == 16.0f);
  }

  // matvec on a random 64x48 matrix: OpenMP and BLAS must match naive
  Matrix v = mat_create(1, 48), ref_v = mat_create(1, 64), got_v = mat_create(1, 64);
  mat_fill_random(v);
  ops_set_backend(BACKEND_NAIVE);
  ops_matvec(ref_v, X, v);
  for (int k = 1; k < 3; k++) {
    ops_set_backend(backends[k]);
    ops_matvec(got_v, X, v);
    check_close(ref_v, got_v);
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

  // linear on 5 random tokens: naive row t == matvec of token t, OpenMP and BLAS == naive
  Matrix T = mat_create(5, 48), ref_t = mat_create(5, 64), got_t = mat_create(5, 64);
  mat_fill_random(T);
  ops_set_backend(BACKEND_NAIVE);
  ops_linear(ref_t, T, X);
  for (int t = 0; t < 5; t++) {
    ops_matvec(ref_v, X, mat_view(T.data + t * 48, 1, 48));
    check_close(ref_v, mat_view(ref_t.data + t * 64, 1, 64));
  }
  for (int k = 1; k < 3; k++) {
    ops_set_backend(backends[k]);
    ops_linear(got_t, T, X);
    check_close(ref_t, got_t);
  }
  mat_free(Xl);
  mat_free(Wl);
  mat_free(Yl);
  mat_free(T);
  mat_free(ref_t);
  mat_free(got_t);
  ops_set_backend(BACKEND_NAIVE);
  mat_free(v);
  mat_free(ref_v);
  mat_free(got_v);
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
