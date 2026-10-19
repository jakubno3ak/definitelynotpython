#include "matrix.h"

#ifdef NO_BLAS
#include "ops.h"
#include <stdio.h>

Matrix mat_mul_blas(Matrix A, Matrix B) {
  fprintf(stderr, "Built without BLAS (BLAS=0). Fallback to Naive.\n");
  return mat_mul_naive(A, B);
}

void matvec_blas(Matrix out, Matrix W, Matrix x) {
  matvec_naive(out, W, x); // silent: called ~43 times per token
}
#else
#include <cblas.h>

Matrix mat_mul_blas(Matrix A, Matrix B) {
  Matrix C = mat_create(A.rows, B.columns);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A.rows, B.columns, A.columns, 1.0f, A.data,
              A.columns, B.data, B.columns, 0.0f, C.data, C.columns);

  return C;
}

void matvec_blas(Matrix out, Matrix W, Matrix x) {
  cblas_sgemv(CblasRowMajor, CblasNoTrans, W.rows, W.columns, 1.0f, W.data, W.columns, x.data, 1,
              0.0f, out.data, 1);
}
#endif
