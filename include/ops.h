#ifndef OPS_H
#define OPS_H

#include "matrix.h"

typedef enum {
  BACKEND_NAIVE, // Single Core - for loops
  BACKEND_OMP,   // OpenMP (CPU Multi Core) - TODO
  BACKEND_CUDA,  // NVIDIA CUDA (GPU) - TODO
  BACKEND_BLAS   // OpenBLAS/MKL - TODO
} ComputeBackend;

void ops_set_backend(ComputeBackend backend);
char *ops_get_backend_name();

Matrix ops_mat_mul(Matrix A, Matrix B);

float sigmoid(float x);

#endif