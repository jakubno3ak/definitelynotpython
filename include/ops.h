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

// mat mul
Matrix mat_mul_naive(Matrix A, Matrix B);
Matrix mat_mul_omp(Matrix A, Matrix B);
Matrix mat_mul_blas(Matrix A, Matrix B);
Matrix ops_mat_mul(Matrix A, Matrix B);

// add bias
void add_bias_naive(Matrix *m, Matrix *bias);
void add_bias_omp(Matrix *m, Matrix *bias);
void ops_add_bias(Matrix *m, Matrix *bias);

// activations & derivates
void relu_naive(Matrix *m);
void relu_omp(Matrix *m);
void relu_prime_naive(Matrix *m);
void relu_prime_omp(Matrix *m);
void ops_relu(Matrix *m);
void ops_relu_prime(Matrix *m);

void sigmoid_naive(Matrix *m);
void sigmoid_omp(Matrix *m);
void sigmoid_prime_naive(Matrix *m);
void sigmoid_prime_omp(Matrix *m);
void ops_sigmoid(Matrix *m);
void ops_sigmoid_prime(Matrix *m);

void softmax_naive(Matrix *m);
void softmax_omp(Matrix *m);
void ops_softmax(Matrix *m);

#endif