#ifndef OPS_KERNELS_H
#define OPS_KERNELS_H

#include "matrix.h"
#include <math.h>

#define KERNEL_RELU(x) ((x) < 0 ? 0.0f : (x))
#define KERNEL_RELU_PRIME(x) ((x) > 0 ? 1.0f : 0.0f)
#define KERNEL_SIGMOID(x) (1.0f / (1.0f + expf(-(x))))
#define KERNEL_SIGMOID_PRIME(s) ((s) * (1.0f - (s)))

#define APPLY_KERNEL_NAIVE(m, kernel_func) \
    for (int i = 0; i < get_size(*(m)); i++) { \
        (m)->data[i] = kernel_func((m)->data[i]); \
    }

#define APPLY_KERNEL_OMP(m, kernel_func) \
    _Pragma("omp parallel for") \
    for (int i = 0; i < get_size(*(m)); i++) { \
        (m)->data[i] = kernel_func((m)->data[i]); \
    }

static inline void kernel_softmax_row(float* row, int cols) {
    float max_val = -1e37f;
    for (int j = 0; j < cols; j++) if (row[j] > max_val) max_val = row[j];
    
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        row[j] = expf(row[j] - max_val);
        sum += row[j];
    }
    for (int j = 0; j < cols; j++) row[j] /= sum;
}

static inline void kernel_add_bias(float* row, float* bias, int cols) {
    for (int j = 0; j < cols; j++) {
        row[j] += bias[j];
    }
}

#endif