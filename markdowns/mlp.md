# Neural Layers (Manual MLP) Implementation

Advanced Multi-Layer Perceptron architecture with manual backpropagation.

```c
#include <stdio.h>
#include "include/matrix.h"
#include "include/ops.h"
#include "include/layer.h"

int main() {
    // 1. Setup Architecture
    Layer l1 = create_layer(8, 16, ACT_RELU);
    Layer l2 = create_layer(16, 1, ACT_SIGMOID);

    // 2. Training Loop (Stochastic Gradient Descent)
    for(int i = 0; i < 5000; i++) {
        // Forward
        Matrix h1 = layer_forward(&l1, X_train);
        Matrix out = layer_forward(&l2, h1);

        // Loss Gradient
        Matrix grad = mat_sub(out, y_train);

        // Backward
        Matrix g_h1 = layer_backward(&l2, grad, 0.01);
        Matrix g_in = layer_backward(&l1, g_h1, 0.01);

        // Memory Management
        free_matrix(h1); free_matrix(out);
        free_matrix(grad); free_matrix(g_h1); free_matrix(g_in);
    }

    layer_free(&l1);
    layer_free(&l2);
    return 0;
}