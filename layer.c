#include <stdio.h>
#include <stdlib.h>
#include "include/matrix.h"
#include "include/ops.h"
#include "include/layer.h"

void fill_matrix(Matrix *m, float value) {
    for (int i = 0; i < m->rows * m->columns; i++) {
        m->data[i] = value;
    }
}

int main() {
    printf("--- Layer Forward ---\n");

    Matrix X = create_matrix(2, 2);
    X.data[0] = 0.0f; X.data[1] = 0.0f;
    X.data[2] = 1.0f; X.data[3] = 1.0f;

    printf("Input X:\n");
    print_matrix(X);

    Layer l = create_layer(2, 1, ACT_SIGMOID);

    fill_matrix(&l.weights, 0.5f);
    fill_matrix(&l.biases, 1.0f);

    Matrix Output = layer_forward(&l, X);

    printf("\nOutput:\n");
    print_matrix(Output);

    free_matrix(X);
    free_matrix(Output);
    free_matrix(l.weights);
    free_matrix(l.biases);
    
    return 0;
}