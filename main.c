#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "include/matrix.h"

int main() {
    Matrix X = create_matrix(5, 7);
    initialize_weights(X);
    printf("Matrix X:\n");
    print_matrix(X);

    Matrix W = create_matrix(7, 10);
    initialize_weights(W);
    printf("\nMatrix W:\n");
    print_matrix(W);

    printf("\nMatMul(Y = X * W):\n");
    Matrix Y = mat_mul(X, W);
    print_matrix(Y);

    free_matrix(X);
    free_matrix(W);
    free_matrix(Y);

    return 0;
}