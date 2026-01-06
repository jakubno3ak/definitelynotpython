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


    Matrix A = create_matrix(2, 3);
    A.data[0] = 1; A.data[1] = 2; A.data[2] = 3;
    A.data[3] = 4; A.data[4] = 5; A.data[5] = 6;

    printf("A (2x3):\n");
    print_matrix(A);

    Matrix T = mat_transpose(A);
    printf("\nTransposed A (3x2):\n");
    print_matrix(T);

    mat_scale(T, 10.0);
    printf("\nScaled Transposed A (*10):\n");
    print_matrix(T);

    free_matrix(A);
    free_matrix(T);

    return 0;
}