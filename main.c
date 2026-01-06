#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "include/matrix.h"
#include "include/linear_regression.h"

int main() {
    // Matrix X = create_matrix(5, 7);
    // initialize_weights(X);
    // printf("Matrix X:\n");
    // print_matrix(X);

    // Matrix W = create_matrix(7, 10);
    // initialize_weights(W);
    // printf("\nMatrix W:\n");
    // print_matrix(W);

    // printf("\nMatMul(Y = X * W):\n");
    // Matrix Y = mat_mul(X, W);
    // print_matrix(Y);

    // free_matrix(X);
    // free_matrix(W);
    // free_matrix(Y);


    // Matrix A = create_matrix(2, 3);
    // A.data[0] = 1; A.data[1] = 2; A.data[2] = 3;
    // A.data[3] = 4; A.data[4] = 5; A.data[5] = 6;

    // printf("A (2x3):\n");
    // print_matrix(A);

    // Matrix T = mat_transpose(A);
    // printf("\nTransposed A (3x2):\n");
    // print_matrix(T);

    // mat_scale(T, 10.0);
    // printf("\nScaled Transposed A (*10):\n");
    // print_matrix(T);

    // free_matrix(A);
    // free_matrix(T);

    srand(time(NULL));
    Matrix X = create_matrix(3, 2);
    X.data[0] = 1; X.data[1] = 1;
    X.data[2] = 2; X.data[3] = 1;
    X.data[4] = 3; X.data[5] = 1;

    Matrix Y = create_matrix(3, 1);
    Y.data[0] = 7; Y.data[1] = 9; Y.data[2] = 11;

    LinearRegression model = create_linear_regression(2, 1, 0.01);
    fit(&model, X, Y, 1000);

    printf("\nLearned weights:\n");
    print_matrix(model.W);

    Matrix TestX = create_matrix(1, 2);
    TestX.data[0] = 5; TestX.data[1] = 1; 
    
    Matrix Result = predict(model, TestX);
    printf("For x=5 result: %f\n", Result.data[0]);

    free_matrix(X); free_matrix(Y); free_matrix(TestX); free_matrix(Result);
    free_matrix(model.W);

    return 0;
}