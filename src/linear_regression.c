#include <stdio.h>
#include <stdlib.h>
#include "../include/linear_regression.h"

LinearRegression create_linear_regression(int input_features, int output_features, float learning_rate) {
    LinearRegression model;
    model.W = create_matrix(input_features, output_features);
    initialize_weights(model.W);
    model.learning_rate = learning_rate;
    return model;
}

Matrix predict(LinearRegression model, Matrix inputs) {
    return mat_mul(inputs, model.W);
}

void fit(LinearRegression *model, Matrix X, Matrix Y, int epochs) {
    printf("--- LinearRegression: Fitting(%d epochs) ---\n", epochs);

    Matrix Xt = mat_transpose(X);
    for (int i = 0; i <= epochs; i++) {
        Matrix Pred = mat_mul(X, model -> W);
        Matrix Err = create_matrix(Pred.rows, Pred.columns);
        for (int j = 0; j < get_size(Pred); j++) {
            Err.data[j] = Pred.data[j];
        }
        mat_scale(Y, -1.0);
        mat_add(Err, Y);
        mat_scale(Y, -1.0);

        Matrix Grad = mat_mul(Xt, Err);
        mat_scale(Grad, -(model->learning_rate));
        mat_add(model -> W, Grad);

        if (i % (epochs / 10) == 0) {
            printf("Epoch %d | MSE Sample: %f\n", i, Err.data[0]);
        }

        free_matrix(Pred); free_matrix(Err); free_matrix(Grad);
    }

    free_matrix(Xt);
}