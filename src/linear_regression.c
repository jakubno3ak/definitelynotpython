#include <stdio.h>
#include <stdlib.h>
#include "../include/ops.h"
#include "../include/linear_regression.h"

LinearRegression create_linear_regression(int input_features, int output_features, float learning_rate) {
    LinearRegression model;
    model.W = create_matrix(input_features + 1, output_features);
    initialize_weights(model.W);
    model.learning_rate = learning_rate;
    return model;
}

Matrix predict(LinearRegression model, Matrix inputs) {
    Matrix InputWithBias = matrix_add_bias(inputs);
    Matrix result = ops_mat_mul(InputWithBias, model.W);
    free_matrix(InputWithBias);
    return result;
}

void fit(LinearRegression *model, Matrix X, Matrix Y, int epochs) {
    printf("--- LinearRegression: Fitting(%d epochs) ---\n", epochs);
    Matrix X_bias = matrix_add_bias(X);
    Matrix Xt = mat_transpose(X_bias);
    for (int i = 0; i <= epochs; i++) {
        Matrix Pred = ops_mat_mul(X_bias, model -> W);
        Matrix Err = create_matrix(Pred.rows, Pred.columns);
        for (int j = 0; j < get_size(Pred); j++) {
            Err.data[j] = Pred.data[j];
        }
        mat_scale(Y, -1.0);
        mat_add(Err, Y);
        mat_scale(Y, -1.0);

        Matrix Grad = ops_mat_mul(Xt, Err);
        mat_scale(Grad, -(model->learning_rate));
        mat_add(model -> W, Grad);

        if (i % (epochs / 10) == 0) {
            float mse = 0.0;
            for (int k = 0; k < Err.rows * Err.columns; k++) {
                float diff = Err.data[k];
                mse += diff * diff;
            }
            mse /= X.rows;
            
            printf("Epoch %d | MSE: %f\n", i, mse);
        }

        free_matrix(Pred); free_matrix(Err); free_matrix(Grad);
    }


    free_matrix(X_bias);
    free_matrix(Xt);
}