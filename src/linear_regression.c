#include "../include/linear_regression.h"
#include "../include/ops.h"
#include <math.h>
#include <stdio.h>

LinearRegression create_linear_regression(int input_features,
                                          int output_features,
                                          float learning_rate) {
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

Matrix predict_proba(LinearRegression model, Matrix inputs) {
  Matrix InputWithBias = matrix_add_bias(inputs);
  Matrix result = ops_mat_mul(InputWithBias, model.W);
  mat_apply(&result, sigmoid);
  free_matrix(InputWithBias);
  return result;
}

Matrix predict_class(LinearRegression model, Matrix X) {
    Matrix Probs = predict_proba(model, X);
    Matrix Classes = create_matrix(Probs.rows, Probs.columns);

    for (int i = 0; i < get_size(Probs); i++) {
        Classes.data[i] = (Probs.data[i] > 0.5f) ? 1.0f : 0.0f;
    }

    free_matrix(Probs);
    return Classes;
}

void fit(LinearRegression *model, Matrix X, Matrix Y, int epochs) {
  printf("--- LinearRegression: Fitting(%d epochs) ---\n", epochs);
  Matrix X_bias = matrix_add_bias(X);
  Matrix Xt = mat_transpose(X_bias);
  for (int i = 0; i <= epochs; i++) {
    Matrix Pred = ops_mat_mul(X_bias, model->W);
    Matrix Err = create_matrix(Pred.rows, Pred.columns);
    for (int j = 0; j < get_size(Pred); j++) {
        Err.data[j] = Pred.data[j] - Y.data[j]; 
    }

    Matrix Grad = ops_mat_mul(Xt, Err);
    mat_scale(Grad, 1.0f / X.rows);
    mat_scale(Grad, -(model->learning_rate));
    mat_add(model->W, Grad);

    if (i % (epochs / 10) == 0) {
      float mse = 0.0;
      for (int k = 0; k < Err.rows * Err.columns; k++) {
        float diff = Err.data[k];
        mse += diff * diff;
      }
      mse /= X.rows;

      printf("Epoch %d | MSE: %f\n", i, mse);
    }

    free_matrix(Pred);
    free_matrix(Err);
    free_matrix(Grad);
  }

  free_matrix(X_bias);
  free_matrix(Xt);
}

void fit_logistic(LinearRegression *model, Matrix X, Matrix Y, int epochs) {
  printf("--- LogisticRegression: Fitting(%d epochs) ---\n", epochs);
  Matrix X_bias = matrix_add_bias(X);
  Matrix Xt = mat_transpose(X_bias);
  for (int i = 0; i <= epochs; i++) {
    Matrix Pred = ops_mat_mul(X_bias, model->W);
    mat_apply(&Pred, sigmoid);
    Matrix Err = create_matrix(Pred.rows, Pred.columns);
    for (int j = 0; j < get_size(Pred); j++) {
        Err.data[j] = Pred.data[j] - Y.data[j]; 
    }

    Matrix Grad = ops_mat_mul(Xt, Err);
    mat_scale(Grad, 1.0f / X.rows);
    mat_scale(Grad, -(model->learning_rate));
    mat_add(model->W, Grad);

    if (i % (epochs / 10) == 0) {
      float mse = 0.0;
      for (int k = 0; k < Err.rows * Err.columns; k++) {
        float diff = Err.data[k];
        mse += diff * diff;
      }
      mse /= X.rows;

      printf("Epoch %d | MSE: %f\n", i, mse);
    }

    free_matrix(Pred);
    free_matrix(Err);
    free_matrix(Grad);
  }

  free_matrix(X_bias);
  free_matrix(Xt);
}

void validate(LinearRegression model, Matrix X_test, Matrix y_test) {
  Matrix predictions = predict(model, X_test);

  float total_mse = 0.0f;
  float total_mae = 0.0f;
  int n = X_test.rows;

  for (int i = 0; i < n; i++) {
    float pred = predictions.data[i];
    float actual = y_test.data[i];

    float diff = pred - actual;

    total_mse += diff * diff;
    total_mae += fabsf(diff);
  }

  float mse = total_mse / n;
  float mae = total_mae / n;

  printf("\n--- Validation Results (on %d unseen samples) ---\n", n);
  printf("MSE: %.4f\n", mse);
  printf("MAE: %.4f k$ (Average Error: ~%.0f$)\n", mae, mae * 1000);

  free_matrix(predictions);
}