#include "linear_regression.h"
#include "ops.h"
#include <math.h>
#include <stdio.h>

LinearRegression linreg_create(int input_features, int output_features, float learning_rate) {
  LinearRegression model;
  model.W = mat_create(input_features + 1, output_features);
  mat_fill_random(model.W);
  model.learning_rate = learning_rate;
  return model;
}

Matrix linreg_predict(LinearRegression model, Matrix inputs) {
  Matrix InputWithBias = mat_append_ones(inputs);
  Matrix result = ops_mat_mul(InputWithBias, model.W);
  mat_free(InputWithBias);
  return result;
}

static void update_weights(LinearRegression *model, Matrix Xt, Matrix Err, int n_samples) {
  Matrix Grad = ops_mat_mul(Xt, Err);
  mat_scale(Grad, 1.0f / n_samples);
  mat_scale(Grad, -(model->learning_rate));
  mat_add(model->W, Grad);
  mat_free(Grad);
}

void linreg_fit(LinearRegression *model, Matrix X, Matrix Y, int epochs) {
  printf("--- LinearRegression: Fitting(%d epochs) using %s ---\n", epochs, ops_get_backend_name());
  Matrix X_bias = mat_append_ones(X);
  Matrix Xt = mat_transpose(X_bias);

  for (int i = 0; i <= epochs; i++) {
    Matrix Pred = ops_mat_mul(X_bias, model->W);
    Matrix Err = mat_create(Pred.rows, Pred.columns);

    for (int j = 0; j < mat_size(Pred); j++) {
      Err.data[j] = Pred.data[j] - Y.data[j];
    }

    update_weights(model, Xt, Err, X.rows);

    if (i % (epochs / 10 + 1) == 0) {
      float mse = 0.0;
      for (int k = 0; k < mat_size(Err); k++)
        mse += Err.data[k] * Err.data[k];
      printf("Epoch %d | MSE: %f\n", i, mse / X.rows);
    }

    mat_free(Pred);
    mat_free(Err);
  }

  mat_free(X_bias);
  mat_free(Xt);
}

void linreg_validate(LinearRegression model, Matrix X_test, Matrix y_test) {
  Matrix predictions = linreg_predict(model, X_test);

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

  mat_free(predictions);
}