#include "logistic_regression.h"
#include "ops.h"
#include <stdio.h>

LogisticRegression logreg_create(int input_features, int output_features, float learning_rate) {
  LogisticRegression model;
  model.W = mat_create(input_features + 1, output_features);
  mat_fill_random(model.W);
  model.learning_rate = learning_rate;
  return model;
}

Matrix logreg_predict_proba(LogisticRegression model, Matrix X) {
  Matrix X_bias = mat_append_ones(X);
  Matrix result = ops_mat_mul(X_bias, model.W);
  mat_free(X_bias);
  ops_sigmoid(&result);
  return result;
}

Matrix logreg_predict_class(LogisticRegression model, Matrix X) {
  Matrix Probs = logreg_predict_proba(model, X);
  Matrix Classes = mat_create(Probs.rows, Probs.columns);

  for (int i = 0; i < mat_size(Probs); i++) {
    Classes.data[i] = (Probs.data[i] > 0.5f) ? 1.0f : 0.0f;
  }

  mat_free(Probs);
  return Classes;
}

static void update_weights(LogisticRegression *model, Matrix Xt, Matrix Err, int n_samples) {
  Matrix Grad = ops_mat_mul(Xt, Err);
  mat_scale(Grad, 1.0f / n_samples);
  mat_scale(Grad, -(model->learning_rate));
  mat_add(model->W, Grad);
  mat_free(Grad);
}

void logreg_fit(LogisticRegression *model, Matrix X, Matrix Y, int epochs) {
  printf("--- LogisticRegression: Fitting(%d epochs) using %s ---\n", epochs,
         ops_get_backend_name());
  Matrix X_bias = mat_append_ones(X);
  Matrix Xt = mat_transpose(X_bias);

  for (int i = 0; i <= epochs; i++) {
    Matrix Pred = ops_mat_mul(X_bias, model->W);
    ops_sigmoid(&Pred);

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
