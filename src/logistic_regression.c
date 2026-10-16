#include "logistic_regression.h"
#include "ops.h"
#include <stdio.h>

LogisticRegression logreg_create(int input_features, int output_features, float learning_rate) {
  LogisticRegression model;
  model.W = create_matrix(input_features + 1, output_features);
  initialize_weights(model.W);
  model.learning_rate = learning_rate;
  return model;
}

Matrix logreg_predict_proba(LogisticRegression model, Matrix X) {
  Matrix X_bias = matrix_add_bias(X);
  Matrix result = ops_mat_mul(X_bias, model.W);
  free_matrix(X_bias);
  ops_sigmoid(&result);
  return result;
}

Matrix logreg_predict_class(LogisticRegression model, Matrix X) {
  Matrix Probs = logreg_predict_proba(model, X);
  Matrix Classes = create_matrix(Probs.rows, Probs.columns);

  for (int i = 0; i < get_size(Probs); i++) {
    Classes.data[i] = (Probs.data[i] > 0.5f) ? 1.0f : 0.0f;
  }

  free_matrix(Probs);
  return Classes;
}

static void update_weights(LogisticRegression *model, Matrix Xt, Matrix Err, int n_samples) {
  Matrix Grad = ops_mat_mul(Xt, Err);
  mat_scale(Grad, 1.0f / n_samples);
  mat_scale(Grad, -(model->learning_rate));
  mat_add(model->W, Grad);
  free_matrix(Grad);
}

void logreg_fit(LogisticRegression *model, Matrix X, Matrix Y, int epochs) {
  printf("--- LogisticRegression: Fitting(%d epochs) using %s ---\n", epochs,
         ops_get_backend_name());
  Matrix X_bias = matrix_add_bias(X);
  Matrix Xt = mat_transpose(X_bias);

  for (int i = 0; i <= epochs; i++) {
    Matrix Pred = ops_mat_mul(X_bias, model->W);
    ops_sigmoid(&Pred);

    Matrix Err = create_matrix(Pred.rows, Pred.columns);
    for (int j = 0; j < get_size(Pred); j++) {
      Err.data[j] = Pred.data[j] - Y.data[j];
    }

    update_weights(model, Xt, Err, X.rows);

    if (i % (epochs / 10 + 1) == 0) {
      float mse = 0.0;
      for (int k = 0; k < get_size(Err); k++)
        mse += Err.data[k] * Err.data[k];
      printf("Epoch %d | MSE: %f\n", i, mse / X.rows);
    }

    free_matrix(Pred);
    free_matrix(Err);
  }
  free_matrix(X_bias);
  free_matrix(Xt);
}
