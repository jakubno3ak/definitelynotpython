#include "data.h"
#include "logistic_regression.h"
#include "matrix.h"
#include "ops.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  srand(42);
  ops_set_backend(BACKEND_OMP);

  printf("--- 1. Loading Data (Diabetes) ---\n");
  Matrix RawData = data_load_csv("pima-indians-diabetes.csv");
  printf("Loaded Shape: %dx%d\n", RawData.rows, RawData.columns);

  int n = RawData.rows;
  int input_cols = RawData.columns - 1;

  Matrix X = mat_create(n, input_cols);
  Matrix y = mat_create(n, 1);

  data_split_xy(input_cols, 1, &RawData, &X, &y);
  mat_free(RawData);

  printf("--- 2. Splitting ---\n");
  DataSplit split = data_train_test_split(&X, &y, 0.2, true);

  printf("--- 3. Normalizing X (scale fitted on train only) ---\n");
  MinMax scaler = minmax_fit(split.X_train);
  minmax_apply(scaler, &split.X_train);
  minmax_apply(scaler, &split.X_test);
  minmax_free(scaler);

  printf("--- 4. Training Logistic Model ---\n");
  LogisticRegression model = logreg_create(input_cols, 1, 0.1);

  logreg_fit(&model, split.X_train, split.y_train, 10000);

  printf("\n--- 5. Evaluation ---\n");

  Matrix Predictions = logreg_predict_class(model, split.X_test);

  int correct = 0;
  int total = split.X_test.rows;

  for (int i = 0; i < total; i++) {
    float pred = Predictions.data[i];
    float actual = split.y_test.data[i];

    if (pred == actual) {
      correct++;
    }
  }

  printf("Accuracy: %.2f%% (%d/%d)\n", (float)correct / total * 100.0, correct, total);

  mat_free(X);
  mat_free(y);
  mat_free(model.W);
  mat_free(split.X_train);
  mat_free(split.y_train);
  mat_free(split.X_test);
  mat_free(split.y_test);
  mat_free(Predictions);

  return 0;
}