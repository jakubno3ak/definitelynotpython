#include "data.h"
#include "linear_regression.h"
#include "matrix.h"
#include "ops.h"
#include <stdbool.h>
#include <stdio.h>

int main() {
  ops_set_backend(BACKEND_OMP);

  printf("--- 1. Loading Data ---\n");
  Matrix RawData = data_load_csv("boston.csv");
  printf("Loaded Shape: %dx%d\n", RawData.rows, RawData.columns);

  int n = RawData.rows;
  int input_cols = RawData.columns - 1;

  Matrix X = mat_create(n, input_cols);
  Matrix y = mat_create(n, 1);

  data_split_xy(input_cols, 1, &RawData, &X, &y);

  mat_free(RawData);

  DataSplit split = data_train_test_split(&X, &y, 0.2, true);

  printf("--- 2. Normalizing X (scale fitted on train only) ---\n");
  MinMax scaler = minmax_fit(split.X_train);
  minmax_apply(scaler, &split.X_train);
  minmax_apply(scaler, &split.X_test);
  minmax_free(scaler);

  printf("--- 3. Training Model ---\n");
  LinearRegression model = linreg_create(input_cols, 1, 0.01);

  linreg_fit(&model, split.X_train, split.y_train, 10000);

  printf("--- 4. Validate Model ---\n");
  linreg_validate(model, split.X_test, split.y_test);

  mat_free(X);
  mat_free(y);
  mat_free(model.W);
  mat_free(split.X_train);
  mat_free(split.y_train);
  mat_free(split.X_test);
  mat_free(split.y_test);

  return 0;
}