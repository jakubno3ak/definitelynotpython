# Linear Regression Implementation

Full experiment flow for predicting housing prices using Mean Squared Error.

```c
#include "include/linear_regression.h"
#include "include/matrix.h"
#include "include/ops.h"
#include <stdbool.h>
#include <stdio.h>

int main() {
  ops_set_backend(BACKEND_OMP);

  printf("--- 1. Loading Data ---\n");
  Matrix RawData = create_matrix_from_csv("boston.csv");
  printf("Loaded Shape: %dx%d\n", RawData.rows, RawData.columns);

  int n = RawData.rows;
  int input_cols = RawData.columns - 1;

  Matrix X = create_matrix(n, input_cols);
  Matrix y = create_matrix(n, 1);

  slice_matrix(input_cols, 1, &RawData, &X, &y);

  free_matrix(RawData);

  printf("--- 2. Normalizing X ---\n");
  normalize_matrix_min_max(&X);

  DataSplit split = train_test_split(&X, &y, 0.2, true);

  printf("--- 3. Training Model ---\n");
  LinearRegression model = create_linear_regression(input_cols, 1, 0.01);

  fit(&model, split.X_train, split.y_train, 10000);

  printf("--- 4. Validate Model ---\n");
  validate(model, split.X_test, split.y_test);

  free_matrix(X);
  free_matrix(y);
  free_matrix(model.W);
  free_matrix(split.X_train);
  free_matrix(split.y_train);
  free_matrix(split.X_test);
  free_matrix(split.y_test);

  return 0;
}
```