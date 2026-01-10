#include "include/linear_regression.h"
#include "include/matrix.h"
#include "include/ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
  srand(time(NULL));

  ops_set_backend(BACKEND_OMP);

  Matrix X = create_matrix(3, 1);
  X.data[0] = 1;
  X.data[1] = 2;
  X.data[2] = 3;

  Matrix Y = create_matrix(3, 1);
  Y.data[0] = 7;
  Y.data[1] = 9;
  Y.data[2] = 11;

  LinearRegression model = create_linear_regression(1, 1, 0.01);
  fit(&model, X, Y, 10000);

  printf("\nLearned weights:\n");
  print_matrix(model.W);

  Matrix TestX = create_matrix(1, 1);
  TestX.data[0] = 5;

  Matrix Result = predict(model, TestX);
  printf("For x=5 result: %f\n", Result.data[0]);

  free_matrix(X);
  free_matrix(Y);
  free_matrix(TestX);
  free_matrix(Result);
  free_matrix(model.W);

  return 0;
}