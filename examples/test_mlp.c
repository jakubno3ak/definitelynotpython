#include "data.h"
#include "layer.h"
#include "matrix.h"
#include "ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

float calculate_accuracy(Matrix predictions, Matrix targets) {
  int correct = 0;
  for (int i = 0; i < predictions.rows; i++) {
    int pred_class = predictions.data[i] > 0.5 ? 1 : 0;
    int actual_class = (int)targets.data[i];

    if (pred_class == actual_class) {
      correct++;
    }
  }
  return (float)correct / predictions.rows;
}

int main() {
  srand(42);
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

  printf("--- 4. Initializes Neural Layer ---\n");
  Layer layer = create_layer(input_cols, 1, ACT_SIGMOID);

  int epochs = 10000;
  float learning_rate = 0.01f;

  printf("--- 5. Training Loop ---\n");
  for (int epoch = 0; epoch < epochs; epoch++) {
    Matrix output = layer_forward(&layer, split.X_train);
    Matrix upstream_grad = mat_create(output.rows, output.columns);

    for (int i = 0; i < output.rows * output.columns; i++) {
      upstream_grad.data[i] = output.data[i] - split.y_train.data[i];
    }

    Matrix dInput = layer_backward(&layer, upstream_grad, learning_rate);

    mat_free(output);
    mat_free(upstream_grad);
    mat_free(dInput);

    if (epoch % 1000 == 0) {
      printf("Epoch %d/ %d completed.\n", epoch, epochs);
    }
  }

  printf("\n--- 6. Evaluation ---\n");

  Matrix test_pred = layer_forward(&layer, split.X_test);
  float acc = calculate_accuracy(test_pred, split.y_test);

  printf("Test Accuracy: %.2f%%\n", acc * 100.0f);

  mat_free(test_pred);
  mat_free(X);
  mat_free(y);
  mat_free(split.X_train);
  mat_free(split.y_train);
  mat_free(split.X_test);
  mat_free(split.y_test);
  layer_free(&layer);

  return 0;
}