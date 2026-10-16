#include "data.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Matrix data_load_csv(const char *filename) {
  FILE *fp = fopen(filename, "r");
  if (fp == NULL) {
    fprintf(stderr, "Error: Could not open file %s\n", filename);
    exit(1);
  }

  int rows = 0;
  int columns = 0;
  char buffer[10240];

  if (fgets(buffer, sizeof(buffer), fp)) {
    rows++;

    char temp_buffer[10240];
    strcpy(temp_buffer, buffer);

    char *token = strtok(temp_buffer, ",");
    while (token) {
      columns++;
      token = strtok(NULL, ",");
    }
  }

  while (fgets(buffer, sizeof(buffer), fp)) {
    rows++;
  }

  printf("CSV Loader: Detected %d rows, %d columns\n", rows, columns);

  Matrix m = mat_create(rows - 1, columns);
  rewind(fp);

  if (fgets(buffer, sizeof(buffer), fp) == NULL) {
    fprintf(stderr, "Error: %s is empty\n", filename);
    exit(1);
  }

  int i = 0;
  while (fgets(buffer, sizeof(buffer), fp)) {
    buffer[strcspn(buffer, "\n")] = 0;

    int j = 0;
    char *token = strtok(buffer, ",");

    while (token && j < columns) {
      m.data[i * columns + j] = atof(token);
      token = strtok(NULL, ",");
      j++;
    }
    i++;
  }

  fclose(fp);
  return m;
}

void data_split_xy(int inputs_features, int output_features, Matrix *m, Matrix *X, Matrix *y) {
  for (int i = 0; i < X->rows; i++) {
    for (int j = 0; j < inputs_features; j++) {
      X->data[i * X->columns + j] = m->data[i * m->columns + j];
    }
    for (int k = 0; k < output_features; k++) {
      y->data[i * output_features + k] = m->data[i * m->columns + inputs_features + k];
    }
  }
}

static void shuffle_samples(Matrix *X, Matrix *y) {
  if (X->rows != y->rows) {
    fprintf(stderr, "Error: X and y rows mismatch in shuffle!\n");
    exit(1);
  }

  int n = X->rows;
  int columns = X->columns;

  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);

    float temp_y = y->data[i];
    y->data[i] = y->data[j];
    y->data[j] = temp_y;

    for (int k = 0; k < columns; k++) {
      float temp_x = X->data[i * columns + k];
      X->data[i * columns + k] = X->data[j * columns + k];
      X->data[j * columns + k] = temp_x;
    }
  }
}

DataSplit data_train_test_split(Matrix *X, Matrix *y, float test_size, bool shuffle) {
  if (shuffle) {
    shuffle_samples(X, y);
  }

  DataSplit split;

  int n = X->rows;
  int test_rows = (int)(n * test_size);
  int train_rows = n - test_rows;

  split.X_train = mat_create(train_rows, X->columns);
  split.y_train = mat_create(train_rows, 1);
  split.X_test = mat_create(test_rows, X->columns);
  split.y_test = mat_create(test_rows, 1);

  for (int i = 0; i < train_rows; i++) {
    for (int j = 0; j < X->columns; j++) {
      split.X_train.data[i * X->columns + j] = X->data[i * X->columns + j];
    }
    split.y_train.data[i] = y->data[i];
  }

  for (int i = 0; i < test_rows; i++) {
    int src_idx = i + train_rows;
    for (int j = 0; j < X->columns; j++) {
      split.X_test.data[i * X->columns + j] = X->data[src_idx * X->columns + j];
    }
    split.y_test.data[i] = y->data[src_idx];
  }

  return split;
}

MinMax minmax_fit(Matrix X) {
  MinMax s = {mat_create(1, X.columns), mat_create(1, X.columns)};

  for (int j = 0; j < X.columns; j++) {
    float min = FLT_MAX;
    float max = -FLT_MAX;
    for (int i = 0; i < X.rows; i++) {
      float val = X.data[i * X.columns + j];
      if (val < min)
        min = val;
      if (val > max)
        max = val;
    }

    float range = max - min;
    if (range < 0.00001f)
      range = 1.0f; // constant column

    s.min.data[j] = min;
    s.range.data[j] = range;
  }
  return s;
}

void minmax_apply(MinMax s, Matrix *X) {
  for (int i = 0; i < X->rows; i++) {
    for (int j = 0; j < X->columns; j++) {
      int idx = i * X->columns + j;
      X->data[idx] = (X->data[idx] - s.min.data[j]) / s.range.data[j];
    }
  }
}

void minmax_free(MinMax s) {
  mat_free(s.min);
  mat_free(s.range);
}
