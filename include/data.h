#ifndef DATA_H
#define DATA_H

#include "matrix.h"

typedef struct {
  Matrix X_train;
  Matrix y_train;
  Matrix X_test;
  Matrix y_test;
} DataSplit;

Matrix data_load_csv(const char *filename);
void data_split_xy(int inputs_features, int output_features, Matrix *m, Matrix *X, Matrix *y);
DataSplit data_train_test_split(Matrix *X, Matrix *y, float test_size, bool shuffle);

typedef struct {
  Matrix min;   // 1 x columns
  Matrix range; // 1 x columns
} MinMax;

MinMax minmax_fit(Matrix X);
void minmax_apply(MinMax s, Matrix *X);
void minmax_free(MinMax s);

#endif
