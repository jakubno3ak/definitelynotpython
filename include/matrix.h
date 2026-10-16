#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

typedef struct {
  float *data;
  int rows, columns;
  bool owns_data; // false = view into someone else's memory
} Matrix;

typedef struct {
  Matrix X_train;
  Matrix y_train;
  Matrix X_test;
  Matrix y_test;
} DataSplit;

Matrix create_matrix(int rows, int columns);
Matrix matrix_view(float *data, int rows, int columns);
Matrix mat_transpose(Matrix m);
Matrix matrix_add_bias(Matrix m);
Matrix create_matrix_from_csv(const char *filename);
void mat_add(Matrix a, Matrix b);
void mat_scale(Matrix m, float n);
void free_matrix(Matrix m);
void print_matrix(Matrix m);
void initialize_weights(Matrix m);
void slice_matrix(int inputs_features, int output_features, Matrix *m, Matrix *X, Matrix *y);
void mat_apply(Matrix *m, float (*func)(float));
int get_size(Matrix m);

typedef struct {
  Matrix min;   // 1 x columns
  Matrix range; // 1 x columns
} MinMax;

MinMax minmax_fit(Matrix X);
void minmax_apply(MinMax s, Matrix *X);
void minmax_free(MinMax s);

DataSplit train_test_split(Matrix *X, Matrix *y, float test_size, bool shuffle);

#endif