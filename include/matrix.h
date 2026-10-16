#ifndef MATRIX_H
#define MATRIX_H

#include <stdbool.h>

typedef struct {
  float *data;
  int rows, columns;
  bool owns_data; // false = view into someone else's memory
} Matrix;

Matrix mat_create(int rows, int columns);
Matrix mat_view(float *data, int rows, int columns);
Matrix mat_transpose(Matrix m);
Matrix mat_append_ones(Matrix m);
void mat_add(Matrix a, Matrix b);
void mat_scale(Matrix m, float n);
void mat_free(Matrix m);
void mat_print(Matrix m);
void mat_fill_random(Matrix m);
void mat_apply(Matrix *m, float (*func)(float));
int mat_size(Matrix m);

#endif