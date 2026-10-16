#include "matrix.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

Matrix mat_create(int rows, int columns) {
  Matrix m;
  m.rows = rows;
  m.columns = columns;

  // calloc zero-fills the memory (malloc leaves garbage) - mat_mul accumulates with +=
  m.data = (float *)calloc(mat_size(m), sizeof(float));

  if (m.data == NULL) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(1);
  }

  m.owns_data = true;
  return m;
}

Matrix mat_view(float *data, int rows, int columns) {
  Matrix m = {.data = data, .rows = rows, .columns = columns, .owns_data = false};
  return m;
}

int mat_size(Matrix m) { return m.rows * m.columns; }

void mat_free(Matrix m) {
  if (m.owns_data)
    free(m.data);
}

void mat_print(Matrix m) {
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.columns; j++) {
      int index = i * m.columns + j;
      printf("%.2f ", m.data[index]);
    }
    printf("\n");
  }
}

void mat_fill_random(Matrix m) {
  for (int i = 0; i < mat_size(m); i++) {
    m.data[i] = (float)rand() / (float)RAND_MAX;
  }
}

Matrix mat_transpose(Matrix m) {
  Matrix t = mat_create(m.columns, m.rows);

  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.columns; j++) {
      int src_idx = (i * m.columns) + j;
      int dst_idx = (j * m.rows) + i;

      t.data[dst_idx] = m.data[src_idx];
    }
  }

  return t;
}

Matrix mat_append_ones(Matrix m) {
  int new_rows = m.rows;
  int new_cols = m.columns + 1;

  Matrix b = mat_create(new_rows, new_cols);

  for (int i = 0; i < new_rows; i++) {
    for (int j = 0; j < m.columns; j++) {
      b.data[i * new_cols + j] = m.data[i * m.columns + j];
    }
    b.data[i * new_cols + m.columns] = 1.0f;
  }
  return b;
}

void mat_scale(Matrix m, float n) {
  for (int i = 0; i < m.rows * m.columns; i++) {
    m.data[i] *= n;
  }
}

void mat_add(Matrix a, Matrix b) {
  if (a.rows != b.rows || a.columns != b.columns) {
    fprintf(stderr, "Error: Dimension mismatch in add\n");
    exit(1);
  }

  for (int i = 0; i < a.rows * a.columns; i++) {
    a.data[i] += b.data[i];
  }
}

void mat_apply(Matrix *m, float (*func)(float)) {
  for (int i = 0; i < mat_size(*m); i++) {
    m->data[i] = func(m->data[i]);
  }
}