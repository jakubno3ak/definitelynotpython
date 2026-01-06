#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float *data;
    int rows, columns;
} Matrix;

Matrix create_matrix(int rows, int columns);
void free_matrix(Matrix m);
void print_matrix(Matrix m);

#endif