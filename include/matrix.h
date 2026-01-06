#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float *data;
    int rows, columns;
} Matrix;

Matrix create_matrix(int rows, int columns);
int get_size(Matrix m);
void free_matrix(Matrix m);
void print_matrix(Matrix m);
void initialize_weights(Matrix m);

#endif