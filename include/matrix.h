#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float *data;
    int rows, columns;
} Matrix;

Matrix create_matrix(int rows, int columns);
Matrix mat_transpose(Matrix m);
Matrix matrix_add_bias(Matrix m);
void mat_add(Matrix a, Matrix b);
void mat_scale(Matrix m, float n);
void free_matrix(Matrix m);
void print_matrix(Matrix m);
void initialize_weights(Matrix m);
int get_size(Matrix m);

#endif