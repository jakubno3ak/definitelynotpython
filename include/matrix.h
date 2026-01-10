#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float *data;
    int rows, columns;
} Matrix;

Matrix create_matrix(int rows, int columns);
Matrix mat_transpose(Matrix m);
Matrix matrix_add_bias(Matrix m);
Matrix create_matrix_from_csv(const char* filename);
void mat_add(Matrix a, Matrix b);
void mat_scale(Matrix m, float n);
void free_matrix(Matrix m);
void print_matrix(Matrix m);
void initialize_weights(Matrix m);
void normalize_matrix_min_max(Matrix* m);
void train_test_split(Matrix X, Matrix y, Matrix *X_train, Matrix *X_test, Matrix *y_train, Matrix *y_test);
void slice_matrix(int inputs_features, int output_features, Matrix m, Matrix *X, Matrix *y);
int get_size(Matrix m);

#endif