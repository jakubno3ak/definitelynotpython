#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include "../include/matrix.h"

Matrix create_matrix(int rows, int columns) {
    Matrix m;
    m.rows = rows;
    m.columns = columns;

    size_t matrix_size = get_size(m) * sizeof(float);

    m.data = (float*)malloc(matrix_size);

    if (m.data == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    return m;
}

Matrix create_matrix_from_csv(const char* filename) {
    FILE* fp = fopen(filename, "r");
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

        char* token = strtok(temp_buffer, ",");
        while (token) {
            columns++;
            token = strtok(NULL, ",");
        }
    }

    while (fgets(buffer, sizeof(buffer), fp)) {
        rows++;
    }

    printf("CSV Loader: Detected %d rows, %d columns\n", rows, columns);

    Matrix m = create_matrix(rows - 1, columns);
    rewind(fp);

    fgets(buffer, sizeof(buffer), fp);

    int i = 0;
    while (fgets(buffer, sizeof(buffer), fp)) {
        buffer[strcspn(buffer, "\n")] = 0;

        int j = 0;
        char* token = strtok(buffer, ",");

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

int get_size(Matrix m) {
    return m.rows * m.columns;
}

void free_matrix(Matrix m) {
    free(m.data);
}

void print_matrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.columns; j++) {
            int index = i * m.columns + j;
            printf("%.2f ", m.data[index]);
        }
        printf("\n");
    }
}

void initialize_weights(Matrix m) {
    for (int i = 0; i < get_size(m); i++) {
        m.data[i] = (float)rand() / (float)RAND_MAX;
    }
}

Matrix mat_transpose(Matrix m) {
    Matrix t = create_matrix(m.columns, m.rows);

    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.columns; j++) {
            int src_idx = (i * m.columns) + j;
            int dst_idx = (j * m.rows ) + i;

            t.data[dst_idx] = m.data[src_idx];
        }
    }

    return t;
}

Matrix matrix_add_bias(Matrix m) {
    int new_rows = m.rows;
    int new_cols = m.columns + 1;
    
    Matrix b = create_matrix(new_rows, new_cols);

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

void normalize_matrix_min_max(Matrix *m) {
    for (int j = 0; j < m -> columns; j++) {
        float min = FLT_MAX;
        float max = -FLT_MAX;

        for (int i = 0; i < m -> rows; i++) {
            int column_idx = i * m -> columns + j;
            float val = m -> data[column_idx];
            
            if (val <= min) {
                min = val;
            }

            if (val >= max) {
                max = val;
            }
        }

        float range = max - min;
        if (range < 0.00001) {
            range = 1.0;
        }

        for (int i = 0; i < m -> rows; i++) {
            int column_idx = i * m -> columns + j;
            float val = m -> data[column_idx];
            m -> data[column_idx] = (val - min) / range;
        }
    }
}

void slice_matrix(int inputs_features, int output_features, Matrix *m, Matrix *X, Matrix *y) {
    for (int i = 0; i < X->rows; i++) {
        for (int j = 0; j < inputs_features; j++) {
            X->data[i * X->columns + j] = m->data[i * m->columns + j];
        }
        y->data[i] = m->data[i * m->columns + inputs_features];
    }
}


DataSplit train_test_split(Matrix *X, Matrix *y, float test_size) {
    DataSplit split;

    int n = X -> rows;
    int test_rows = (int)(n * test_size);
    int train_rows = n - test_rows;

    split.X_train = create_matrix(train_rows, X -> columns);
    split.y_train = create_matrix(train_rows, 1);
    split.X_test  = create_matrix(test_rows, X -> columns);
    split.y_test  = create_matrix(test_rows, 1);

    for (int i = 0; i < train_rows; i++) {
       for (int j = 0; j < X -> columns; j++) {
        split.X_train.data[i * X -> columns + j] = X -> data[i * X -> columns + j];
       }
       split.y_train.data[i] = y -> data[i];
    }

    for (int i = 0; i < test_rows; i ++) {
        int src_idx = i + train_rows;
        for (int j = 0; j < X -> columns; j++) {
            split.X_test.data[i * X -> columns + j] = X -> data[src_idx * X -> columns + j];
        }
        split.y_test.data[i] = y ->data [src_idx];
    }

    return split;
}