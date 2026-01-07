#include <stdio.h>
#include <stdlib.h>
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