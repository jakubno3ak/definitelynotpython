#include <stdio.h>
#include <stdlib.h>
#include "../include/matrix.h"

Matrix mat_mul(Matrix a, Matrix b) {
    if (a.columns != b.rows) {
        fprintf(stderr, "Dims mismatch: %dx%d doesn't match with %dx%d\n", 
                a.rows, a.columns, b.rows, b.columns);
        exit(1);
    }

    // L, N
    // N, B

    // L, B
    Matrix result = create_matrix(a.rows, b.columns);

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.columns; j++) {
            float sum = 0.0;
            for (int k = 0; k < a.columns; k++) {
                int idx_a = (i * a.columns) + k;
                int idx_b = (k * b.columns) + j;

                sum += a.data[idx_a] * b.data[idx_b];
            }

            int index = (i * result.columns) + j;
            result.data[index] = sum;
        }
    }

    return result;
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