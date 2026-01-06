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

    //L, B
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