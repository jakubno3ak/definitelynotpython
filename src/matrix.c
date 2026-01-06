#include <stdio.h>
#include <stdlib.h>
#include "../include/matrix.h"

Matrix create_matrix(int rows, int columns) {
    Matrix m;
    m.rows = rows;
    m.columns = columns;

    size_t matrix_size = rows * columns * sizeof(float);

    m.data = (float*)malloc(matrix_size);

    if (m.data == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    return m;
}

void free_matrix(Matrix m) {
    free(m.data);
}

void print_matrix(Matrix m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.columns; j++) {
            int index = i * m.rows + j;
            printf("%.2f ", m.data[index]);
        }
        printf("\n");
    }
}