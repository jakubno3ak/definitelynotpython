#include <stdio.h>
#include <stdlib.h>
#include "include/matrix.h"
#include "include/linear_regression.h"
#include "include/ops.h"

int main() {
    ops_set_backend(BACKEND_OMP); 
    
    printf("--- 1. Loading Data ---\n");
    Matrix RawData = create_matrix_from_csv("boston.csv");
    printf("Loaded Shape: %dx%d\n", RawData.rows, RawData.columns);

    int n = RawData.rows;
    int input_cols = RawData.columns - 1;
    
    Matrix X = create_matrix(n, input_cols);
    Matrix Y = create_matrix(n, 1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < input_cols; j++) {
            X.data[i * input_cols + j] = RawData.data[i * RawData.columns + j];
        }
        Y.data[i] = RawData.data[i * RawData.columns + input_cols];
    }
    
    free_matrix(RawData); 

    printf("--- 2. Normalizing X ---\n");
    normalize_matrix_min_max(&X);

    printf("--- 3. Training Model ---\n");
    LinearRegression model = create_linear_regression(input_cols, 1, 0.01);
    
    fit(&model, X, Y, 100000); 

    printf("\n--- 4. Predictions (First 5 houses) ---\n");
    for (int i = 0; i < 5; i++) {
        Matrix Sample = create_matrix(1, input_cols);
        
        for (int j = 0; j < input_cols; j++) {
            Sample.data[j] = X.data[i * input_cols + j];
        }

        Matrix Result = predict(model, Sample);
        
        float actual_price = Y.data[i];
        float predicted_price = Result.data[0];

        printf("House %d | Real: %.2fk$ | Pred: %.2fk$ | Error: %.2fk$\n", 
               i, actual_price, predicted_price, predicted_price - actual_price);

        free_matrix(Sample);
        free_matrix(Result);
    }

    free_matrix(X);
    free_matrix(Y);
    free_matrix(model.W);

    return 0;
}