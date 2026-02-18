# Linear Regression Implementation

Full experiment flow for predicting housing prices using Mean Squared Error.

```c
#include <stdio.h>
#include "include/matrix.h"
#include "include/linear_regression.h"
#include "include/ops.h"

int main() {
    ops_set_backend(BACKEND_OMP);
    
    // Load and Prepare Data
    Matrix RawData = create_matrix_from_csv("boston.csv");
    int input_cols = RawData.columns - 1;
    Matrix X = create_matrix(RawData.rows, input_cols);
    Matrix y = create_matrix(RawData.rows, 1);
    slice_matrix(input_cols, 1, &RawData, &X, &y);
    free_matrix(RawData);

    // Preprocessing
    normalize_matrix_min_max(&X);
    DataSplit split = train_test_split(&X, &y, 0.2, true);

    // Train
    LinearRegression model = create_linear_regression(input_cols, 1, 0.01);
    fit(&model, split.X_train, split.y_train, 10000); 

    // Evaluate
    validate(model, split.X_test, split.y_test);

    // Cleanup
    free_matrix(X); free_matrix(y); free_matrix(model.W);
    free_matrix(split.X_train); free_matrix(split.y_train);
    free_matrix(split.X_test); free_matrix(split.y_test);
    return 0;
}