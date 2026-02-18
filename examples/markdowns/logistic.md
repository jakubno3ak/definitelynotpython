# Logistic Regression Implementation

Binary classification flow for the Diabetes dataset using Sigmoid activation.

```c
#include <stdio.h>
#include "include/matrix.h"
#include "include/linear_regression.h"
#include "include/ops.h"

int main() {
    ops_set_backend(BACKEND_OMP);
    
    // Load and Prepare Data
    Matrix RawData = create_matrix_from_csv("pima-indians-diabetes.csv");
    int input_cols = RawData.columns - 1;
    Matrix X = create_matrix(RawData.rows, input_cols);
    Matrix y = create_matrix(RawData.rows, 1);
    slice_matrix(input_cols, 1, &RawData, &X, &y);
    free_matrix(RawData);

    // Preprocessing
    normalize_matrix_min_max(&X);
    DataSplit split = train_test_split(&X, &y, 0.2, true);

    // Train Logistic Model
    LinearRegression model = create_linear_regression(input_cols, 1, 0.1); 
    fit_logistic(&model, split.X_train, split.y_train, 10000); 

    // Manual Validation Loop
    // ... (Accuracy calculation logic) ...

    // Cleanup
    free_matrix(X); free_matrix(y); free_matrix(model.W);
    // ... (rest of free_matrix calls) ...
    return 0;
}
```