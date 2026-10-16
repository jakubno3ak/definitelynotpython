# Logistic Regression Implementation

Binary classification flow for the Diabetes dataset using Sigmoid activation.

```c
#include <stdio.h>
#include "data.h"
#include "matrix.h"
#include "logistic_regression.h"
#include "ops.h"

int main() {
    ops_set_backend(BACKEND_OMP);
    
    // Load and Prepare Data
    Matrix RawData = data_load_csv("pima-indians-diabetes.csv");
    int input_cols = RawData.columns - 1;
    Matrix X = mat_create(RawData.rows, input_cols);
    Matrix y = mat_create(RawData.rows, 1);
    data_split_xy(input_cols, 1, &RawData, &X, &y);
    mat_free(RawData);

    // Preprocessing: split first, then fit the scale on train only
    DataSplit split = data_train_test_split(&X, &y, 0.2, true);
    MinMax scaler = minmax_fit(split.X_train);
    minmax_apply(scaler, &split.X_train);
    minmax_apply(scaler, &split.X_test);
    minmax_free(scaler);

    // Train Logistic Model
    LogisticRegression model = logreg_create(input_cols, 1, 0.1); 
    logreg_fit(&model, split.X_train, split.y_train, 10000); 

    // Manual Validation Loop
    // ... (Accuracy calculation logic) ...

    // Cleanup
    mat_free(X); mat_free(y); mat_free(model.W);
    // ... (rest of mat_free calls) ...
    return 0;
}
```