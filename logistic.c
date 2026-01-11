#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "include/matrix.h"
#include "include/linear_regression.h"
#include "include/ops.h"

int main() {
    srand(time(NULL));
    ops_set_backend(BACKEND_OMP); 
    
    printf("--- 1. Loading Data (Diabetes) ---\n");
    Matrix RawData = create_matrix_from_csv("pima-indians-diabetes.csv");
    printf("Loaded Shape: %dx%d\n", RawData.rows, RawData.columns);

    int n = RawData.rows;
    int input_cols = RawData.columns - 1;
    
    Matrix X = create_matrix(n, input_cols);
    Matrix y = create_matrix(n, 1);

    slice_matrix(input_cols, 1, &RawData, &X, &y);
    free_matrix(RawData); 

    printf("--- 2. Normalizing X ---\n");
    normalize_matrix_min_max(&X);


    printf("--- 3. Splitting ---\n");
    DataSplit split = train_test_split(&X, &y, 0.2, true);

    printf("--- 4. Training Logistic Model ---\n");
    LinearRegression model = create_linear_regression(input_cols, 1, 0.1); 
    
    fit_logistic(&model, split.X_train, split.y_train, 10000); 

    printf("\n--- 5. Evaluation ---\n");
    
    Matrix Predictions = predict_class(model, split.X_test);
    
    int correct = 0;
    int total = split.X_test.rows;

    for(int i=0; i<total; i++) {
        float pred = Predictions.data[i];
        float actual = split.y_test.data[i];

        if(pred == actual) {
            correct++;
        }
    }

    printf("Accuracy: %.2f%% (%d/%d)\n", (float)correct/total * 100.0, correct, total);

    free_matrix(X); free_matrix(y); free_matrix(model.W);
    free_matrix(split.X_train); free_matrix(split.y_train);
    free_matrix(split.X_test); free_matrix(split.y_test);
    free_matrix(Predictions);

    return 0;
}