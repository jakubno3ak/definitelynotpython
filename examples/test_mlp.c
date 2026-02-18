#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "include/matrix.h"
#include "include/ops.h"
#include "include/layer.h"

float calculate_accuracy(Matrix predictions, Matrix targets) {
    int correct = 0;
    for(int i=0; i<predictions.rows; i++) {
        int pred_class = predictions.data[i] > 0.5 ? 1 : 0;
        int actual_class = (int)targets.data[i];
        
        if(pred_class == actual_class) {
            correct++;
        }
    }
    return (float)correct / predictions.rows;
}

int main() {
    srand(time(NULL));
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

    printf("--- 4. Initializes Neural Layer ---\n");
    Layer layer = create_layer(input_cols, 1, ACT_SIGMOID);
    
    int epochs = 10000;
    float learning_rate = 0.01f;

    printf("--- 5. Training Loop ---\n");
    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix output = layer_forward(&layer, split.X_train);
        Matrix upstream_grad = create_matrix(output.rows, output.columns);

        for(int i=0; i<output.rows * output.columns; i++) {
            upstream_grad.data[i] = output.data[i] - split.y_train.data[i];
        }

        Matrix dInput = layer_backward(&layer, upstream_grad, learning_rate);

        free_matrix(output); 
        free_matrix(upstream_grad);
        free_matrix(dInput);

        if (epoch % 1000 == 0) {
            float loss = 0.0f;
            printf("Epoch %d/ %d completed.\n", epoch, epochs);
        }
    }

    printf("\n--- 6. Evaluation ---\n");
    
    Matrix test_pred = layer_forward(&layer, split.X_test);
    float acc = calculate_accuracy(test_pred, split.y_test);
    
    printf("Test Accuracy: %.2f%%\n", acc * 100.0f);

    free_matrix(test_pred);
    free_matrix(X); 
    free_matrix(y);     
    free_matrix(split.X_train); 
    free_matrix(split.y_train);
    free_matrix(split.X_test); 
    free_matrix(split.y_test);
    layer_free(&layer);

    return 0;
}