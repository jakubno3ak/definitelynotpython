#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"

typedef struct {
    Matrix W;
    float learning_rate;
} LinearRegression;

LinearRegression create_linear_regression(int input_features, int output_features, float learning_rate);
void fit(LinearRegression *model, Matrix X, Matrix Y, int epochs);
Matrix predict(LinearRegression model, Matrix input);

#endif