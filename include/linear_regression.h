#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "matrix.h"

typedef struct {
  Matrix W;
  float learning_rate;
} LinearRegression;

LinearRegression linreg_create(int input_features, int output_features, float learning_rate);
void linreg_fit(LinearRegression *model, Matrix X, Matrix Y, int epochs);
Matrix linreg_predict(LinearRegression model, Matrix X);
void linreg_validate(LinearRegression model, Matrix X_test, Matrix y_test);

#endif
