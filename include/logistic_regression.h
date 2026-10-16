#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include "matrix.h"

typedef struct {
  Matrix W;
  float learning_rate;
} LogisticRegression;

LogisticRegression logreg_create(int input_features, int output_features, float learning_rate);
void logreg_fit(LogisticRegression *model, Matrix X, Matrix Y, int epochs);
Matrix logreg_predict_proba(LogisticRegression model, Matrix X);
Matrix logreg_predict_class(LogisticRegression model, Matrix X);

#endif
