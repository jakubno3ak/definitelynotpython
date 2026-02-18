#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

typedef enum {
    ACT_RELU,
    ACT_SIGMOID,
    ACT_LINEAR
} ActivationType;

typedef struct {
    Matrix weights;
    Matrix biases;

    Matrix input_cache;
    Matrix output_cache;

    ActivationType activation;
} Layer;

Layer create_layer(int input_size, int output_size, ActivationType activation);
Matrix layer_forward(Layer *layer, Matrix input);
Matrix layer_backward(Layer *layer, Matrix upstream_grad, float learning_rate);
void layer_free(Layer *layer);

#endif
