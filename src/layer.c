#include <stdio.h>
#include <stdlib.h>
#include "include/ops.h"
#include "include/layer.h"
#include "include/ops_kernels.h"

Layer create_layer(int input_size, int output_size, ActivationType activation) {
    Layer layer;
    layer.weights = create_matrix(input_size, output_size);
    layer.biases = create_matrix(1, output_size);
    layer.activation = activation;
    layer.input_cache.data = NULL;
    layer.output_cache.data = NULL;

    initialize_weights(layer.weights);
    
    // initialize biases with zeros
    for(int i = 0; i < get_size(layer.biases); i ++) {
        layer.biases.data[i] = 0.0f;
    }

    return layer;
}

Matrix layer_forward(Layer *layer, Matrix input) {
    Matrix Z = ops_mat_mul(input, layer->weights);
    ops_add_bias(&Z, &layer->biases);

    if (layer->activation == ACT_SIGMOID) {
        ops_sigmoid(&Z);
    } else if (layer->activation == ACT_RELU) {
        ops_relu(&Z);
    } else if (layer->activation == ACT_SOFTMAX) {
        ops_softmax(&Z);
    }
    
    layer->input_cache = input;
    layer->output_cache = Z;

    return Z;
}

Matrix layer_backward(Layer *layer, Matrix upstream_grad, float learning_rate) {
    Matrix dZ = create_matrix(upstream_grad.rows, upstream_grad.columns);
    int size = get_size(dZ);
    
    if (layer->activation == ACT_SIGMOID) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            dZ.data[i] = upstream_grad.data[i] * KERNEL_SIGMOID_PRIME(layer->output_cache.data[i]);
        }
    } else if (layer->activation == ACT_RELU) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            dZ.data[i] = upstream_grad.data[i] * KERNEL_RELU_PRIME(layer->output_cache.data[i]);
        }
    } else {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            dZ.data[i] = upstream_grad.data[i];
        }
    }

    Matrix input_T = mat_transpose(layer->input_cache);
    Matrix dW = ops_mat_mul(input_T, dZ);
    Matrix db = create_matrix(1, dZ.columns);

    for (int k=0; k<db.rows*db.columns; k++) db.data[k] = 0.0f;

    for (int i = 0; i < dZ.rows; i++) {
        for (int j = 0; j < dZ.columns; j++) {
            db.data[j] += dZ.data[i * dZ.columns + j];
        }
    }

    Matrix W_T = mat_transpose(layer->weights);
    Matrix dInput = ops_mat_mul(dZ, W_T);

    mat_scale(dW, -learning_rate);
    mat_scale(db, -learning_rate);
    mat_add(layer->weights, dW);
    mat_add(layer->biases, db);

    free_matrix(W_T);
    free_matrix(dZ);
    free_matrix(input_T);
    free_matrix(dW);
    free_matrix(db);

    return dInput;
}

void layer_free(Layer *layer) {
    free_matrix(layer->weights);
    free_matrix(layer->biases);
}