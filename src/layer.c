#include <stdio.h>
#include <stdlib.h>
#include "include/ops.h"
#include "include/layer.h"

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

    for (int i = 0; i < Z.rows; i++) {
        for (int j = 0; j < Z.columns; j++) {
            int z_idx = i * Z.columns + j;
            float bias_val = layer->biases.data[j]; 
            Z.data[z_idx] += bias_val;
        }
    }

    float sample_before = Z.data[0];
    if (layer->activation == ACT_SIGMOID) {
        mat_apply(&Z, sigmoid);
    } else if (layer->activation == ACT_RELU) {
        mat_apply(&Z, relu);
    }
    
    layer->input_cache = input;
    layer->output_cache = Z;

    return Z;
}

Matrix layer_backward(Layer *layer, Matrix upstream_grad, float learning_rate) {
    Matrix dZ = create_matrix(upstream_grad.rows, upstream_grad.columns);
    
    for (int i = 0; i < upstream_grad.rows * upstream_grad.columns; i++) {
        float grad = upstream_grad.data[i];
        float out  = layer->output_cache.data[i];
        float derivative = 1.0f;

        if (layer->activation == ACT_SIGMOID) derivative = sigmoid_prime(out);
        else if (layer->activation == ACT_RELU) derivative = relu_prime(out);
    
        dZ.data[i] = grad * derivative;
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