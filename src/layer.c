#include<stdio.h>
#include<stdlib.h>
#include<ops.h>
#include "layer.h"

Layer create_layer(int input_size, int output_size, ActivationType activation) {
    Layer layer;

    layer.weights = create_matrix(input_size, output_size);
    layer.biases = create_matrix(1, output_size);
    layer.activation = activation;

    layer.input_cache.data = NULL;
    layer.output_cache.data = NULL;

    initialize_weights(layer.weights);
    initialize_weights(layer.biases);

    return layer;
}


Matrix layer_forward(Layer *layer, Matrix input) {
    Matrix Z = ops_mat_mul(input, layer -> weights);
    
    // broadcast
    for (int i = 0; i < Z.rows; i++) {
        for (int j = 0; j < Z.columns; j++) {
            int z_idx = i * Z.columns + j;
            float bias_val = layer -> biases.data[j];
            Z.data[z_idx] +=  bias_val;
        }
    }

    // apply activation
    if (layer -> activation == ACT_SIGMOID) {
        mat_apply(&Z, sigmoid);
    } else if (layer -> activation == ACT_RELU)
    {
        mat_apply(&Z, relu);
    }

    // initialize cache for future back prop
    layer -> input_cache = input;
    layer -> output_cache = Z;

    return Z;
}