#include <stdio.h>

struct Neuron {
    float weights[3];
    float bias;
};

float forward(struct Neuron *n, float inputs[3]) {
    float output = 0.0; 

    for (int i = 0; i < 3; i++) {
        // or (*n).weights :) 
        output += inputs[i] * n->weights[i];
    }

    output += n->bias;

    return output;
}

// int main() {
//     struct Neuron neuron = {
//         {0.5, 0.5, 0.5}, // weights
//         1.0              // bias
//     };

//     float inputs[3] = {2.0, 4.0, 6.0};

//     // inputs are already a pointer!
//     float result = forward(&neuron, inputs);
//     printf("Neuron result: %f\n", result);
    
//     return 0;
// }