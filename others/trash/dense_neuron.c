#include <stdio.h>
#include "struct_neuron.h"


int main() {
    struct Neuron layer[2] = {
        { {0.5, 0.5, 0.5}, 1.0 }, 
        { {0.1, 0.2, 0.3}, 0.0 } 
    };

    float inputs[3] = {2.0, 4.0, 6.0};
    float outputs[2];

    for (int i = 0; i < 2; i++) {
        outputs[i] = forward(&layer[i], inputs);
    }

    printf("Layer result: [%f, %f]\n", outputs[0], outputs[1]);
    
    return 0;
}