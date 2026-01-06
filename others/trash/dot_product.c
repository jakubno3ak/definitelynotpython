#include <stdio.h>

int main(){
    float inputs[3] = {1.0, 2.0, 3.0};
    float weights[3] = {0.5, 0.25, 0.1};
    float bias = 1.0;

    float output = 0.0;

    for (int i = 0; i < 3 ; i++) {
        float activation = inputs[i] * weights[i];
        output += activation;
        printf("Step: %d, Input: %0.1f * weight: %0.1f = Activation: %0.2f\n", i, inputs[i], weights[i], activation);
    }

    output += bias;

    printf("---------------------\n");
    printf("Result: %f\n", output);
}