#include <stdio.h>

int main() {
    float input = 1.5f;
    float bias = 0.5f;

    float output = input + bias;

    printf("Input: %f, Bias: %f\n", input, bias);
    printf("Output of the neuron: %f", output);

    return 0;
}