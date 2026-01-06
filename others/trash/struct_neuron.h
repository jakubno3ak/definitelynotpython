#ifndef STRUCT_NEURON_H
#define STRUCT_NEURON_H

struct Neuron {
    float weights[3];
    float bias;
};

float forward(struct Neuron *n, float inputs[3]);

#endif // STRUCT_NEURON_H
