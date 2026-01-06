#include <stdio.h>

float relu(float x) {
    if (x < 0) {
        return 0.0;
    } else {
        return x;
    }
}

int main() {
    float val1 = -5.0;
    float val2 = 3.0;

    printf("Relu(val1): %f\n", relu(val1));
    printf("Relu(val2): %f\n", relu(val2));

    return 0;
}