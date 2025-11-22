#include <stdio.h>

void relu(float *x) {
    if (*x < 0) {
        *x = 0;
    }
}

int main() {
    float val = -10.0;

    printf("Before in_place relu: %f\n", val);

    relu(&val);

    printf("After in_place relu: %f\n", val);

    return 0;
}