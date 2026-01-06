#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void genericSwap(void *a, void *b, unsigned int size) {
    void *tempMemory = malloc(size);

    memcpy(tempMemory, a, size);
    memcpy(a, b, size);
    memcpy(b, tempMemory, size);

    free(tempMemory);
}

int main() {
    int num1 = 5, num2 = 7;
    genericSwap(&num1, &num2, sizeof(int));

    double x = 80.8, y = 90.9;
    genericSwap(&x, &y, sizeof(double));

    return 0;
}