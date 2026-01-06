#include <stdio.h>
#include <stdlib.h>

int findMax(int num1, int num2) {
    if (num1 > num2) {
        return num1;
    } else {
        return num2;
    }
}


int findMin(int num1, int num2) {
    if (num1 > num2) {
        return num2;
    } else {
        return num1;
    }
}

void findMinMax(int num1, int num2, int *min, int *max) {
    if (num1 > num2) {
        *min = num2;
        *max = num1;
    } else {
        *min = num1;
        *max = num2;
    }
}

int main() {
    int a = 5; int b = 7;
    int max, min;
    
    printf("val a: %d\n", a);
    printf("addr a: %p\n", &a);
    printf("val b: %d\n", b);
    printf("addr b: %p\n", &b);

    findMinMax(a, b, &min, &max);

    printf("val max: %d\n", max);
    printf("addr max: %p\n", &max);
    printf("val min: %d\n", min);
    printf("addr min: %p\n", &min);

    return 0;
}