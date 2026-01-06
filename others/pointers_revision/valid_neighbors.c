#include <stdio.h>
#include <stdlib.h>


int main() {
    int grades[SIZE];
    inputArray(grades, SIZE);
    displayArray(grades, SIZE);

    int min, max;

    findMinMax(grades, SIZE, &min, &max);

    printf("The min in grades is: %d\n", min);
    printf("The max in grades is: %d\n", max);

    return 0;
}