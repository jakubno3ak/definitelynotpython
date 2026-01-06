#include <stdio.h>
#include <stdlib.h>

void swap(int *ptrNum1, int *ptrNum2) {
    int temp;
    temp = *ptrNum1;
    *ptrNum1 = *ptrNum2;
    *ptrNum2 = temp;
}

int main() {
    int num1 = 1, num2 = 2;
    printf("Before swap %d, %d\n", num1, num2);
    swap(&num1, &num2);
    printf("After swap %d, %d\n", num1, num2);

    return 0;
}