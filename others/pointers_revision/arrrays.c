#include <stdio.h>
#include <stdlib.h>

#define SIZE 4

int findInArray(int *arr, int size, int key) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == key) {
            return 1;
        }
    }

    return 0;
}

int findMaxOfArray(int *arr, int size) {
    int currMax = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > currMax) {
            currMax = arr[i];
        }
    }

    return currMax;
}

double calculateAvgOfArray(int *arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return (double)(sum / size);
}

void inputArray(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("Enter value #%d", i+1);
        scanf("%d", arr + i);
    }
}

void displayArray(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("Value #%d:  %d\n", i+1, arr[i]);
    }
}


// by ref

void findMinMax(int *pArr, int size, int *min, int *max) {
    *min = pArr[0];
    *max = pArr[0];

    for (int i = 1; i < size; i++) {
        if (pArr[i] < *min) {
            *min = pArr[i];
        }
        if (pArr[i] > *max) {
            *max = pArr[i];
        }
    }
}

void isArraySorted(int *arr, int size, int *is_sorted) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            *is_sorted = 0;
        }
    }
}

int main() {
    int grades[SIZE];
    inputArray(grades, SIZE);
    displayArray(grades, SIZE);

    // int min, max;

    // findMinMax(grades, SIZE, &min, &max);

    // printf("The min in grades is: %d\n", min);
    // printf("The max in grades is: %d\n", max);

    int is_sorted = 1;
    isArraySorted(grades, SIZE, &is_sorted);

    printf("Result: %d\n", is_sorted);

    return 0;
}