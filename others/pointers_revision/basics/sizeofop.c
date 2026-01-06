#include <stdio.h>
#include <stdlib.h>

int main() {
    int grade = 80, grade3 = 90;
    printf("size of variable = %d\n", sizeof(grade));

    double grade2 = 79.8;
    printf("size of variable = %d\n", sizeof(grade2));

    printf("size of variable = %d\n", sizeof(grade + grade3));
    printf("size of variable = %d\n", sizeof(grade) + sizeof(grade3));

    printf("size of variable = %d\n", sizeof(double));
    printf("size of variable = %d\n", sizeof(char));
    printf("size of variable = %d\n", sizeof(int));
    printf("size of variable = %d\n", sizeof('a'));
    printf("size of variable = %d\n", sizeof("a"));

    int *ptrInt = &grade3;
    printf("size of variable = %d\n", sizeof(ptrInt));

    return 0;
}