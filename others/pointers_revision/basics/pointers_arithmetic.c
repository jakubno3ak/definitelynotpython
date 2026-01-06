#include <stdio.h>
#include <stdlib.h>

int main() {
    int num = 30;
    int *p;

    printf("%d\n", num);
    p = &num;

    printf("%p\n", &num);
    printf("%p\n", p);

    *p = 20;
    printf("%d\n", num);
    return 0;
}