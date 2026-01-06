#include <stdio.h>
#include "include/matrix.h"

int main() {
    Matrix m = create_matrix(3, 3);
    initialize_weights(m);

    m.data[0] = 1.0;
    m.data[4] = 5.0;
    m.data[8] = 9.0;
    
    print_matrix(m);
    free_matrix(m);

    return 0;
}