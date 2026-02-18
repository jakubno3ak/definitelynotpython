#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/matrix.h"
#include "include/ops.h"

void print_softmax_results(Matrix m) {
    float sum = 0.0f;
    printf("Softmax Output:\n");
    for (int j = 0; j < m.columns; j++) {
        float val = m.data[j];
        sum += val;
        printf("  Class %d: %.4f (%.1f%%)\n", j, val, val * 100);
    }
    printf("Total Sum: %.4f (Expected: 1.0000)\n", sum);
}

int main() {
    // We create a matrix [1 x 3] representing raw scores (logits)
    // for 3 different classes (e.g., Cat, Dog, Bird)
    Matrix logits = create_matrix(1, 3);
    
    // Test Case 1: Balanced scores
    logits.data[0] = 1.0f;
    logits.data[1] = 1.0f;
    logits.data[2] = 1.0f;
    
    printf("--- Test 1: Equal Logits [1, 1, 1] ---\n");
    ops_softmax(&logits);
    print_softmax_results(logits);

    // Test Case 2: One dominant score
    logits.data[0] = 2.0f;  // High confidence
    logits.data[1] = -1.0f; // Low confidence
    logits.data[2] = 0.5f;  // Medium confidence
    
    printf("\n--- Test 2: Diverse Logits [2.0, -1.0, 0.5] ---\n");
    ops_softmax(&logits);
    print_softmax_results(logits);

    // Test Case 3: Numerical Stability (Large numbers)
    logits.data[0] = 1000.0f; 
    logits.data[1] = 1001.0f;
    logits.data[2] = 999.0f;
    
    printf("\n--- Test 3: Large Logits (Stability Check) ---\n");
    ops_softmax(&logits);
    print_softmax_results(logits);

    free_matrix(logits);
    return 0;
}