# definitelynotpython
Not Python at all. Well…

This is an ML-from-scratch-in-C playground focused on what high-level libraries don’t make you think about: manual memory management, performance-oriented kernels, and GPU compute.

The long-term goal is to ship it as a small Python-friendly mini-framework — because after writing everything in C, it would be a shame not to call it from Python anyway.

## Supported backends
The core logic uses a Dispatcher Pattern (ops.h), allowing the library to switch compute engines at runtime.

- **Naive**: Standard single-threaded C loops (Baseline).

- **OpenMP**: Multi-threaded CPU parallelization.

- **CUDA**: GPU acceleration (Planned).

- **BLAS**: Wrapper for OpenBLAS/MKL (Planned).

### Benchmarks
```markdown
=== BENCHMARK START (Matrix Size: 1024x1024) ===
Threads available: 2 - github workspaces ;]

Running NAIVE backend
NAIVE Time: 3.6847 seconds

Running OPENMP backend
OPENMP Time: 2.9390 seconds

Speedup: 1.25x
```

### Currently implemented ML

1. **Linear Regression**: via Batch Gradient Descent
```c
#include <stdio.h>
#include "include/matrix.h"
#include "include/linear_regression.h"
#include "include/ops.h"

int main() {
    // Optional: Switch compute backend (Default is NAIVE)
    // ops_set_backend(BACKEND_OMP);
    
    printf("--- 1. Loading Data ---\n");
    Matrix RawData = create_matrix_from_csv("boston.csv");
    printf("Loaded Shape: %dx%d\n", RawData.rows, RawData.columns);

    int n = RawData.rows;
    int input_cols = RawData.columns - 1;
    
    Matrix X = create_matrix(n, input_cols);
    Matrix Y = create_matrix(n, 1);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < input_cols; j++) {
            X.data[i * input_cols + j] = RawData.data[i * RawData.columns + j];
        }
        Y.data[i] = RawData.data[i * RawData.columns + input_cols];
    }
    
    free_matrix(RawData); 

    printf("--- 2. Normalizing X ---\n");
    normalize_matrix_min_max(&X);

    printf("--- 3. Training Model ---\n");
    LinearRegression model = create_linear_regression(input_cols, 1, 0.01);
    
    fit(&model, X, Y, 100000); 

    printf("\n--- 4. Predictions (First 5 houses) ---\n");
    for (int i = 0; i < 5; i++) {
        Matrix Sample = create_matrix(1, input_cols);
        
        for (int j = 0; j < input_cols; j++) {
            Sample.data[j] = X.data[i * input_cols + j];
        }

        Matrix Result = predict(model, Sample);
        
        float actual_price = Y.data[i];
        float predicted_price = Result.data[0];

        printf("House %d | Real: %.2fk$ | Pred: %.2fk$ | Error: %.2fk$\n", 
               i, actual_price, predicted_price, predicted_price - actual_price);

        free_matrix(Sample);
        free_matrix(Result);
    }

    free_matrix(X);
    free_matrix(Y);
    free_matrix(model.W);

    return 0;
}
```
