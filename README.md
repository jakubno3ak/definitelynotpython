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
#include <stdlib.h>
#include <stdbool.h>
#include "include/matrix.h"
#include "include/linear_regression.h"
#include "include/ops.h"

int main() {
    ops_set_backend(BACKEND_OMP); // only for mat_mul for now :) 
    
    printf("--- 1. Loading Data ---\n");
    Matrix RawData = create_matrix_from_csv("boston.csv");
    printf("Loaded Shape: %dx%d\n", RawData.rows, RawData.columns);

    int n = RawData.rows;
    int input_cols = RawData.columns - 1;
    
    Matrix X = create_matrix(n, input_cols);
    Matrix y = create_matrix(n, 1);

    slice_matrix(input_cols, 1, &RawData, &X, &y);

    free_matrix(RawData); 

    printf("--- 2. Normalizing X ---\n");
    normalize_matrix_min_max(&X);

    DataSplit split = train_test_split(&X, &y, 0.2, true);

    printf("--- 3. Training Model ---\n");
    LinearRegression model = create_linear_regression(input_cols, 1, 0.01);
    
    fit(&model, split.X_train, split.y_train, 10000); 

    printf("--- 4. Validate Model ---\n");
    validate(model, split.X_test, split.y_test);

    free_matrix(X);
    free_matrix(y);
    free_matrix(model.W);
    free_matrix(split.X_train);
    free_matrix(split.y_train);
    free_matrix(split.X_test);
    free_matrix(split.y_test);

    return 0;
}
```
