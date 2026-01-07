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

    // 1. Data Preparation (Target: y = 2x + 5)
    Matrix X = create_matrix(3, 1);
    X.data[0] = 1; X.data[1] = 2; X.data[2] = 3;

    Matrix Y = create_matrix(3, 1);
    Y.data[0] = 7; Y.data[1] = 9; Y.data[2] = 11;

    // 2. Initialization
    LinearRegression model = create_linear_regression(1, 1, 0.01);

    // 3. Training
    fit(&model, X, Y, 2000);

    // 4. Inference
    Matrix Input = create_matrix(1, 1);
    Input.data[0] = 5; 
    
    Matrix Result = predict(model, Input);
    printf("Result for x=5: %f\n", Result.data[0]);

    // 5. Cleanup
    free_matrix(X); 
    free_matrix(Y); 
    free_matrix(Input); 
    free_matrix(Result);
    free_matrix(model.W);

    return 0;
}
```
