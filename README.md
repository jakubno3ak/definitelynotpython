# definitelynotpython
Not Python at all. Well…

This is an ML-from-scratch-in-C playground focused on what high-level libraries don’t make you think about: manual memory management, performance-oriented kernels, and GPU compute.

The long-term goal is to ship it as a small Python-friendly mini-framework — because after writing everything in C, it would be a shame not to call it from Python anyway.

## Quick Navigation
- [Linear Regression](markdowns/linear.md) - Classic BGD on Boston Housing dataset.
- [Logistic Regression](markdowns/logistic.md) - Binary classification with Sigmoid.
- [Neural Layers (MLP)](markdowns/mlp.md) - Deep learning building blocks and Manual Backpropagation.
- [Data Helpers](markdowns/helpers.md) - Normalization, Train/Test splitting, and Shuffling logic.

## Supported backends
The core logic uses a Dispatcher Pattern (ops.h), allowing the library to switch compute engines at runtime.

- Naive: Standard single-threaded C loops (Baseline).
- OpenMP: Multi-threaded CPU parallelization.
- CUDA: GPU acceleration (Planned).
- BLAS: Wrapper for OpenBLAS/MKL (Planned).

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

### Build and run
```markdown
# Make the script executable
chmod +x build.sh

# Compile any example from the examples/ folder
./build.sh examples/test_layer.c

# Run the generated binary
./test_layer
```