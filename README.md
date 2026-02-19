# definitelynotpython
Not Python at all. Well…

This is an ML-from-scratch-in-C playground focused on what high-level libraries don’t make you think about: manual memory management, performance-oriented kernels, and GPU compute.

The long-term goal is to ship it as a small Python-friendly mini-framework — because after writing everything in C, it would be a shame not to call it from Python anyway.

## Quick Navigation
- [Linear Regression](examples/markdowns/linear.md) - Classic BGD on Boston Housing dataset.
- [Logistic Regression](examples/markdowns/logistic.md) - Binary classification with Sigmoid.
- [Neural Layers (MLP)](examples/markdowns/mlp.md) - Deep learning building blocks and Manual Backpropagation.
- [Data Helpers](examples/markdowns/helpers.md) - Normalization, Train/Test splitting, and Shuffling logic.

## Supported backends
The core logic uses a Dispatcher Pattern (ops.h), allowing the library to switch compute engines at runtime.

- [Naive](src/backends/ops_naive.c): Standard single-threaded C loops (Baseline).
- [OpenMP](src/backends/ops_omp.c): Multi-threaded CPU parallelization.
- [BLAS](src/backends/ops_blas.c): Wrapper for OpenBLAS/MKL.
- CUDA: GPU acceleration (Planned).


To enable <b>BLAS</b> (Basic Linear Algebra Subprograms) run: 

```bash
sudo apt-get update
sudo apt-get install libopenblas-dev
```

### Benchmarks - for now only for matrix multiplication
```markdown
=== BENCHMARK START (Matrix Size: 1024x1024) ===
Threads available: 2

Running NAIVE backend...
--- Backend switched to: Naive (Single Thread) ---
NAIVE Time: 3.9615 seconds

Running OPENMP backend...
--- Backend switched to: OpenMP (Multi Thread) ---
OPENMP Time: 3.4465 seconds
Verification PASSED for OPENMP

Running BLAS backend...
--- Backend switched to: BLAS (Optimized) ---
BLAS Time: 0.0291 seconds

------------------------------------------------
FINAL RESULTS SUMMARY (Reference: Naive)
OPENMP Speedup: 1.15x
BLAS   Speedup: 135.98x
BLAS vs OPENMP: 118.30x faster
------------------------------------------------
--- Done ---
```

### Build and run
```markdown
# Make the script executable
chmod +x run.sh

# Compile and run any example from the examples/ folder
./run.sh examples/test_layer.c
```