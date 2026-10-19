# definitelynotpython
[![CI](https://github.com/jakubno3ak/definitelynotpython/actions/workflows/ci.yml/badge.svg)](https://github.com/jakubno3ak/definitelynotpython/actions/workflows/ci.yml)

Not Python at all. Well…

This is an ML-from-scratch-in-C playground focused on what high-level libraries don’t make you think about: manual memory management, performance-oriented kernels, and GPU compute.

The long-term goal is to ship it as a small Python-friendly mini-framework, because after writing everything in C, it would be a shame not to call it from Python anyway.

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

### Benchmarks
Intel i5-9600K (6 cores), `-O3 -march=native` (max optimizations + AVX2).

#### Matrix multiplication (1024×1024, float32)
Best of 5 runs, `make bench`.

| Backend | Before (i-j-k loops) | Now | GFLOP/s (now) |
|---|---|---|---|
| Naive (1 thread) | 1.58 s | 0.116 s | 18 |
| OpenMP (6 threads) | 0.30 s | 0.026 s | 82 |
| OpenBLAS (6 threads) | n/a | 0.0032 s | 670 |

**Loop order matters (~13x).** Matrices are stored row-major. `i-j-k` reads B column by column,
so every access lands on a new cache line. `i-k-j` computes `C[i][:] += A[i][k] * B[k][:]`,
reading B and C row by row (sequential memory, auto-vectorized by the compiler).

#### LLM inference: TinyStories 15M
`make run EX=examples/generate.c [ARGS=omp|blas]`

| matvec | tokens/s | time split |
|---|---|---|
| naive, 1 thread | 81 | matvec 96%, attention 2%, other 2% |
| OpenMP, 6 threads | 177 | matvec 89%, attention 4%, other 7% |
| OpenBLAS, 6 threads | 259 | matvec 85%, attention 6%, other 9% |
| OpenBLAS, 1 thread (`OPENBLAS_NUM_THREADS=1`) | 314 | matvec 86%, attention 7%, other 8% |

### Build and run
```bash
make run EX=examples/test_mlp.c   # build & run an example (downloads datasets)
make bench                        # matmul benchmark
make model && make test           # download the model, run all tests
make MODE=debug run EX=...        # -O0 -g + AddressSanitizer
make BLAS=0 ...                   # build without OpenBLAS
```