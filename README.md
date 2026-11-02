# definitelynotpython
[![CI](https://github.com/jakubno3ak/definitelynotpython/actions/workflows/ci.yml/badge.svg)](https://github.com/jakubno3ak/definitelynotpython/actions/workflows/ci.yml)

Not Python at all. Well…

ML from scratch in C: kernels, memory and a Llama-style LLM, with benchmarks and experiments
along the way. Python bindings eventually, because after writing everything in C, it would be
a shame not to call it from Python anyway.

## Backends
One dispatcher ([ops.h](include/ops.h)) switches the compute engine at runtime:
- [naive](src/backends/ops_naive.c): plain single-threaded C loops (baseline)
- [OpenMP](src/backends/ops_omp.c): multi-threaded CPU
- [OpenBLAS](src/backends/ops_blas.c): optimized BLAS (`sgemm`, `sgemv`)
- CUDA: planned

## Results
Intel i5-9600K (6 cores), `-O3 -march=native` (max optimizations + AVX2).

### Matrix multiplication (1024×1024, float32)
Best of 5 runs, `make bench`.

| Backend | Before (i-j-k loops) | Now | GFLOP/s (now) |
|---|---|---|---|
| Naive (1 thread) | 1.58 s | 0.116 s | 18 |
| OpenMP (6 threads) | 0.30 s | 0.026 s | 82 |
| OpenBLAS (6 threads) | n/a | 0.0032 s | 670 |

**Loop order matters (~13x).** Matrices are stored row-major. `i-j-k` reads B column by column,
so every access lands on a new cache line. `i-k-j` computes `C[i][:] += A[i][k] * B[k][:]`,
reading B and C row by row (sequential memory, auto-vectorized by the compiler).

### LLM inference: TinyStories 15M
`make run EX=examples/generate.c ARGS='blas "Once upon a time"'`. TTFT for a 40-token prompt,
newer rows are the median of 5 runs.

| Step | Backend | TTFT | TPOT | Decode tokens/s | Time split |
|---|---|---|---|---|---|
| Matvec per token | naive, 1 thread | — | 12.3 ms | 81 | matvec 96%, attention 2%, other 2% |
| | OpenMP, 6 threads | — | 5.6 ms | 177 | matvec 89%, attention 4%, other 7% |
| | OpenBLAS, 6 threads | — | 3.9 ms | 259 | matvec 85%, attention 6%, other 9% |
| | OpenBLAS, 1 thread (`OPENBLAS_NUM_THREADS=1`) | — | 3.2 ms | 314 | matvec 86%, attention 7%, other 8% |
| Unified forward, prompt in one pass | naive, 1 thread | 189 ms | 12.1 ms | 83 | linear 96%, attention 2%, other 2% |
| | OpenMP, 6 threads | 36 ms | 2.7 ms | 367 | linear 83%, attention 8%, other 10% |
| | OpenBLAS | 6.6 ms | 2.3 ms | 437 | linear 79%, attention 9%, other 12% |

Before prompts were processed in one pass, prefill cost the same per token as decode, so TTFT
would have been about 40 × TPOT: ~485 ms (naive), ~109 ms (OpenMP), ~92 ms (OpenBLAS).

**Prefill vs decode.** The prompt goes through the model in one pass (`X · Wᵀ` with `sgemm`),
so every weight is read once for all 40 tokens. Decode can't: each new token depends on the
previous one. Only the last prompt token goes through the 32k-word output layer, which alone
explains the 2.5x on naive.

## Quickstart
```bash
sudo apt-get install libopenblas-dev
make model                                                    # TinyStories 15M + tokenizer
make run EX=examples/generate.c ARGS='blas "The dragon was"'  # continue a prompt, print timings
make test                                                     # tests (CI also runs debug+ASan and no-BLAS)
```

More: `make bench` (matmul benchmark), `make MODE=debug ...` (AddressSanitizer/UBSan),
`make BLAS=0 ...` (without OpenBLAS), `make` (all targets).

## What's inside
- **Transformer forward pass in C:** RMSNorm, RoPE, multi-head attention, SwiGLU MLP, KV cache.
- **Zero-copy model loading:** weights are views into an `mmap`'ed file.
- **BPE tokenizer:** encodes text to the same token ids as sentencepiece (tested).
- **Batched prefill + KV cache decode:** one forward pass for the whole prompt, TTFT/TPOT reported.
- **Tested against independent references:** raw file bytes and a separate Python implementation.
- **Classic ML:** linear and logistic regression, MLP with manual backprop ([docs/classic](docs/classic/)).
