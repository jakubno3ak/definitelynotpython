# Architecture

## One forward pass (one token)

```
token id → embedding (row of a 32000 × 288 table) → x [288]

block l (× 6):
  attention_block
    xnorm    = (x / rms(x)) * rms_att[l]               [288]
    q, k, v  = wq·xnorm, wk·xnorm, wv·xnorm             [288] each
    RoPE(q, k)                                          rotate by position
    k, v     → KV cache row pos
    att_out  = attention(q, cached k, v)                6 heads × 48 = [288]
    x        = x + wo · att_out                         [288]
  mlp_block
    xnorm    = (x / rms(x)) * rms_ffn[l]               [288]
    a, b     = w1·xnorm, w3·xnorm                       [768] each
    h        = SiLU(a) * b                              [768]
    x        = x + w2 · h                               [288]

xnorm  = (x / rms(x)) * rms_final                      [288]
logits = wcls · xnorm                                   [32000], argmax → next token
```

`·` is matrix × vector (`ops_matvec`), `*` is element-wise. `x` (the residual stream) is never
normalized itself; only copies of it go into attention and the MLP.

## Where things live

| file | what |
|---|---|
| `include/matrix.h`, `src/matrix.c` | `Matrix` (data, rows, columns, `owns_data`), views, basic ops |
| `include/ops.h`, `src/ops.c` | backend dispatcher (naive / OpenMP / BLAS) |
| `src/backends/` | `ops_naive.c`, `ops_omp.c`, `ops_blas.c` |
| `include/transformer.h`, `src/transformer.c` | config, mmap loading, `RunState` buffers, attention, MLP, `forward` |
| `include/tokenizer.h`, `src/tokenizer.c` | vocabulary from `tokenizer.bin`, decoding |
| `examples/generate.c` | generation loop + tokens/s and time split |

## Memory

- Weights are read-only views into the `mmap`'ed model file; only `Transformer` owns that memory.
- All per-token work happens in preallocated `RunState` buffers (no allocation per token).
- KV cache: `n_layers × seq_len × dim` floats for keys and the same for values.
