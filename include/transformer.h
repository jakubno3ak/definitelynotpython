#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "matrix.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct {
  int dim;        // length of each token's vector (288)
  int hidden_dim; // inner size of the MLP (768)
  int n_layers;   // number of transformer blocks (6)
  int n_heads;    // attention heads (6)
  int n_kv_heads; // key/value heads, can be fewer than n_heads in bigger models (6)
  int vocab_size; // tokens in the vocabulary (32000)
  int seq_len;    // max context length in tokens (256)
} Config;

typedef struct {
  Config config;
  bool shared_weights;
  Matrix token_embedding; // view, vocab_size x dim
  Matrix *rms_att;        // n_layers views, each 1 x dim
  Matrix *wq;             // n_layers views, each dim x dim
  Matrix *wk;             // n_layers views, each dim x dim
  Matrix *wv;             // n_layers views, each dim x dim
  Matrix *wo;             // n_layers views, each dim x dim
  Matrix *rms_ffn;        // n_layers views, each 1 x dim
  Matrix *w1;             // n_layers views, each hidden_dim x dim
  Matrix *w2;             // n_layers views, each dim x hidden_dim
  Matrix *w3;             // n_layers views, each hidden_dim x dim
  Matrix rms_final;       // view, 1 x dim
  Matrix wcls;            // view, vocab_size x dim (= token_embedding if shared)
  float *data;           // owner of the mmap'ed file
  size_t file_size;
} Transformer;

Config read_config(const char *path, bool *shared_weights);
void print_config(Config c, bool shared_weights);
void load_transformer(Transformer *t, const char *path);
void free_transformer(Transformer *t);

#endif
