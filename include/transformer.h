#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <stdbool.h>

typedef struct {
  int dim;        // length of each token's vector (288)
  int hidden_dim; // inner size of the MLP (768)
  int n_layers;   // number of transformer blocks (6)
  int n_heads;    // attention heads (6)
  int n_kv_heads; // key/value heads, can be fewer than n_heads in bigger models (6)
  int vocab_size; // tokens in the vocabulary (32000)
  int seq_len;    // max context length in tokens (256)
} Config;


Config read_config(const char *path, bool *shared_weights);
void print_config(Config c, bool shared_weights);

#endif
