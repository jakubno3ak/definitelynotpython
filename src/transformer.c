#include "../include/transformer.h"
#include <stdio.h>
#include <stdlib.h>

Config read_config(const char *path, bool *shared_weights) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "Error: could not open model %s\n", path);
    exit(1);
  }

  Config c;
  if (fread(&c, sizeof(Config), 1, f) != 1) {
    fprintf(stderr, "Error: could not read header of %s\n", path);
    exit(1);
  }
  fclose(f);

  // llama2.c convention: a negative vocab_size means the output matrix is stored
  // separately; positive means the embedding table is reused (weight tying).
  *shared_weights = c.vocab_size > 0;
  c.vocab_size = abs(c.vocab_size);
  return c;
}

void print_config(Config c, bool shared_weights) {
  printf("dim:            %d\n", c.dim);
  printf("hidden_dim:     %d\n", c.hidden_dim);
  printf("n_layers:       %d\n", c.n_layers);
  printf("n_heads:        %d\n", c.n_heads);
  printf("n_kv_heads:     %d\n", c.n_kv_heads);
  printf("vocab_size:     %d\n", c.vocab_size);
  printf("seq_len:        %d\n", c.seq_len);
  printf("shared_weights: %s\n", shared_weights ? "yes" : "no");
}
