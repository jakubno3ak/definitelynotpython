#include "../include/transformer.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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

// n views of rows x cols, stored back to back from start
static Matrix *layer_views(float *start, int n, int rows, int cols) {
  Matrix *views = malloc(n * sizeof(Matrix));
  for (int l = 0; l < n; l++) {
    views[l] = matrix_view(start + l * rows * cols, rows, cols);
  }
  return views;
}

void load_transformer(Transformer *t, const char *path) {
  t->config = read_config(path, &t->shared_weights);

  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd == -1 || fstat(fd, &st) == -1) {
    fprintf(stderr, "Error: could not open model %s\n", path);
    exit(1);
  }
  t->file_size = st.st_size;

  t->data = mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (t->data == MAP_FAILED) {
    fprintf(stderr, "Error: mmap failed for %s\n", path);
    exit(1);
  }

  float *ptr = t->data + sizeof(Config) / sizeof(float); // skip the header
  t->token_embedding = matrix_view(ptr, t->config.vocab_size, t->config.dim);

  ptr += t->config.vocab_size * t->config.dim;

  t->rms_att = layer_views(ptr, t->config.n_layers, 1, t->config.dim);
  ptr += t->config.n_layers * t->config.dim;

  t->wq = layer_views(ptr, t->config.n_layers, t->config.dim, t->config.dim);
  ptr += t->config.n_layers * t->config.dim * t->config.dim;

  t->wk = layer_views(ptr, t->config.n_layers, t->config.dim, t->config.dim);
  ptr += t->config.n_layers * t->config.dim * t->config.dim;

  t->wv = layer_views(ptr, t->config.n_layers, t->config.dim, t->config.dim);
  ptr += t->config.n_layers * t->config.dim * t->config.dim;

  t->wo = layer_views(ptr, t->config.n_layers, t->config.dim, t->config.dim);
  ptr += t->config.n_layers * t->config.dim * t->config.dim;
}

void free_transformer(Transformer *t) {
  free(t->rms_att);
  free(t->wq);
  free(t->wk);
  free(t->wv);
  free(t->wo);
  munmap(t->data, t->file_size);
}
