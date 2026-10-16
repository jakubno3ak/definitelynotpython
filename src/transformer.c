#include "transformer.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    views[l] = mat_view(start + l * rows * cols, rows, cols);
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
  t->token_embedding = mat_view(ptr, t->config.vocab_size, t->config.dim);

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

  t->rms_ffn = layer_views(ptr, t->config.n_layers, 1, t->config.dim);
  ptr += t->config.n_layers * t->config.dim;

  t->w1 = layer_views(ptr, t->config.n_layers, t->config.hidden_dim, t->config.dim);
  ptr += t->config.n_layers * t->config.hidden_dim * t->config.dim;

  t->w2 = layer_views(ptr, t->config.n_layers, t->config.dim, t->config.hidden_dim);
  ptr += t->config.n_layers * t->config.dim * t->config.hidden_dim;

  t->w3 = layer_views(ptr, t->config.n_layers, t->config.hidden_dim, t->config.dim);
  ptr += t->config.n_layers * t->config.hidden_dim * t->config.dim;

  t->rms_final = mat_view(ptr, 1, t->config.dim);
  ptr += t->config.dim;

  ptr += t->config.seq_len * (t->config.dim / t->config.n_heads); // skip unused freq_cis

  t->wcls =
      t->shared_weights ? t->token_embedding : mat_view(ptr, t->config.vocab_size, t->config.dim);
  if (!t->shared_weights)
    ptr += t->config.vocab_size * t->config.dim;

  float *end = t->data + t->file_size / sizeof(float);
  if (ptr != end) {
    fprintf(stderr, "Error: weights end at float %ld, file has %ld floats\n", (long)(ptr - t->data),
            (long)(end - t->data));
    exit(1);
  }
}

void free_transformer(Transformer *t) {
  free(t->rms_att);
  free(t->wq);
  free(t->wk);
  free(t->wv);
  free(t->wo);
  free(t->rms_ffn);
  free(t->w1);
  free(t->w2);
  free(t->w3);
  munmap(t->data, t->file_size);
}

RunState run_state_create(Config c) {
  RunState state;
  state.x = mat_create(1, c.dim);
  state.xnorm = mat_create(1, c.dim);
  state.q = mat_create(1, c.dim);
  state.k = mat_create(1, c.dim);
  state.v = mat_create(1, c.dim);
  return state;
}

void run_state_free(RunState *state) {
  mat_free(state->x);
  mat_free(state->xnorm);
  mat_free(state->q);
  mat_free(state->k);
  mat_free(state->v);
}

void embed(Transformer *t, RunState *state, int token) {
  if (token < 0 || token >= t->config.vocab_size) {
    fprintf(stderr, "Error: token %d outside vocabulary (0..%d)\n", token,
            t->config.vocab_size - 1);
    exit(1);
  }
  float *row = t->token_embedding.data + token * t->config.dim;
  memcpy(state->x.data, row, t->config.dim * sizeof(float));
}

void rmsnorm(Matrix out, Matrix x, Matrix weight) {
  float sum_sq = 0.0f;
  for (int i = 0; i < x.columns; i++)
    sum_sq += x.data[i] * x.data[i];
  float rms = sqrtf(sum_sq / x.columns + 1e-5f);

  for (int i = 0; i < x.columns; i++)
    out.data[i] = x.data[i] / rms * weight.data[i];
}
