#include "transformer.h"
#include "ops.h"
#include "ops_kernels.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>
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

static Matrix batch_buffer(int capacity, int columns) {
  Matrix m = mat_create(capacity, columns);
  m.rows = 1; // rows in use: one token until forward says more
  return m;
}

RunState run_state_create(Config c) {
  RunState state;
  state.x = batch_buffer(c.seq_len, c.dim);
  state.xnorm = batch_buffer(c.seq_len, c.dim);
  state.q = batch_buffer(c.seq_len, c.dim);
  state.k = batch_buffer(c.seq_len, c.dim);
  state.v = batch_buffer(c.seq_len, c.dim);

  state.att = mat_create(c.n_heads, c.seq_len);
  state.att_out = batch_buffer(c.seq_len, c.dim);
  state.att_proj = batch_buffer(c.seq_len, c.dim);
  state.ffn_a = batch_buffer(c.seq_len, c.hidden_dim);
  state.ffn_b = batch_buffer(c.seq_len, c.hidden_dim);
  state.ffn_out = batch_buffer(c.seq_len, c.dim);
  state.logits = mat_create(1, c.vocab_size);

  state.t_linear = 0;
  state.t_attention = 0;
  state.att_dump = NULL;
  state.n_layers = c.n_layers;
  state.key_cache = malloc(c.n_layers * sizeof(Matrix));
  state.value_cache = malloc(c.n_layers * sizeof(Matrix));
  for (int l = 0; l < c.n_layers; l++) {
    state.key_cache[l] = mat_create(c.seq_len, c.dim);
    state.value_cache[l] = mat_create(c.seq_len, c.dim);
  }
  return state;
}

void run_state_free(RunState *state) {
  mat_free(state->x);
  mat_free(state->xnorm);
  mat_free(state->q);
  mat_free(state->k);
  mat_free(state->v);
  mat_free(state->att);
  mat_free(state->att_out);
  mat_free(state->att_proj);
  mat_free(state->ffn_a);
  mat_free(state->ffn_b);
  mat_free(state->ffn_out);
  mat_free(state->logits);
  for (int l = 0; l < state->n_layers; l++) {
    mat_free(state->key_cache[l]);
    mat_free(state->value_cache[l]);
  }
  free(state->key_cache);
  free(state->value_cache);
}

void embed(Transformer *t, RunState *state, int *tokens, int n) {
  for (int i = 0; i < n; i++) {
    if (tokens[i] < 0 || tokens[i] >= t->config.vocab_size) {
      fprintf(stderr, "Error: token %d outside vocabulary (0..%d)\n", tokens[i],
              t->config.vocab_size - 1);
      exit(1);
    }
    float *row = t->token_embedding.data + tokens[i] * t->config.dim;
    memcpy(state->x.data + i * t->config.dim, row, t->config.dim * sizeof(float));
  }
}

void rmsnorm(Matrix out, Matrix x, Matrix weight) {
  for (int r = 0; r < x.rows; r++) {
    float *xr = x.data + r * x.columns;
    float *outr = out.data + r * out.columns;
    float sum_sq = 0.0f;
    for (int i = 0; i < x.columns; i++)
      sum_sq += xr[i] * xr[i];
    float rms = sqrtf(sum_sq / x.columns + 1e-5f);

    for (int i = 0; i < x.columns; i++)
      outr[i] = xr[i] / rms * weight.data[i];
  }
}

void silu(Matrix m) {
  for (int i = 0; i < mat_size(m); i++)
    m.data[i] = m.data[i] / (1.0f + expf(-m.data[i]));
}

void rope(Matrix q, Matrix k, int pos0, int head_size) {
  for (int r = 0; r < q.rows; r++) {
    float *qr = q.data + r * q.columns;
    float *kr = k.data + r * k.columns;
    int pos = pos0 + r;
    for (int i = 0; i < q.columns; i += 2) {
      int pair = (i % head_size) / 2;
      float theta = 1.0f / powf(10000.0f, 2.0f * pair / head_size);
      float angle = pos * theta;
      float c = cosf(angle), s = sinf(angle);

      float q0 = qr[i], q1 = qr[i + 1];
      qr[i] = q0 * c - q1 * s;
      qr[i + 1] = q0 * s + q1 * c;

      float k0 = kr[i], k1 = kr[i + 1];
      kr[i] = k0 * c - k1 * s;
      kr[i + 1] = k0 * s + k1 * c;
    }
  }
}

void kv_store(RunState *state, int layer, int pos0) {
  int dim = state->k.columns;
  int bytes = state->k.rows * dim * sizeof(float); // all n rows at once
  memcpy(state->key_cache[layer].data + pos0 * dim, state->k.data, bytes);
  memcpy(state->value_cache[layer].data + pos0 * dim, state->v.data, bytes);
}

void attention(Transformer *t, RunState *state, int layer, int pos) {
  int dim = t->config.dim;
  int head_size = dim / t->config.n_heads;
  for (int h = 0; h < t->config.n_heads; h++) {
    float *q_head = state->q.data + h * head_size;
    float *scores = state->att.data + h * t->config.seq_len;
    for (int p = 0; p <= pos; p++) {
      float *k_head = state->key_cache[layer].data + p * dim + h * head_size;
      float dot = 0.0f;
      for (int i = 0; i < head_size; i++)
        dot += q_head[i] * k_head[i];
      scores[p] = dot / sqrtf(head_size);
    }
    kernel_softmax_row(scores, pos + 1);
    if (state->att_dump)
      fwrite(scores, sizeof(float), pos + 1, state->att_dump);

    float *out_head = state->att_out.data + h * head_size;
    memset(out_head, 0, head_size * sizeof(float));
    for (int p = 0; p <= pos; p++) {
      float *v_head = state->value_cache[layer].data + p * dim + h * head_size;
      for (int i = 0; i < head_size; i++)
        out_head[i] += scores[p] * v_head[i];
    }
  }
}

static void timed_linear(RunState *state, Matrix out, Matrix W, Matrix X) {
  double start = omp_get_wtime();
  ops_linear(out, X, W);
  state->t_linear += omp_get_wtime() - start;
}

void attention_block(Transformer *t, RunState *state, int layer, int pos) {
  int head_size = t->config.dim / t->config.n_heads;

  rmsnorm(state->xnorm, state->x, t->rms_att[layer]);
  timed_linear(state, state->q, t->wq[layer], state->xnorm);
  timed_linear(state, state->k, t->wk[layer], state->xnorm);
  timed_linear(state, state->v, t->wv[layer], state->xnorm);
  rope(state->q, state->k, pos, head_size);
  kv_store(state, layer, pos);
  double start = omp_get_wtime();
  attention(t, state, layer, pos);
  state->t_attention += omp_get_wtime() - start;
  timed_linear(state, state->att_proj, t->wo[layer], state->att_out);

  mat_add(state->x, state->att_proj); // residual: x += attention result
}

void mlp_block(Transformer *t, RunState *state, int layer) {
  rmsnorm(state->xnorm, state->x, t->rms_ffn[layer]);
  timed_linear(state, state->ffn_a, t->w1[layer], state->xnorm);
  timed_linear(state, state->ffn_b, t->w3[layer], state->xnorm);
  silu(state->ffn_a);
  for (int i = 0; i < mat_size(state->ffn_a); i++)
    state->ffn_a.data[i] *= state->ffn_b.data[i]; // gate
  timed_linear(state, state->ffn_out, t->w2[layer], state->ffn_a);

  mat_add(state->x, state->ffn_out); // residual: x += MLP result
}

static void set_rows(RunState *state, int n) {
  state->x.rows = n;
  state->xnorm.rows = n;
  state->q.rows = n;
  state->k.rows = n;
  state->v.rows = n;
  state->att_out.rows = n;
  state->att_proj.rows = n;
  state->ffn_a.rows = n;
  state->ffn_b.rows = n;
  state->ffn_out.rows = n;
}

void forward(Transformer *t, RunState *state, int *tokens, int n, int pos0) {
  if (pos0 + n > t->config.seq_len) {
    fprintf(stderr, "Error: positions %d..%d exceed seq_len %d\n", pos0, pos0 + n - 1,
            t->config.seq_len);
    exit(1);
  }
  if (state->att_dump)
    fwrite(tokens, sizeof(int), n, state->att_dump);
  set_rows(state, n);
  embed(t, state, tokens, n);
  for (int l = 0; l < t->config.n_layers; l++) {
    attention_block(t, state, l, pos0);
    mlp_block(t, state, l);
  }
  rmsnorm(state->xnorm, state->x, t->rms_final);
  timed_linear(state, state->logits, t->wcls, state->xnorm);
}

void prefill(Transformer *t, RunState *state, int *tokens, int n) {
  for (int pos = 0; pos < n; pos++)
    forward(t, state, &tokens[pos], 1, pos);
}

int argmax(Matrix logits) {
  int best = 0;
  for (int i = 1; i < mat_size(logits); i++)
    if (logits.data[i] > logits.data[best])
      best = i;
  return best;
}
