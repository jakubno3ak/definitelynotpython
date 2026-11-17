#include "ops.h"
#include "tokenizer.h"
#include "transformer.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// One timed run: prefill the prompt, then decode n_decode tokens greedily.
static void one_run(Transformer *t, RunState *state, int *prompt, int n_prompt, int n_decode,
                    double *prefill_s, double *decode_s) {
  double start = omp_get_wtime();
  prefill(t, state, prompt, n_prompt);
  double mid = omp_get_wtime();
  int token = argmax(state->logits);
  for (int pos = n_prompt; pos < n_prompt + n_decode; pos++) {
    forward(t, state, &token, 1, pos);
    token = argmax(state->logits);
  }
  double end = omp_get_wtime();
  *prefill_s = mid - start;
  *decode_s = end - mid;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double median(double *v, int n) {
  qsort(v, n, sizeof(double), cmp_double);
  return v[n / 2];
}

static double stddev(double *v, int n) {
  double mean = 0.0, var = 0.0;
  for (int i = 0; i < n; i++)
    mean += v[i] / n;
  for (int i = 0; i < n; i++)
    var += (v[i] - mean) * (v[i] - mean) / (n - 1);
  return sqrt(var);
}

// Usage: llm_bench [naive|omp|blas] [prompt]   Env: MODEL, RUNS (5), DECODE (128)
int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "omp") == 0)
    ops_set_backend(BACKEND_OMP);
  if (argc > 1 && strcmp(argv[1], "blas") == 0)
    ops_set_backend(BACKEND_BLAS);
  const char *model = getenv("MODEL") ? getenv("MODEL") : "models/stories15M.bin";
  int runs = getenv("RUNS") ? atoi(getenv("RUNS")) : 5;
  int n_decode = getenv("DECODE") ? atoi(getenv("DECODE")) : 128;
  const char *prompt = argc > 2 ? argv[2] : "The dragon was";

  Transformer t;
  load_transformer(&t, model);
  Tokenizer tok;
  tokenizer_load(&tok, "models/tokenizer.bin", t.config.vocab_size);
  RunState state = run_state_create(t.config);

  char *text = malloc(strlen(prompt) + 2);
  sprintf(text, " %s", prompt); // dummy prefix space, like sentencepiece
  int *tokens = malloc((strlen(text) + 1) * sizeof(int));
  tokens[0] = 1; // BOS
  int n_prompt = 1 + tokenizer_encode(&tok, text, tokens + 1);
  if (n_prompt + n_decode > t.config.seq_len) {
    fprintf(stderr, "prompt + DECODE exceeds seq_len %d\n", t.config.seq_len);
    return 1;
  }

  double *prefill_s = malloc(runs * sizeof(double));
  double *decode_s = malloc(runs * sizeof(double));
  double warm_p, warm_d;
  one_run(&t, &state, tokens, n_prompt, n_decode, &warm_p, &warm_d); // page cache, thread pools
  for (int r = 0; r < runs; r++)
    one_run(&t, &state, tokens, n_prompt, n_decode, &prefill_s[r], &decode_s[r]);

  for (int r = 0; r < runs; r++)
    decode_s[r] /= n_decode; // per token
  double ttft = median(prefill_s, runs), ttft_sd = stddev(prefill_s, runs);
  double tpot = median(decode_s, runs), tpot_sd = stddev(decode_s, runs);
  printf("| %s | %s | %d | %.1f ± %.1f ms | %.2f ± %.2f ms | %.0f | %.1f |\n",
         strrchr(model, '/') + 1, ops_get_backend_name(), n_prompt, 1000 * ttft, 1000 * ttft_sd,
         1000 * tpot, 1000 * tpot_sd, 1 / tpot, t.file_size / tpot / 1e9);

  free(prefill_s);
  free(decode_s);
  free(tokens);
  free(text);
  run_state_free(&state);
  tokenizer_free(&tok);
  free_transformer(&t);
  return 0;
}
