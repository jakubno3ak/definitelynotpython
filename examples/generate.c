#include "ops.h"
#include "ops_kernels.h"
#include "tokenizer.h"
#include "transformer.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int *encode_prompt(Tokenizer *tok, const char *prompt, int *n) {
  char *text = malloc(strlen(prompt) + 2);
  sprintf(text, " %s", prompt); // dummy prefix space, like sentencepiece
  int *tokens = malloc((strlen(text) + 1) * sizeof(int));
  tokens[0] = 1; // BOS
  *n = 1;
  if (prompt[0] != '\0')
    *n += tokenizer_encode(tok, text, tokens + 1);
  free(text);
  return tokens;
}

#define TOP_MAX 10

static void print_top(Tokenizer *tok, Matrix logits, int k) {
  int n = mat_size(logits);
  float *p = malloc(n * sizeof(float));
  memcpy(p, logits.data, n * sizeof(float));
  kernel_softmax_row(p, n); // probabilities; logits stay untouched
  int top[TOP_MAX];
  int found = 0;
  for (int i = 0; i < n; i++) {
    if (found == k && p[i] <= p[top[k - 1]])
      continue;
    int j = found < k ? found++ : k - 1;
    while (j > 0 && p[i] > p[top[j - 1]]) { // insertion: shift weaker ones down
      top[j] = top[j - 1];
      j--;
    }
    top[j] = i;
  }
  fprintf(stderr, "  <-");
  for (int j = 0; j < found; j++)
    fprintf(stderr, " %s %.0f%%%s", tokenizer_decode(tok, top[j]), 100 * p[top[j]],
            j + 1 < found ? " |" : "\n");
  free(p);
}

static int decode(Transformer *t, RunState *state, Tokenizer *tok, int token, int pos, int top_k,
                  int *n_decode) {
  int generated = 0;
  *n_decode = 0;
  while (1) {
    int next = argmax(state->logits);
    if (next == 1 || next == 2) // BOS separates stories in TinyStories, EOS ends text
      break;
    print_token(tok, token, next);
    if (top_k > 0)
      print_top(tok, state->logits, top_k);
    generated++;
    if (pos == t->config.seq_len) // KV cache full
      break;
    token = next;
    forward(t, state, &token, 1, pos);
    (*n_decode)++;
    pos++;
  }
  printf("\n");
  return generated;
}

static void print_stats(Transformer *t, RunState *state, int n_prompt, int n_decode, int generated,
                        double prefill_s, double decode_s) {
  double seconds = prefill_s + decode_s;
  double tok_s = generated / seconds;
  double gb_s = tok_s * t->file_size / 1e9; // all weights are read once per token
  fprintf(stderr, "--- %s: %d tokens in %.2f s: %.1f tokens/s, %.1f GB/s of weights ---\n",
          ops_get_backend_name(), generated, seconds, tok_s, gb_s);
  double other = seconds - state->t_linear - state->t_attention;
  fprintf(stderr, "    linear %.2f s (%.0f%%), attention %.2f s (%.0f%%), other %.2f s (%.0f%%)\n",
          state->t_linear, 100 * state->t_linear / seconds, state->t_attention,
          100 * state->t_attention / seconds, other, 100 * other / seconds);
  fprintf(stderr,
          "    prefill: %3d tokens in %6.1f ms (%.2f ms/token, %.1f tokens/s)  TTFT %.1f ms\n",
          n_prompt, 1000 * prefill_s, 1000 * prefill_s / n_prompt, n_prompt / prefill_s,
          1000 * prefill_s);
  fprintf(stderr,
          "    decode:  %3d tokens in %6.1f ms (%.2f ms/token, %.1f tokens/s)  TPOT %.2f ms\n",
          n_decode, 1000 * decode_s, 1000 * decode_s / n_decode, n_decode / decode_s,
          1000 * decode_s / n_decode);
}

// Usage: generate [naive|omp|blas] [prompt] [attention_dump.bin]
int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "omp") == 0)
    ops_set_backend(BACKEND_OMP);
  if (argc > 1 && strcmp(argv[1], "blas") == 0)
    ops_set_backend(BACKEND_BLAS);
  Transformer t;
  load_transformer(&t, "models/stories15M.bin");
  Tokenizer tok;
  tokenizer_load(&tok, "models/tokenizer.bin", t.config.vocab_size);
  RunState state = run_state_create(t.config);
  if (argc > 3) {
    state.att_dump = fopen(argv[3], "wb");
    if (!state.att_dump) {
      perror(argv[3]);
      return 1;
    }
  }

  int n_prompt;
  int *prompt_tokens = encode_prompt(&tok, argc > 2 ? argv[2] : "", &n_prompt);
  if (n_prompt > t.config.seq_len) {
    fprintf(stderr, "Prompt too long: %d tokens, max %d\n", n_prompt, t.config.seq_len);
    return 1;
  }
  for (int i = 1; i < n_prompt; i++)
    print_token(&tok, prompt_tokens[i - 1], prompt_tokens[i]);

  double start = omp_get_wtime();
  prefill(&t, &state, prompt_tokens, n_prompt);
  double prefill_end = omp_get_wtime();
  int top_k = getenv("TOPK") ? atoi(getenv("TOPK")) : 0;
  if (top_k > TOP_MAX)
    top_k = TOP_MAX;
  int n_decode;
  int generated = decode(&t, &state, &tok, prompt_tokens[n_prompt - 1], n_prompt, top_k, &n_decode);
  double end = omp_get_wtime();

  print_stats(&t, &state, n_prompt, n_decode, generated, prefill_end - start, end - prefill_end);

  if (state.att_dump)
    fclose(state.att_dump);
  free(prompt_tokens);
  run_state_free(&state);
  tokenizer_free(&tok);
  free_transformer(&t);
  return 0;
}
