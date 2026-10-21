#include "ops.h"
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

static int decode(Transformer *t, RunState *state, Tokenizer *tok, int token, int pos,
                  int *n_decode) {
  int generated = 0;
  *n_decode = 0;
  while (1) {
    int next = argmax(state->logits);
    if (next == 1 || next == 2) // BOS separates stories in TinyStories, EOS ends text
      break;
    print_token(tok, token, next);
    generated++;
    if (pos == t->config.seq_len) // KV cache full
      break;
    token = next;
    forward(t, state, token, pos);
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
  double other = seconds - state->t_matvec - state->t_attention;
  fprintf(stderr, "    matvec %.2f s (%.0f%%), attention %.2f s (%.0f%%), other %.2f s (%.0f%%)\n",
          state->t_matvec, 100 * state->t_matvec / seconds, state->t_attention,
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
  int n_decode;
  int generated = decode(&t, &state, &tok, prompt_tokens[n_prompt - 1], n_prompt, &n_decode);
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
