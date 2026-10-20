#include "ops.h"
#include "tokenizer.h"
#include "transformer.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

  const char *prompt = argc > 2 ? argv[2] : "";
  char *text = malloc(strlen(prompt) + 2);
  sprintf(text, " %s", prompt); // dummy prefix space, like sentencepiece
  int *prompt_tokens = malloc((strlen(text) + 1) * sizeof(int));
  prompt_tokens[0] = 1; // BOS
  int n_prompt = 1;
  if (prompt[0] != '\0')
    n_prompt += tokenizer_encode(&tok, text, prompt_tokens + 1);

  int token = prompt_tokens[0];
  int generated = 0;
  double start = omp_get_wtime();
  for (int pos = 0; pos < t.config.seq_len; pos++) {
    if (state.att_dump)
      fwrite(&token, sizeof(int), 1, state.att_dump);
    forward(&t, &state, token, pos);
    int next = pos + 1 < n_prompt ? prompt_tokens[pos + 1] : argmax(state.logits);
    if (next == 1 || next == 2) // BOS separates stories in TinyStories, EOS ends text
      break;
    print_token(&tok, token, next);
    token = next;
    generated++;
  }
  printf("\n");

  double seconds = omp_get_wtime() - start;
  double tok_s = generated / seconds;
  double gb_s = tok_s * t.file_size / 1e9; // all weights are read once per token
  fprintf(stderr, "--- %s: %d tokens in %.2f s: %.1f tokens/s, %.1f GB/s of weights ---\n",
          ops_get_backend_name(), generated, seconds, tok_s, gb_s);
  double other = seconds - state.t_matvec - state.t_attention;
  fprintf(stderr, "    matvec %.2f s (%.0f%%), attention %.2f s (%.0f%%), other %.2f s (%.0f%%)\n",
          state.t_matvec, 100 * state.t_matvec / seconds, state.t_attention,
          100 * state.t_attention / seconds, other, 100 * other / seconds);

  if (state.att_dump)
    fclose(state.att_dump);
  free(text);
  free(prompt_tokens);
  run_state_free(&state);
  tokenizer_free(&tok);
  free_transformer(&t);
  return 0;
}
