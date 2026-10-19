#include "tokenizer.h"
#include "transformer.h"
#include <omp.h>
#include <stdio.h>

int main(void) {
  Transformer t;
  load_transformer(&t, "models/stories15M.bin");
  Tokenizer tok;
  tokenizer_load(&tok, "models/tokenizer.bin", t.config.vocab_size);
  RunState state = run_state_create(t.config);

  int token = 1; // BOS
  int generated = 0;
  double start = omp_get_wtime();
  for (int pos = 0; pos < t.config.seq_len; pos++) {
    forward(&t, &state, token, pos);
    int next = argmax(state.logits);
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
  fprintf(stderr, "--- %d tokens in %.2f s: %.1f tokens/s, %.1f GB/s of weights ---\n", generated,
          seconds, tok_s, gb_s);

  run_state_free(&state);
  tokenizer_free(&tok);
  free_transformer(&t);
  return 0;
}
