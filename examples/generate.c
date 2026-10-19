#include "tokenizer.h"
#include "transformer.h"
#include <stdio.h>

int main(void) {
  Transformer t;
  load_transformer(&t, "models/stories15M.bin");
  Tokenizer tok;
  tokenizer_load(&tok, "models/tokenizer.bin", t.config.vocab_size);
  RunState state = run_state_create(t.config);

  int token = 1; // BOS
  for (int pos = 0; pos < t.config.seq_len; pos++) {
    forward(&t, &state, token, pos);
    int next = argmax(state.logits);
    if (next == 1 || next == 2) // BOS separates stories in TinyStories, EOS ends text
      break;
    print_token(&tok, token, next);
    token = next;
  }
  printf("\n");

  run_state_free(&state);
  tokenizer_free(&tok);
  free_transformer(&t);
  return 0;
}
