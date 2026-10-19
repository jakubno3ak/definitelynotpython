#include "check.h"
#include "tokenizer.h"
#include <string.h>

int main(void) {
  Tokenizer tok;
  tokenizer_load(&tok, "models/tokenizer.bin", 32000);

  CHECK(tok.max_token_length == 27);
  CHECK(strcmp(tokenizer_decode(&tok, 0), "<unk>") == 0);
  CHECK(strcmp(tokenizer_decode(&tok, 1), "\n<s>\n") == 0); // BOS
  CHECK(strcmp(tokenizer_decode(&tok, 263), " a") == 0);
  CHECK(strcmp(tokenizer_decode(&tok, 9038), " Once") == 0);
  CHECK_NEAR(tok.scores[263], -4.0f, 1e-6f);

  tokenizer_free(&tok);
  return report("test_tokenizer");
}
