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

  CHECK(tokenizer_lookup(&tok, " Once") == 9038);
  CHECK(tokenizer_lookup(&tok, "ce") == 346);
  CHECK(tokenizer_lookup(&tok, " drag") == 8338);
  CHECK(tokenizer_lookup(&tok, "nce") == -1);

  int chars[5] = {29871, 29949, 29876, 29883, 29872}; // ' ', 'O', 'n', 'c', 'e'
  int merged;
  CHECK(tokenizer_best_pair(&tok, chars, 5, &merged) == 3 && merged == 346); // 'c'+'e' -> "ce"
  int one[1] = {9038};
  CHECK(tokenizer_best_pair(&tok, one, 1, &merged) == -1); // no pairs left

  int ids[32]; // >= bytes of the longest test text: encode writes one id per byte first
  int n = tokenizer_encode(&tok, " Once", ids);
  CHECK(n == 1 && ids[0] == 9038);
  n = tokenizer_encode(&tok, " Once upon a time", ids);
  CHECK(n == 4 && ids[0] == 9038 && ids[1] == 2501 && ids[2] == 263 && ids[3] == 931);
  n = tokenizer_encode(&tok, " The dragon was", ids);
  CHECK(n == 4 && ids[0] == 450 && ids[1] == 8338 && ids[2] == 265 && ids[3] == 471);

  // expected ids from sentencepiece + tokenizer.model (notes/code/sp_reference.py)
  int quotes[] = {365, 2354, 29915, 29879, 11203, 1497, 29892, 376, 10994, 3850};
  n = tokenizer_encode(&tok, " Lily's dog said, \"Hello!\"", ids);
  CHECK(n == 10 && memcmp(ids, quotes, sizeof(quotes)) == 0);
  int digits[] = {739, 471, 29871, 29906, 29900, 29906, 29946, 29889};
  n = tokenizer_encode(&tok, " It was 2024.", ids);
  CHECK(n == 8 && memcmp(ids, digits, sizeof(digits)) == 0);
  int spaces[] = {263, 29871, 289};
  n = tokenizer_encode(&tok, " a  b", ids);
  CHECK(n == 3 && memcmp(ids, spaces, sizeof(spaces)) == 0);
  n = tokenizer_encode(&tok, " Hi\nthere", ids);
  CHECK(n == 3 && ids[0] == 6324 && ids[1] == 13 && ids[2] == 12711); // '\n' -> <0x0A>

  tokenizer_free(&tok);
  return report("test_tokenizer");
}
