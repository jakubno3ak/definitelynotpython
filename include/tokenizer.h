#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
  char **vocab;  // vocab_size strings, '\0'-terminated
  float *scores; // needed later for encoding (BPE)
  int vocab_size;
  int max_token_length;
} Tokenizer;

void tokenizer_load(Tokenizer *tok, const char *path, int vocab_size);
const char *tokenizer_decode(Tokenizer *tok, int token);
void tokenizer_free(Tokenizer *tok);
void print_token(Tokenizer *tok, int prev, int token);

#endif
