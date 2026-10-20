#ifndef TOKENIZER_H
#define TOKENIZER_H

typedef struct {
  char **vocab;  // vocab_size strings, '\0'-terminated
  float *scores; // BPE merge priority: higher = merged earlier
  int vocab_size;
  int max_token_length;
} Tokenizer;

void tokenizer_load(Tokenizer *tok, const char *path, int vocab_size);
const char *tokenizer_decode(Tokenizer *tok, int token);
int tokenizer_lookup(Tokenizer *tok, const char *str);
int tokenizer_best_pair(Tokenizer *tok, int *tokens, int n, int *merged);
int tokenizer_encode(Tokenizer *tok, const char *text, int *tokens);
void tokenizer_free(Tokenizer *tok);
void print_token(Tokenizer *tok, int prev, int token);

#endif
