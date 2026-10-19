#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>

static void read_or_die(void *dst, size_t size, FILE *f) {
  if (fread(dst, size, 1, f) != 1) {
    fprintf(stderr, "Error: tokenizer file ended too early\n");
    exit(1);
  }
}

void tokenizer_load(Tokenizer *tok, const char *path, int vocab_size) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) {
    fprintf(stderr, "Error: could not open tokenizer %s\n", path);
    exit(1);
  }
  tok->vocab_size = vocab_size;
  tok->vocab = malloc(vocab_size * sizeof(char *));
  tok->scores = malloc(vocab_size * sizeof(float));
  read_or_die(&tok->max_token_length, sizeof(int), f);

  for (int i = 0; i < vocab_size; i++) {
    int len;
    read_or_die(&tok->scores[i], sizeof(float), f);
    read_or_die(&len, sizeof(int), f);
    tok->vocab[i] = malloc(len + 1);
    read_or_die(tok->vocab[i], len, f);
    tok->vocab[i][len] = '\0';
  }
  fclose(f);
}

const char *tokenizer_decode(Tokenizer *tok, int token) {
  if (token < 0 || token >= tok->vocab_size) {
    fprintf(stderr, "Error: token %d outside vocabulary (0..%d)\n", token, tok->vocab_size - 1);
    exit(1);
  }
  return tok->vocab[token];
}

void tokenizer_free(Tokenizer *tok) {
  for (int i = 0; i < tok->vocab_size; i++)
    free(tok->vocab[i]);
  free(tok->vocab);
  free(tok->scores);
}
