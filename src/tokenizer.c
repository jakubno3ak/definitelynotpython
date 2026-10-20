#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int tokenizer_lookup(Tokenizer *tok, const char *str) {
  for (int i = 0; i < tok->vocab_size; i++)
    if (strcmp(tok->vocab[i], str) == 0)
      return i;
  return -1; // not in vocabulary
}

int tokenizer_best_pair(Tokenizer *tok, int *tokens, int n, int *merged) {
  int best = -1;
  float best_score = -1e10f;
  char pair[64]; // two tokens of max 27 bytes + '\0'
  for (int i = 0; i < n - 1; i++) {
    snprintf(pair, sizeof(pair), "%s%s", tok->vocab[tokens[i]], tok->vocab[tokens[i + 1]]);
    int id = tokenizer_lookup(tok, pair);
    if (id != -1 && tok->scores[id] > best_score) {
      best = i;
      best_score = tok->scores[id];
      *merged = id;
    }
  }
  return best;
}

int tokenizer_encode(Tokenizer *tok, const char *text, int *tokens) {
  int n = 0;
  for (int i = 0; text[i] != '\0'; i++) {
    char piece[2] = {text[i], '\0'}; // one byte as a 1-char string
    int id = tokenizer_lookup(tok, piece);
    if (id == -1)
      id = (unsigned char)text[i] + 3; // byte token <0x..>: ids 3..258
    tokens[n++] = id;
  }

  while (1) {
    int merged;
    int best = tokenizer_best_pair(tok, tokens, n, &merged);
    if (best == -1)
      break;
    tokens[best] = merged;
    for (int i = best + 1; i < n - 1; i++)
      tokens[i] = tokens[i + 1];
    n--;
  }
  return n;
}

void tokenizer_free(Tokenizer *tok) {
  for (int i = 0; i < tok->vocab_size; i++)
    free(tok->vocab[i]);
  free(tok->vocab);
  free(tok->scores);
}

void print_token(Tokenizer *tok, int prev, int token) {
  const char *piece = tokenizer_decode(tok, token);
  if (prev == 1 && piece[0] == ' ')
    piece++; // story start: drop the leading space
  if (strcmp(piece, "<0x0A>") == 0)
    piece = "\n"; // byte token for a newline
  printf("%s", piece);
  fflush(stdout);
}
