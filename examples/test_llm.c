#include "include/transformer.h"
#include <stdio.h>

int main(void) {
  Transformer t;
  load_transformer(&t, "models/stories15M.bin");
  print_config(t.config, t.shared_weights);

  Matrix e = t.token_embedding;
  printf("\ntoken_embedding: %d x %d\n", e.rows, e.columns);
  printf("row 0, first 4: %f %f %f %f\n", e.data[0], e.data[1], e.data[2], e.data[3]);

  free_transformer(&t);
  return 0;
}
