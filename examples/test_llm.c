#include "include/transformer.h"
#include <stdio.h>

int main(void) {
  Transformer t;
  load_transformer(&t, "models/stories15M.bin");
  print_config(t.config, t.shared_weights);

  Matrix e = t.token_embedding;
  printf("\ntoken_embedding: %d x %d\n", e.rows, e.columns);
  printf("row 0, first 4: %f %f %f %f\n", e.data[0], e.data[1], e.data[2], e.data[3]);

  Matrix r = t.rms_att_0;
  printf("\nrms_att layer 0: %d x %d\n", r.rows, r.columns);
  printf("first 4: %f %f %f %f\n", r.data[0], r.data[1], r.data[2], r.data[3]);

  free_transformer(&t);
  return 0;
}
