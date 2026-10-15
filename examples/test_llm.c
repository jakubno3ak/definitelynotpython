#include "include/transformer.h"
#include <stdio.h>

int main(void) {
  Transformer t;
  load_transformer(&t, "models/stories15M.bin");
  print_config(t.config, t.shared_weights);

  Matrix e = t.token_embedding;
  printf("\ntoken_embedding: %d x %d\n", e.rows, e.columns);
  printf("row 0, first 4: %f %f %f %f\n", e.data[0], e.data[1], e.data[2], e.data[3]);

  printf("\n");
  for (int l = 0; l < t.config.n_layers; l++) {
    Matrix r = t.rms_att[l];
    printf("rms_att layer %d (%d x %d): %f %f %f %f\n", l, r.rows, r.columns,
           r.data[0], r.data[1], r.data[2], r.data[3]);
  }

  printf("\n");
  for (int l = 0; l < t.config.n_layers; l++) {
    Matrix w = t.wq[l];
    printf("wq layer %d (%d x %d), row 0: %f %f %f %f\n", l, w.rows, w.columns,
           w.data[0], w.data[1], w.data[2], w.data[3]);
  }
  float *row1 = t.wq[0].data + t.wq[0].columns;
  printf("wq layer 0, row 1: %f %f %f %f\n", row1[0], row1[1], row1[2], row1[3]);

  free_transformer(&t);
  return 0;
}
