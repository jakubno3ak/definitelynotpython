#include "include/transformer.h"
#include <stdio.h>

static void print_layers(const char *name, Matrix *w, int n_layers) {
  Matrix first = w[0], last = w[n_layers - 1];
  printf("%-8s  %d layers, each %4d x %-4d  layer 0: %f %f %f | layer %d: %f %f %f\n",
         name, n_layers, first.rows, first.columns,
         first.data[0], first.data[1], first.data[2],
         n_layers - 1, last.data[0], last.data[1], last.data[2]);
}

int main(void) {
  Transformer t;
  load_transformer(&t, "models/stories15M.bin");
  print_config(t.config, t.shared_weights);
  printf("\n");

  Matrix e = t.token_embedding;
  printf("token_embedding           %d x %d  row 0: %f %f %f\n",
         e.rows, e.columns, e.data[0], e.data[1], e.data[2]);

  print_layers("rms_att", t.rms_att, t.config.n_layers);
  print_layers("wq", t.wq, t.config.n_layers);
  print_layers("wk", t.wk, t.config.n_layers);
  print_layers("wv", t.wv, t.config.n_layers);
  print_layers("wo", t.wo, t.config.n_layers);
  print_layers("rms_ffn", t.rms_ffn, t.config.n_layers);
  print_layers("w1", t.w1, t.config.n_layers);
  print_layers("w2", t.w2, t.config.n_layers);
  print_layers("w3", t.w3, t.config.n_layers);

  Matrix f = t.rms_final;
  printf("rms_final                 %d x %d    %f %f %f\n", f.rows, f.columns, f.data[0], f.data[1], f.data[2]);

  free_transformer(&t);
  return 0;
}
