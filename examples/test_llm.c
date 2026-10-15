#include "include/transformer.h"
#include <stdio.h>

int main(void) {
  bool shared_weights;
  Config config = read_config("models/stories15M.bin", &shared_weights);
  print_config(config, shared_weights);
  return 0;
}
