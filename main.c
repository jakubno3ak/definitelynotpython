#include "include/linear_regression.h"
#include "include/matrix.h"
#include "include/ops.h"
#include <stdlib.h>
#include <time.h>

int main() {
  srand(time(NULL));

  ops_set_backend(BACKEND_OMP);

  Matrix X = create_matrix(3, 1);
  X.data[0] = 99.9;

  print_matrix(X);
  mat_apply(&X, sigmoid);
  print_matrix(X);

  free_matrix(X);
  return 0;
}