#include "check.h"
#include "matrix.h"
#include <stdlib.h>

// Row i holds X = [i, 2i], y = i, so after shuffling every row must still satisfy X = [y, 2y]
// and every original row must appear exactly once.
static void test_shuffle_split(void) {
  int n = 767;
  Matrix X = create_matrix(n, 2), y = create_matrix(n, 1);
  for (int i = 0; i < n; i++) {
    X.data[i * 2] = i;
    X.data[i * 2 + 1] = 2 * i;
    y.data[i] = i;
  }

  srand(662); // the seed that made the old `rand() & (i + 1)` read past the end
  DataSplit s = train_test_split(&X, &y, 0.2, true);
  CHECK(s.X_train.rows + s.X_test.rows == n);

  int *seen = calloc(n, sizeof(int));
  Matrix parts_X[] = {s.X_train, s.X_test}, parts_y[] = {s.y_train, s.y_test};
  for (int p = 0; p < 2; p++) {
    for (int r = 0; r < parts_X[p].rows; r++) {
      float id = parts_y[p].data[r];
      CHECK(parts_X[p].data[r * 2] == id && parts_X[p].data[r * 2 + 1] == 2 * id);
      seen[(int)id]++;
    }
  }
  for (int i = 0; i < n; i++)
    CHECK(seen[i] == 1);

  int moved = 0;
  for (int r = 0; r < s.y_train.rows; r++)
    moved += s.y_train.data[r] != r;
  CHECK(moved > n / 2); // actually shuffled

  free(seen);
  free_matrix(X);
  free_matrix(y);
  free_matrix(s.X_train);
  free_matrix(s.y_train);
  free_matrix(s.X_test);
  free_matrix(s.y_test);
}

static void test_minmax(void) {
  Matrix train = create_matrix(3, 2), test = create_matrix(1, 2);
  float t[] = {1, 5, 2, 5, 3, 5}; // column 1 is constant
  for (int i = 0; i < 6; i++)
    train.data[i] = t[i];
  test.data[0] = 4; // outside the training range
  test.data[1] = 5;

  MinMax s = minmax_fit(train);
  minmax_apply(s, &train);
  minmax_apply(s, &test);

  CHECK(train.data[0] == 0.0f && train.data[2] == 0.5f && train.data[4] == 1.0f);
  CHECK(train.data[1] == 0.0f); // constant column: range 1, value - min = 0
  CHECK(test.data[0] == 1.5f);  // test data keeps the train scale

  minmax_free(s);
  free_matrix(train);
  free_matrix(test);
}

int main(void) {
  test_shuffle_split();
  test_minmax();
  return report("test_data");
}
