#include "check.h"
#include "matrix.h"

int main(void) {
  Matrix m = mat_create(3, 4);
  CHECK(m.rows == 3 && m.columns == 4);
  CHECK(m.owns_data);
  for (int i = 0; i < mat_size(m); i++)
    CHECK(m.data[i] == 0.0f);

  float block[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  Matrix v = mat_view(block + 4, 2, 3);
  CHECK(!v.owns_data);
  CHECK(v.data == block + 4);
  CHECK(v.data[0] == 5.0f && v.data[1 * 3 + 2] == 10.0f);
  mat_free(v); // must not free the stack array (sanitizer would catch it)
  CHECK(block[4] == 5.0f);

  mat_free(m);
  return report("test_matrix");
}
