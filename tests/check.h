#ifndef CHECK_H
#define CHECK_H

#include <math.h>
#include <stdio.h>

static int failures = 0;

#define CHECK(cond)                                                                                \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                       \
      failures++;                                                                                  \
    }                                                                                              \
  } while (0)

#define CHECK_NEAR(a, b, tol) CHECK(fabsf((a) - (b)) <= (tol))

static int report(const char *name) {
  if (failures == 0)
    printf("PASS %s\n", name);
  return failures != 0;
}

#endif
