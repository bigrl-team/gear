#pragma once

#include <cassert>
#include <cstdio>

#define GEAR_ASSERT(B, X, ...)                                                 \
  {                                                                            \
    if (!(B)) {                                                                \
      fprintf(stdout, X, ##__VA_ARGS__);                                       \
      fflush(stdout);                                                          \
      assert(0);                                                               \
    }                                                                          \
  }

#define GEAR_ERROR(X, ...)                                                     \
  {                                                                            \
    fprintf(stdout, X, ##__VA_ARGS__);                                         \
    fflush(stdout);                                                            \
  }

#ifdef GEAR_VERBOSE_DEBUG_ON
#define GEAR_DEBUG_PRINT(X, ...)                                               \
  {                                                                            \
    fprintf(stdout, X, ##__VA_ARGS__);                                         \
    fflush(stdout);                                                            \
  }
#else
#define GEAR_DEBUG_PRINT(X, ...)
#endif
