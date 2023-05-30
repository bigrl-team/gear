#pragma once

#define GEAR_COND_EXCEPT(EXPECTED_TRUE_COND, EXCEPT_TYPE, EXCEPT_MSG)          \
  {                                                                            \
    if (!(EXPECTED_TRUE_COND)) {                                               \
      throw EXCEPT_TYPE(EXCEPT_MSG);                                           \
    }                                                                          \
  }
