#pragma once

#include <atomic>

#include "common/align.h"

typedef struct GEAR_STRUCT_ALIGN(64) SimulStatus {
  std::atomic<bool> ownership; // 0 - simulator, 1 - model

  // guarded by ownership;
  bool terminated = false;
  bool alloc = false;
  bool valid = false;
  int64_t tindex;
  int64_t timestep;
  double weight;
} sim_status_t;