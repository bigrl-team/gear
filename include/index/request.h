#pragma once

#include <vector>

#include "common/span.h"
#include "memory/cuda_uvm.h"
using gear::common::Float32Span;
using gear::common::Int64Span;

namespace gear::index {

typedef struct TrajectoryGeneratorRecord {
  bool active;
  float timestamp;
} record_t;

struct InferenceRequestArray {
  size_t count;
  size_t capacity;

  gear::memory::CudaUVMemory src_mem_blk;
  int32_t *srcs;

  gear::memory::CudaUVMemory idx_mem_blk;
  int64_t *idxs;
  int64_t *tss;

  InferenceRequestArray(int64_t capacity);

  void insert(size_t sim, int64_t index, int64_t timestep);
};

struct WeightUpdateRequestArray {
  size_t count;
  size_t capacity;

  std::vector<size_t> list;

  gear::memory::CudaUVMemory update_mem_blk;
  bool *updated;

  gear::memory::CudaUVMemory ts_mem_blk;
  int64_t *timesteps;

  gear::memory::CudaUVMemory weight_mem_blk;
  float *weights;

  WeightUpdateRequestArray(int64_t capacity);

  void update(size_t index, int64_t timestep, float weight);

  void continuous(int64_t &num, Int64Span &idxs, Float32Span &weights);
};

struct CachedRequest {
  size_t capacity;

  InferenceRequestArray iarr;
  WeightUpdateRequestArray warr;

  CachedRequest(int capacity);
};

} // namespace gear::index