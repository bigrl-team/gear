#include "gear_errors.h"
#include "index/request.h"

namespace gear::index {

InferenceRequestArray::InferenceRequestArray(int64_t capacity)
    : count(0), capacity(static_cast<size_t>(capacity)),
      src_mem_blk(capacity * sizeof(int32_t)),
      idx_mem_blk(2 * capacity * sizeof(int64_t)) {
  this->src_mem_blk.alloc();
  this->srcs = reinterpret_cast<int32_t *>(this->src_mem_blk.addr);

  this->idx_mem_blk.alloc();
  this->idxs = reinterpret_cast<int64_t *>(this->idx_mem_blk.addr);
  this->tss = this->idxs + capacity;
}

void InferenceRequestArray::insert(size_t sim, int64_t index,
                                   int64_t timestep) {
  this->srcs[count] = static_cast<int32_t>(sim);
  this->idxs[count] = index;
  this->tss[count] = timestep;
  ++this->count;
}

WeightUpdateRequestArray::WeightUpdateRequestArray(int64_t capacity)
    : count(0), capacity(static_cast<size_t>(capacity)),
      update_mem_blk(capacity * sizeof(bool)),
      ts_mem_blk(capacity * sizeof(size_t)),
      weight_mem_blk(capacity * sizeof(float)) {
  this->update_mem_blk.alloc();
  this->updated = reinterpret_cast<bool *>(this->update_mem_blk.addr);

  this->ts_mem_blk.alloc();
  this->timesteps = reinterpret_cast<int64_t *>(this->ts_mem_blk.addr);

  this->weight_mem_blk.alloc();
  this->weights = reinterpret_cast<float *>(this->weight_mem_blk.addr);
}

void WeightUpdateRequestArray::update(size_t index, int64_t timestep,
                                      float weight) {
  if (!updated[index]) {
    count += 1;
    updated[index] = true;
    list.push_back(index);
  }

  timesteps[index] = timestep;
  weights[index] = weight;
}

void WeightUpdateRequestArray::continuous(int64_t &num, Int64Span &idxs,
                                          Float32Span &weights) {
  num = this->count;
  GEAR_COND_EXCEPT(idxs.size >= 2 * num, std::runtime_error,
                   "Index output array for continuous weight update array has "
                   "less space than required");
  GEAR_COND_EXCEPT(
      weights.size >= num, std::runtime_error,
      "Weights output array for continuous weight update array has "
      "less space than required");

  for (size_t i = 0; i < this->count; ++i) {
    size_t idx = this->list[i];

    idxs[i] = idx;
    idxs[this->capacity + i] = this->timesteps[idx];
    weights[i] = this->weights[idx];

    updated[idx] = false;
  }
  this->count = 0;
  this->list.clear();
}

CachedRequest::CachedRequest(int capacity) : iarr(capacity), warr(capacity) {}

} // namespace gear::index