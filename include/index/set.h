#pragma once

#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

#include "common/range.h"
#include "memory/memory_ref.h"

using gear::common::Range;
using gear::memory::UvmMemoryRef;

namespace gear::index {

constexpr size_t ALIGNMENT_BLOCK_SIZE = 128;
constexpr size_t WAIT_LOOP_SLEEP_TIME = 100;
constexpr size_t WAIT_MAX_LOOP = 1000;

inline size_t calc_aligned_block_size(size_t elem_num, size_t elem_size,
                                      size_t alignment) {
  return alignment * ((elem_num * elem_size + alignment - 1) / alignment);
}

inline size_t calc_shard_mem_size(size_t elem_num) {
  return calc_aligned_block_size(elem_num, sizeof(int64_t),
                                 ALIGNMENT_BLOCK_SIZE) +
         calc_aligned_block_size(elem_num, sizeof(float), ALIGNMENT_BLOCK_SIZE);
}

inline size_t calc_state_mem_size(size_t elem_num) {
  return elem_num * (sizeof(int64_t) + sizeof(float));
}

struct StatusTableShard {
  size_t global_capacity;
  size_t local_capacity;
  size_t index_offset;

  UvmMemoryRef mem;

  int64_t *timesteps = nullptr;
  float *weights = nullptr;

  StatusTableShard()
      : global_capacity(0), local_capacity(0), index_offset(0),
        mem(nullptr, 0, 0) {}

  StatusTableShard(size_t global, size_t local, size_t offset,
                   bool shared = false, key_t key = 0, bool create = false)
      : global_capacity(global), local_capacity(local), index_offset(offset),
        mem(nullptr, 0, calc_shard_mem_size(local), shared, key, create) {
    size_t loop_count = 0;
    while (this->mem.raw->addr == nullptr && loop_count < WAIT_MAX_LOOP) {
      this->mem.raw->alloc();
      std::this_thread::sleep_for(
          std::chrono::milliseconds(WAIT_LOOP_SLEEP_TIME));
      loop_count++;
    }
    if (this->mem.raw->addr == nullptr) {
      throw std::runtime_error("Error allocate status table memory!");
    }

    this->timesteps = reinterpret_cast<int64_t *>(this->mem.raw->addr);
    this->weights = reinterpret_cast<float *>(
        reinterpret_cast<char *>(this->mem.raw->addr) +
        calc_aligned_block_size(this->local_capacity, sizeof(int64_t),
                                ALIGNMENT_BLOCK_SIZE));
  }

  StatusTableShard(const StatusTableShard &other)
      : global_capacity(other.global_capacity),
        local_capacity(other.local_capacity), index_offset(other.index_offset),
        mem(other.mem), timesteps(other.timesteps), weights(other.weights) {}

  ~StatusTableShard() {
    this->timesteps = nullptr;
    this->weights = nullptr;
  }
  void join(const StatusTableShard &other) {
    Range op = Range(index_offset, index_offset + local_capacity)
                   .join(Range(other.index_offset,
                               other.index_offset + other.local_capacity));
    if (op.valid()) {
      memcpy(reinterpret_cast<void *>(timesteps + op.lb - index_offset),
             reinterpret_cast<void *>(other.timesteps),
             op.size() * sizeof(int64_t));
      memcpy(reinterpret_cast<void *>(weights + op.lb - index_offset),
             reinterpret_cast<void *>(other.weights),
             op.size() * sizeof(float));
    }
  }

  StatusTableShard merge(const StatusTableShard &other) {
    Range other_range =
        Range(other.index_offset, other.index_offset + other.local_capacity);
    Range this_range = Range(index_offset, index_offset + local_capacity);
    Range op = this_range.merge(other_range);
    // Range joined_range = this_range.join(other_range);

    if (op.valid()) {
      StatusTableShard ret(global_capacity, op.size(), op.lb);
      Range to_copy = op.join(other_range);
      size_t copy_to_low_side = size_t(to_copy.lb == op.lb);
      memcpy(reinterpret_cast<void *>(ret.timesteps + to_copy.lb - op.lb),
             reinterpret_cast<void *>(other.timesteps),
             to_copy.size() * sizeof(int64_t));
      memcpy(reinterpret_cast<void *>(ret.timesteps +
                                      copy_to_low_side * to_copy.size()),
             reinterpret_cast<void *>(
                 timesteps + copy_to_low_side * to_copy.hb - this_range.lb),
             (op.size() - to_copy.size()) * sizeof(int64_t));

      memcpy(reinterpret_cast<void *>(ret.weights + to_copy.lb - op.lb),
             reinterpret_cast<void *>(other.weights),
             to_copy.size() * sizeof(float));
      memcpy(reinterpret_cast<void *>(ret.weights +
                                      copy_to_low_side * to_copy.size()),
             reinterpret_cast<void *>(weights + copy_to_low_side * to_copy.hb -
                                      this_range.lb),
             (op.size() - to_copy.size()) * sizeof(float));
    } else {
      throw std::runtime_error("Two shards cannot merge");
    }
  }

  StatusTableShard sub(size_t offset, size_t length) {
    if (offset + length > local_capacity) {
      throw std::runtime_error("oversubscribe status table shard");
    }
    StatusTableShard ret(*this);
    ret.local_capacity = length;
    ret.index_offset += offset;
    ret.timesteps += offset;
    ret.weights += offset;
    return ret;
  }

  std::pair<StatusTableShard, StatusTableShard> split(size_t split_size) {
    return std::pair<StatusTableShard, StatusTableShard>(
        this->sub(0, split_size),
        this->sub(split_size, local_capacity - split_size));
  }
};

struct IndexsetState {
  size_t global_capacity;
  size_t local_capacity;
  size_t index_offset;

  CpuMemoryRef mem;

  IndexsetState(size_t global_capacity, size_t local_capacity,
                size_t index_offset, const int64_t *timesteps,
                const float *weights)
      : global_capacity(global_capacity), local_capacity(local_capacity),
        index_offset(index_offset),
        mem(nullptr, 0, local_capacity * (sizeof(int64_t) + sizeof(float))) {
    GEAR_DEBUG_PRINT("Begin building IndexsetState...\n");
    memcpy(mem.raw->addr, timesteps, local_capacity * sizeof(int64_t));
    GEAR_DEBUG_PRINT("Building IndexsetState...timesteps copied....\n");
    memcpy(reinterpret_cast<char *>(mem.raw->addr) +
               local_capacity * sizeof(int64_t),
           weights, local_capacity * sizeof(float));
    GEAR_DEBUG_PRINT("Building IndexsetState...weights copied....\n");
  }

  IndexsetState(const IndexsetState &s)
      : global_capacity(s.global_capacity), local_capacity(s.local_capacity),
        index_offset(s.index_offset), mem(s.mem) {}

  int64_t *get_timesteps() {
    return reinterpret_cast<int64_t *>(mem.raw->addr);
  }

  float *get_weights() {
    return reinterpret_cast<float *>(
        reinterpret_cast<char *>(mem.raw->addr) +
        calc_aligned_block_size(this->local_capacity, sizeof(int64_t),
                                ALIGNMENT_BLOCK_SIZE));
  }
};

struct IndexsetDescription {
  size_t global_capacity;
  size_t local_capacity;
  size_t index_offset;
  bool shared;
  key_t key;
};
class Indexset {

public:
  Indexset(size_t global_capacity, size_t local_capacity, size_t index_offset,
           bool shared = false, key_t key = 0, bool create = false)
      : status_table(global_capacity, local_capacity, index_offset, shared, key,
                     create) {}

  Indexset(const IndexsetState &state, bool shared = false, key_t key = 0,
           bool create = false)
      : status_table(StatusTableShard(state.global_capacity,
                                      state.local_capacity, state.index_offset,
                                      shared, key, create)) {
    memcpy(reinterpret_cast<void *>(status_table.timesteps),
           state.mem.raw->addr, sizeof(int64_t) * state.local_capacity);
    memcpy(reinterpret_cast<void *>(status_table.weights),
           reinterpret_cast<char *>(state.mem.raw->addr) +
               sizeof(int64_t) * state.local_capacity,
           sizeof(float) * state.local_capacity);
  }

  int64_t *get_timesteps() { return status_table.timesteps; }

  int64_t get_global_capacity() const { return status_table.global_capacity; }

  int64_t get_local_capacity() const { return status_table.local_capacity; }

  int64_t get_index_offset() const { return status_table.index_offset; }

  bool is_sharing() const { return status_table.mem.raw->sharing_base_mem; }

  key_t get_shm_key() const { return status_table.mem.raw->key; }

  float *get_weights() { return status_table.weights; }

  IndexsetState get_state() const {
    return IndexsetState(status_table.global_capacity,
                         status_table.local_capacity, status_table.index_offset,
                         const_cast<const int64_t *>(status_table.timesteps),
                         const_cast<const float *>(status_table.weights));
  }

  // void join(const Indexset &other) const {}

  // std::shared_ptr<Indexset> merge(const Indexset &other) const {}

  // std::pair<std::shared_ptr<Indexset>, std::shared_ptr<Indexset>>
  // split(size_t split_size) const {}

  bool is_empty() { return this->status_table.mem.raw == nullptr; }

  void set_timestep(size_t index, int64_t ts) {
    this->status_table.timesteps[index] = ts;
  }
  void set_weight(size_t index, float w) {
    this->status_table.weights[index] = w;
  }

private:
  StatusTableShard status_table;
};

} // namespace gear::index