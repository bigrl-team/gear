#pragma once

#include <sys/types.h>

#include "circular_buffer.h"
#include "index/request.h"
#include "index/sim_status.h"
#include "memory/memory.h"
#include "memory/shared_memory.h"

using gear::memory::CpuMemoryRef;
using gear::memory::SharedMemoryRef;
using gear::memory::UvmMemoryRef;
namespace gear::index {

using IndexBuffer = CircularBuffer<int64_t>;
using IndexBufferState = CircularBufferState<int64_t>;

struct SharedMemoryIndexServerState {
  size_t capacity;
  size_t num_clients;
  size_t index_offset;
  key_t key;

  CpuMemoryRef status;
  CpuMemoryRef weights;
  CpuMemoryRef timesteps;
  CpuMemoryRef records;

  IndexBufferState buffer_state;

  SharedMemoryIndexServerState(size_t capacity, size_t num_clients,
                               size_t index_offset, key_t key,
                               const void *status, const void *weights,
                               const void *timesteps, const void *records,
                               const IndexBufferState &buffer_state);

  SharedMemoryIndexServerState(SharedMemoryIndexServerState &&s);

  SharedMemoryIndexServerState(const SharedMemoryIndexServerState &state);

  // SharedMemoryIndexServerState(size_t capacity, size_t num_clients, key_t
  // key,
  //                              gear::memory::Memory &&status,
  //                              gear::memory::Memory &&weights,
  //                              gear::memory::Memory &&timesteps,
  //                              gear::memory::Memory &&records,
  //                              const IndexBufferState &buffer_state);
};

class SharedMemoryIndexServer {
public:
  SharedMemoryIndexServer(key_t shm_key, size_t num_clients,
                          size_t index_offset, size_t capacity);

  ~SharedMemoryIndexServer();

  size_t get_capacity();

  ssize_t connect();

  void scan(CachedRequest &cache);

  void update(CachedRequest &cache);

  void callback(CachedRequest &cache);

  // apis reserved for data structure exposure
  float *get_weights();

  int64_t *get_timesteps();

  size_t get_num_clients();

  SharedMemoryIndexServerState get_state() const;

  void set_state(const SharedMemoryIndexServerState &state);

private:
  size_t capacity;
  size_t num_clients;
  size_t index_offset;

  gear::memory::SharedMemory shm_blk;        // status
  gear::memory::CudaUVMemory weight_mem_blk; // weights
  gear::memory::CudaUVMemory ts_mem_blk;     // weights
  sim_status_t *status_array = nullptr;
  float *weights = nullptr;
  int64_t *timesteps = nullptr;

  record_t *records = nullptr;

  IndexBuffer free_list;
};

} // namespace gear::index