#pragma once

#include "index/sim_status.h"
#include "memory/shared_memory.h"

namespace gear::index {

struct RawSimulStatus {
  bool ownership;
  bool terminated;
  bool alloc;
  bool valid;
  int64_t tindex;
  int64_t timestep;
  double weight;
};

class SharedMemoryIndexClient {
public:
  SharedMemoryIndexClient(key_t shm_key, size_t client_rank,
                          size_t num_clients);

  ssize_t connect();

  void release();

  void acquire();

  void wait();

  int64_t get_index();

  int64_t writeback(bool terminated, bool alloc_new);

  int64_t get_timestep();

  std::shared_ptr<RawSimulStatus> get_status_unsafe();

  int64_t step_inc();

  size_t get_num_clients() const;

private:
  size_t client_rank;
  size_t num_clients;

  gear::memory::SharedMemory shm_blk;

  size_t timestep;
  bool on_the_fly = false;
  int64_t tindex;
  sim_status_t *status;

  SharedMemoryIndexClient() = delete;

  SharedMemoryIndexClient(const SharedMemoryIndexClient &other) = delete;

  SharedMemoryIndexClient &operator=(const SharedMemoryIndexClient &) = delete;
};

} // namespace gear::index