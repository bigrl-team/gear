#include <cstring>

#include "debug.h"
#include "gear_errors.h"
#include "index/server.h"
#include "memory/shared_memory.h"

namespace gear::index {
SharedMemoryIndexServerState::SharedMemoryIndexServerState(
    size_t capacity, size_t num_clients, size_t index_offset, key_t key,
    const void *status, const void *weights, const void *timesteps,
    const void *records, const IndexBufferState &buffer_state)
    : capacity(capacity), num_clients(num_clients), index_offset(index_offset),
      key(key),
      status(std::move(CpuMemoryRef(const_cast<const void *>(status), 0,
                                    num_clients * sizeof(sim_status_t)))),
      weights(std::move(CpuMemoryRef(const_cast<const void *>(weights), 0,
                                     capacity * sizeof(float)))),
      timesteps(std::move(CpuMemoryRef(const_cast<const void *>(timesteps), 0,
                                       capacity * sizeof(int64_t)))),
      records(std::move(CpuMemoryRef(const_cast<const void *>(records), 0,
                                     num_clients * sizeof(record_t)))),
      buffer_state(buffer_state) {}

SharedMemoryIndexServerState::SharedMemoryIndexServerState(
    SharedMemoryIndexServerState &&state)
    : capacity(state.capacity), num_clients(state.num_clients),
      index_offset(state.index_offset), key(state.key),
      status(std::move(state.status)), weights(std::move(state.weights)),
      timesteps(std::move(state.timesteps)), records(std::move(state.records)),
      buffer_state(std::move(state.buffer_state)) {}

SharedMemoryIndexServerState::SharedMemoryIndexServerState(
    const SharedMemoryIndexServerState &state)
    : capacity(state.capacity), num_clients(state.num_clients),
      index_offset(state.index_offset), key(state.key), status(state.status),
      weights(state.weights), timesteps(state.timesteps),
      records(state.records), buffer_state(state.buffer_state) {}

// SharedMemoryIndexServerState::SharedMemoryIndexServerState(
//     size_t capacity, size_t num_clients, key_t key,
//     gear::memory::Memory &&status, gear::memory::Memory &&weights,
//     gear::memory::Memory &&timesteps, gear::memory::Memory &&records,
//     const IndexBufferState &buffer_state_)
//     : capacity(capacity), num_clients(num_clients), key(key),
//       status(std::move(status)), weights(std::move(weights)),
//       timesteps(std::move(timesteps)), records(std::move(records)),
//       buffer_state(buffer_state_) {}

SharedMemoryIndexServer::SharedMemoryIndexServer(key_t shm_key,
                                                 size_t num_clients,
                                                 size_t index_offset,
                                                 size_t capacity)
    : capacity(capacity), num_clients(num_clients), index_offset(index_offset),
      shm_blk(shm_key, num_clients * sizeof(sim_status_t) /* shm_size */,
              true /* create */),
      weight_mem_blk(capacity * sizeof(float)),
      ts_mem_blk(capacity * sizeof(int64_t)), free_list(capacity) {
  this->records =
      reinterpret_cast<record_t *>(calloc(num_clients, sizeof(record_t)));
  for (size_t i = 0; i < capacity; ++i) {
    this->free_list.push(static_cast<int64_t>(i + index_offset));
  }
}

SharedMemoryIndexServer::~SharedMemoryIndexServer() { free(this->records); }

ssize_t SharedMemoryIndexServer::connect() {
  ssize_t rc = 0;

  if (this->status_array == nullptr && this->shm_blk.size != 0) {
    rc = this->shm_blk.alloc();
    if (rc == -1) {
      GEAR_ERROR("<SharedMemoryIndexServer> shm_blk lazy allocate failed");
      return rc;
    }
    this->status_array = reinterpret_cast<sim_status_t *>(this->shm_blk.addr);
  }

  if (this->timesteps == nullptr && this->ts_mem_blk.size != 0) {
    rc = this->ts_mem_blk.alloc();
    if (rc == -1) {
      GEAR_ERROR("<SharedMemoryIndexServer> ts_mem_blk lazy allocate failed");
      return rc;
    }

    this->timesteps = reinterpret_cast<int64_t *>(this->ts_mem_blk.addr);
    memset(this->ts_mem_blk.addr, 0, this->ts_mem_blk.size);
  }

  if (this->weights == nullptr && this->weight_mem_blk.size != 0) {
    rc = this->weight_mem_blk.alloc();
    if (rc == -1) {
      GEAR_ERROR(
          "<SharedMemoryIndexServer> weight_mem_blk lazy allocate failed");
      return rc;
    }

    this->weights = reinterpret_cast<float *>(this->weight_mem_blk.addr);
    memset(this->weight_mem_blk.addr, 0, this->weight_mem_blk.size);
  }

  ssize_t count = 0;
  for (size_t i = 0; i < this->num_clients; ++i) {
    record_t *rec = records + i;
    if (rec->active) {
      ++count;
    }
  }
  return count;
}

size_t SharedMemoryIndexServer::get_capacity() { return this->capacity; }

void SharedMemoryIndexServer::scan(CachedRequest &cache) {

  bool ret = false;
  size_t i;

  for (i = 0; i < num_clients; ++i) {
    GEAR_DEBUG_PRINT("scanning %ld-th simulator status ....\n", i);

    sim_status_t *status = this->status_array + i;
    if (status->ownership.load(std::memory_order_acquire) != 1) {
      GEAR_DEBUG_PRINT("ownership check failed, skipped\n");
      continue;
    }

    if (!status->terminated) {
      if (status->valid) {
        GEAR_DEBUG_PRINT("inference required for tindex %ld timestep %ld\n",
                         status->tindex, status->timestep);
        cache.iarr.insert(i, status->tindex, status->timestep);
      } else {
        GEAR_ERROR("Invalid status encountered: unterminated ")
      }
    }

    if (status->terminated && status->valid) {
      GEAR_DEBUG_PRINT(
          "trajectory committed and no allocation required ....\n");
      this->free_list.push(status->tindex);
      status->valid = false;
      cache.warr.update(status->tindex, status->timestep, status->weight);
    }

    // weight = 0 indicates unselectable in sampling
    if (status->terminated && status->alloc) {
      GEAR_DEBUG_PRINT("handling allocation requirement ...\n");
      ret = this->free_list.pop(status->tindex);
      if (ret) {
        GEAR_DEBUG_PRINT("allocation done with index %ld ...\n",
                         status->tindex);
        status->alloc = false;
        status->terminated = false;

        status->timestep = 0;
        status->weight = 0;

        status->valid = true;

        cache.warr.update(status->tindex, status->timestep, status->weight);
        GEAR_DEBUG_PRINT("handover ownership to simulator side ...\n");
        status->ownership.store(0, std::memory_order_release);
      }
    } else if (status->terminated && !status->alloc) {
      GEAR_DEBUG_PRINT("handover ownership to simulator side ...\n");
      status->ownership.store(0, std::memory_order_release);
    }
  }

  GEAR_DEBUG_PRINT("ended...\n");
}

void SharedMemoryIndexServer::update(CachedRequest &cache) {
  for (size_t i = 0; i < cache.warr.count; ++i) {
    this->timesteps[i] = cache.warr.timesteps[i];
    this->weights[i] = cache.warr.weights[i];
  }
}

void SharedMemoryIndexServer::callback(CachedRequest &cache) {
  for (size_t i = 0; i < cache.iarr.count; ++i) {
    GEAR_DEBUG_PRINT(
        "Inference callback %zu invoked with %d, traj %ld, timestep %ld ...\n",
        i, cache.iarr.srcs[i], cache.iarr.idxs[i], cache.iarr.tss[i]);
    status_array[cache.iarr.srcs[i]].ownership.store(0,
                                                     std::memory_order_release);
  }
}

float *SharedMemoryIndexServer::get_weights() { return this->weights; }

int64_t *SharedMemoryIndexServer::get_timesteps() { return this->timesteps; }

size_t SharedMemoryIndexServer::get_num_clients() { return this->num_clients; }

SharedMemoryIndexServerState SharedMemoryIndexServer::get_state() const {
  GEAR_DEBUG_PRINT("IndexServer enter get state call");
  return SharedMemoryIndexServerState(
      this->capacity, this->num_clients, this->index_offset, this->shm_blk.key,
      reinterpret_cast<void *>(this->status_array),
      reinterpret_cast<void *>(this->weights),
      reinterpret_cast<void *>(this->timesteps),
      reinterpret_cast<void *>(this->records), (this->free_list).get_state());
}

void SharedMemoryIndexServer::set_state(
    const SharedMemoryIndexServerState &state) {
  GEAR_DEBUG_PRINT("SharedMemoryIndexServer performing pre-set checks...\n");
  GEAR_COND_EXCEPT(this->shm_blk.key == state.key &&
                       this->capacity == state.capacity &&
                       this->num_clients == state.num_clients,
                   std::runtime_error,
                   "Can not rebuild SharedMemoryIndexServer from State, "
                   "key, capacity or num_clients mismatch");
  this->index_offset = state.index_offset;
  GEAR_COND_EXCEPT(
      this->status_array == nullptr, std::runtime_error,
      "Cannot set state for a serving(connected) SharedMemoryIndexServer.");
  GEAR_DEBUG_PRINT("SharedMemoryIndexServer recovering memory allocation...\n");
  int rc = this->connect();
  GEAR_DEBUG_PRINT("SharedMemoryIndexServer recovering memory state...\n");
  GEAR_COND_EXCEPT(rc != -1, std::runtime_error,
                   "SharedMemoryIndexServer mem allocation error");
  GEAR_DEBUG_PRINT("SharedMemoryIndexServer recovering status_array state, "
                   "expected num %zu , elem size %zu, total len %zu...\n",
                   this->num_clients, sizeof(sim_status_t),
                   sizeof(sim_status_t) * this->num_clients);
  memcpy(reinterpret_cast<void *>(this->status_array),
         reinterpret_cast<void *>(state.status.raw->addr),
         sizeof(sim_status_t) * this->num_clients);
  GEAR_DEBUG_PRINT("SharedMemoryIndexServer recovering weights state...\n");
  memcpy(reinterpret_cast<void *>(this->weights),
         reinterpret_cast<void *>(state.weights.raw->addr),
         sizeof(float) * this->capacity);
  GEAR_DEBUG_PRINT("SharedMemoryIndexServer recovering records state...\n");
  memcpy(reinterpret_cast<void *>(this->records),
         reinterpret_cast<void *>(state.records.raw->addr),
         sizeof(record_t) * num_clients);
  GEAR_DEBUG_PRINT("SharedMemoryIndexServer recovering timesteps state...\n");
  memcpy(reinterpret_cast<void *>(this->timesteps),
         reinterpret_cast<void *>(state.timesteps.raw->addr),
         sizeof(int64_t) * capacity);

  GEAR_DEBUG_PRINT("SharedMemoryIndexServer setting buffer state...\n");
  this->free_list.set_state(state.buffer_state);
}
} // namespace gear::index