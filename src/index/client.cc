#include <pybind11/pybind11.h>
#include <thread>

#include "debug.h"
#include "gear_errors.h"
#include "index/client.h"

namespace gear::index {
SharedMemoryIndexClient::SharedMemoryIndexClient(key_t shm_key,
                                                 size_t client_rank,
                                                 size_t num_clients)
    : client_rank(client_rank), num_clients(num_clients),
      shm_blk(shm_key, num_clients * sizeof(sim_status_t) /* shm_size */,
              false /* create */) {}

ssize_t SharedMemoryIndexClient::connect() {
  ssize_t rc = this->shm_blk.alloc();
  if (rc == -1) {
    return rc;
  }
  this->status =
      reinterpret_cast<sim_status_t *>(this->shm_blk.addr) + this->client_rank;
  return rc;
}

void SharedMemoryIndexClient::release() {
  this->status->ownership.store(1, std::memory_order_release);
}

void SharedMemoryIndexClient::acquire() {
  int rc;
  while (rc = PyErr_CheckSignals(), rc == 0) {
    if (this->status->ownership.load(std::memory_order_acquire) != 0) {
      // status ownership belongs to the server-side, waiting ....
      std::this_thread::yield();
    } else if (!this->status->valid) {
      // apply for allocation ...
      this->status->terminated = true;
      this->status->alloc = true;
      this->status->ownership.store(1, std::memory_order_release);
    } else {
      this->on_the_fly = true;
      this->tindex = this->status->tindex;
      this->timestep = 0;
      break;
    }
  }

  GEAR_COND_EXCEPT(rc == 0, pybind11::error_already_set, );
}

void SharedMemoryIndexClient::wait() {
  int rc;
  while (rc = PyErr_CheckSignals(), rc == 0) {
    if (this->status->ownership.load(std::memory_order_acquire) != 0) {
      // status ownership belongs to the server-side, waiting ....
      std::this_thread::yield();
    } else {
      break;
    }
  }

  GEAR_COND_EXCEPT(rc == 0, pybind11::error_already_set, );
}

int64_t SharedMemoryIndexClient::get_index() {
  return this->on_the_fly ? this->tindex : 0;
}

std::shared_ptr<RawSimulStatus> SharedMemoryIndexClient::get_status_unsafe() {
  return std::make_shared<RawSimulStatus>(RawSimulStatus{
      this->status->ownership.load(std::memory_order_relaxed),
      this->status->terminated, this->status->alloc, this->status->valid,
      this->status->tindex, this->status->timestep, this->status->weight});
}

int64_t SharedMemoryIndexClient::writeback(bool terminated, bool alloc_new) {
  if (!this->on_the_fly) {
    GEAR_DEBUG_PRINT("writeback called on a inactive client, skipped ....\n")
    return 0;
  }

  this->status->terminated = terminated;
  this->status->timestep = this->timestep;
  this->status->alloc = alloc_new;

  this->on_the_fly = !terminated;

  this->release();
  return this->timestep;
}

int64_t SharedMemoryIndexClient::get_timestep() {
  return this->on_the_fly ? this->timestep : 0;
}

int64_t SharedMemoryIndexClient::step_inc() {
  GEAR_DEBUG_PRINT("calling step, status %d, with timestep %ld ...\n",
                   this->on_the_fly, this->timestep);
  this->timestep += 1;
  return this->on_the_fly ? this->timestep - 1 : 0;
}

size_t SharedMemoryIndexClient::get_num_clients() const { return this->num_clients; }

} // namespace gear::index