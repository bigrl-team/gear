#include <errno.h>
#include <sys/shm.h>

#include "debug.h"
#include "macros.h"
#include "memory/shared_memory.h"

namespace gear::memory {
SharedMemory::SharedMemory() : Memory(0, Memory::MemoryType::kShared) {
  this->key = 0;
  this->create = false;
}

SharedMemory::SharedMemory(key_t key, size_t size, bool create)
    : Memory(size, Memory::MemoryType::kShared) {
  this->key = key;
  this->create = create;
}

SharedMemory &SharedMemory::operator=(SharedMemory &&shm) {
  this->free();

  this->addr = shm.addr;
  this->size = shm.size;

  this->key = shm.key;
  this->create = shm.create;
  this->shmid = shm.shmid;

  return *this;
}

SharedMemory::~SharedMemory() { this->free(); }

ssize_t SharedMemory::alloc() {
  if (this->addr != nullptr) {
    return this->size;
  }

  if (create) {
    this->shmid = shmget(key, size, IPC_CREAT | 0777);
  } else {
    this->shmid = shmget(key, size, 0666);
  }
  if (this->shmid == -1) {
    GEAR_ERROR("SharedMemory shmget failed with errno:%d, create: %d, key:%d, "
               "size: %zu",
               errno, create, key, size);
    return -1;
  }
  this->addr = shmat(shmid, 0, 0);
  if (this->addr == nullptr) {
    GEAR_ERROR("SharedMemory shmat failed with errno:%d, create: %d, key:%d, "
               "size: %zu",
               errno, create, key, size);
    return -1;
  }
  GEAR_DEBUG_PRINT("register SharedMemory %p.\n", this->addr);
  CUDA_CHECK(
      cudaHostRegister(this->addr, (size_t)size,
                       cudaHostRegisterMapped | cudaHostRegisterPortable));
  return this->size;
}

ssize_t SharedMemory::free() {
  if (this->addr == nullptr) {
    return -1;
  }
  GEAR_DEBUG_PRINT("free SharedMemory %p.\n", this->addr);
  CUDA_CHECK(cudaHostUnregister(this->addr));
  if (shmdt(this->addr) == -1) {
    return -1;
  }
  this->addr = nullptr;

  if (this->create) {
    if (shmctl(this->shmid, IPC_RMID, NULL) == -1) {
      return -1;
    }
  }
  return this->size;
}

} // namespace gear::memory