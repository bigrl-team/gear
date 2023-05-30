#pragma once

#include "memory/memory.h"

namespace gear::memory {
struct SharedMemory : public Memory {
  key_t key;
  bool create; // create shm block if true, else attach
  int shmid = 0;

  SharedMemory &operator=(const SharedMemory &shm) = delete;

  SharedMemory();

  SharedMemory(key_t key, size_t size, bool create);

  SharedMemory &operator=(SharedMemory &&shm);

  ~SharedMemory();

  ssize_t alloc() override;

  ssize_t free() override;
};

} // namespace gear::memory