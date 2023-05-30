#pragma once

#include "memory/shared_memory.h"

namespace gear::memory {
struct CudaUVMemory : public SharedMemory {
  bool sharing_base_mem = false;

  CudaUVMemory(size_t size, bool shared = false, key_t shm_key = 0,
               bool create = false);

  ~CudaUVMemory();

  ssize_t alloc() override;
  ssize_t free() override;
};

} // namespace gear::memory