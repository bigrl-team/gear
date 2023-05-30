#pragma once

#include "memory/memory.h"

namespace gear::memory {
struct CudaMemory : public Memory {
  CudaMemory(size_t size);

  ~CudaMemory();

  ssize_t alloc() override;

  ssize_t free() override;
};

} // namespace gear::memory