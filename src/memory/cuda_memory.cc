#include "config.h"
#include "macros.h"
#include "memory/cuda_memory.h"

namespace gear::memory {
CudaMemory::CudaMemory(size_t size)
    : Memory(((size + GPU_PAGE_SIZE - 1) / GPU_PAGE_SIZE) *
                 GPU_PAGE_SIZE, // aligned with GPU page size
             Memory::MemoryType::kCuda) {}

CudaMemory::~CudaMemory() { this->free(); }

ssize_t CudaMemory::alloc() {
  if (this->addr != nullptr) {
    return 0;
  }
  CUDA_CHECK(cudaMalloc(&this->addr, this->size));
  return this->size;
}

ssize_t CudaMemory::free() {
  if (this->addr == nullptr) {
    return 0;
  }
  CUDA_CHECK(cudaFree(this->addr));
  this->addr = nullptr;
  return this->size;
}

} // namespace gear::memory
