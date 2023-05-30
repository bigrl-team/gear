
#include "macros.h"
#include "memory/cuda_uvm.h"

namespace gear::memory {
CudaUVMemory::CudaUVMemory(size_t size, bool shared, key_t shm_key, bool create)
    : SharedMemory(shm_key, size, create), sharing_base_mem(shared) {}

CudaUVMemory::~CudaUVMemory() { this->free(); }
ssize_t CudaUVMemory::alloc() {
  if (this->addr != nullptr) {
    // throw std::runtime_error("Duplicate call of alloc on a memory
    // instance.");
    return 0;
  }

  void *ptr = nullptr;
  ssize_t rc = 0;
  if (this->sharing_base_mem) {
    rc = SharedMemory::alloc();
#ifdef GEAR_VERBOSE_DEBUG_ON
    int device;
    cudaGetDevice(&device);
    fprintf(
        stdout,
        "<Context>: Register shared-memory <shmid: %d, addr: %p, size: %ld, "
        "current cuda device: %d>.\n",
        this->shmid, this->addr, this->size, device);
#endif
  } else {
    CUDA_CHECK(
        cudaHostAlloc(&ptr, size, cudaHostAllocMapped | cudaHostAllocPortable));
    this->addr = ptr;
    rc = size;
  }

  return rc;
}

ssize_t CudaUVMemory::free() {
  if (this->addr == nullptr) {
    // throw std::runtime_error("Duplicate call of alloc on a memory
    // instance.");
    return 0;
  }
  if (this->sharing_base_mem) {
    // CUDA_CHECK(cudaHostUnregister(this->addr));
    SharedMemory::free();
  } else {
    CUDA_CHECK(cudaFreeHost(this->addr));
  }
  this->addr = nullptr;
  return this->size;
}

} // namespace gear::memory