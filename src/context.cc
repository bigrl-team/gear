
#include "config.h"
#include "context.h"
#include "debug.h"
#include "macros.h"
#include <stdio.h>

RegisteredSharedMemoryBlock::RegisteredSharedMemoryBlock(key_t key, int shmid,
                                                         void *addr,
                                                         int64_t size,
                                                         bool create) {
  this->key = key;
  this->shmid = shmid;
  this->addr = addr;
  this->size = size;
  this->create = create;
  this->active = true;
}

Context::Context() {}

Context::~Context() { this->free(); }

void Context::free() {
  for (auto p = allocated_memory_ptrs.begin(); p != allocated_memory_ptrs.end();
       ++p) {
    if (p->second) {
      this->host_free(p->first);
    }
  }

  for (auto p = registered_shm_ptrs.begin(); p != registered_shm_ptrs.end();
       ++p) {
    if (p->second.active) {
      this->host_free_shm(p->first);
    }
  }

  for (auto p = allocated_device_ptr.begin(); p != allocated_device_ptr.end();
       ++p) {
    if (p->second) {
      this->device_free(p->first);
    }
  }
}

uintptr_t Context::host_malloc(int64_t size) {
  void *ptr;
  CUDA_CHECK(
      cudaHostAlloc(&ptr, size, cudaHostAllocMapped | cudaHostAllocPortable));
  uintptr_t ptr_t = reinterpret_cast<uintptr_t>(ptr);
  this->allocated_memory_ptrs.emplace(ptr_t, true);
  return ptr_t;
}

uintptr_t Context::host_register_shm(key_t key, int64_t size, bool create) {
  int shmid = 0;
  if (create) {
    shmid = shmget(key, size, IPC_CREAT | 0777);
  } else {
    shmid = shmget(key, size, 0666);
  }
  GEAR_ASSERT(shmid != -1, "<Context> Error in shmget.")
  void *shm_addr = shmat(shmid, 0, 0);
  GEAR_ASSERT(shm_addr != nullptr, "<Context> Error in shmat.")

#ifdef GEAR_VERBOSE_DEBUG_ON
  int device;
  cudaGetDevice(&device);
  fprintf(stdout,
          "<Context>: Register shared-memory <shmid: %d, addr: %p, size: %ld, "
          "current cuda device: %d>.\n",
          shmid, shm_addr, size, device);
#endif

  CUDA_CHECK(
      cudaHostRegister(shm_addr, (size_t)size,
                       cudaHostRegisterMapped | cudaHostRegisterPortable));
  this->registered_shm_ptrs.emplace(
      key, RegisteredSharedMemoryBlock(key, shmid, shm_addr, size, create));

  return reinterpret_cast<uintptr_t>(shm_addr);
}

void Context::host_free_shm(key_t key) {
  auto itr = this->registered_shm_ptrs.find(key);
  GEAR_ASSERT(itr != this->registered_shm_ptrs.end(),
              "shm key have not been registered");

  CUDA_CHECK(cudaHostUnregister(itr->second.addr))

  GEAR_ASSERT(shmdt(itr->second.addr) == 0, "shmdt failed");
  if (itr->second.create) {
    GEAR_ASSERT(shmctl(itr->second.shmid, IPC_RMID, NULL) == 0,
                "shmctl failed");
  }
  itr->second.active = false;
}

void Context::host_free(uintptr_t ptr) {
  auto itr = this->allocated_memory_ptrs.find(ptr);
  GEAR_ASSERT(itr != this->allocated_memory_ptrs.end(),
              "Memory pointer have not been registered");
  CUDA_CHECK(cudaFreeHost(reinterpret_cast<void *>(ptr)));
  itr->second = false;
  // printf("memory %u freed", ptr);
}

uintptr_t Context::device_malloc(int64_t size) {
  void *ptr;
  // malloc align with gpu page size
  size = ((size + GPU_PAGE_SIZE - 1) / GPU_PAGE_SIZE) * GPU_PAGE_SIZE;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  this->allocated_device_ptr.emplace(addr, true);
  return addr;
}

void Context::device_free(uintptr_t ptr) {
  auto itr = allocated_device_ptr.find(ptr);
  GEAR_ASSERT(itr != this->allocated_device_ptr.end(), "Invalid device pointer")

  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(ptr)));
  itr->second = false;
}

int Context::get_current_device() {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

void register_context(py::module &m) {
  py::class_<Context, std::shared_ptr<Context>>(m, "Context")
      .def(py::init<>())
      .def("host_malloc", &Context::host_malloc)
      .def("host_free", &Context::host_free)
      .def("host_register_shm", &Context::host_register_shm)
      .def("host_free_shm", &Context::host_free_shm)
      .def("device_malloc", &Context::device_malloc)
      .def("device_free", &Context::device_free)
      .def("free", &Context::host_free)
      .def("get_current_device", &Context::get_current_device);
}