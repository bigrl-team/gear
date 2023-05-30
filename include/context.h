#pragma once
#include <cuda.h>
#include <map>
#include <pybind11/pybind11.h>
#include <sys/shm.h>

namespace py = pybind11;

struct RegisteredSharedMemoryBlock {
  key_t key;
  int shmid;
  void *addr;
  int64_t size;
  bool create;
  bool active;
  RegisteredSharedMemoryBlock(key_t key, int shmid, void *addr, int64_t size,
                              bool create);
};

enum MemoryStatus { kInvalid = 0, kActive = 1, kExposed = 2, kFreed = 3 };

class Context {
public:
  Context();

  ~Context();

  void free();

  uintptr_t host_malloc(int64_t size);

  void host_free(uintptr_t ptr);

  uintptr_t host_register_shm(key_t key, int64_t size, bool create);

  void host_free_shm(key_t key);

  uintptr_t device_malloc(int64_t size);

  void device_free(uintptr_t ptr);

  // debug use functions
  static int get_current_device();

private:
  std::map<uintptr_t, bool> allocated_memory_ptrs;
  std::map<key_t, RegisteredSharedMemoryBlock> registered_shm_ptrs;
  std::map<uintptr_t, bool> allocated_device_ptr;
};

void register_context(py::module &m);