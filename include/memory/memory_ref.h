#pragma once
#include <limits>

#include "common/cmp.h"
#include "debug.h"
#include "memory/cuda_uvm.h"
#include "memory/memory.h"
#include "memory/shared_memory.h"

namespace gear::memory {

constexpr size_t MAX_SIZE_T = std::numeric_limits<size_t>::max();
template <typename MemoryBlobType> struct MemoryRef {
  std::shared_ptr<MemoryBlobType> raw = nullptr;
  size_t offset;
  size_t length;

  template <typename... Args>
  MemoryRef<MemoryBlobType>(const void *mem = nullptr, size_t offset = 0,
                            size_t length = MAX_SIZE_T, Args &&...args)
      : raw(length > 0 ? std::make_shared<MemoryBlobType>(
                             length, std::forward<Args>(args)...)
                       : nullptr),
        offset(offset), length(length) {

    GEAR_DEBUG_PRINT("Constructing MemoryRef from const pointer.\n");
    if (raw != nullptr) {
      raw->alloc();
    }
    if (mem != nullptr) {
      memcpy(this->raw->addr, mem, length);
    }
    GEAR_DEBUG_PRINT(
        "MemoryRef constructed complete, memcpy end, %zu bytes copied.\n",
        length);
  }

  MemoryRef<MemoryBlobType>(std::shared_ptr<MemoryBlobType> &mem = nullptr,
                            size_t offset = 0, size_t length = MAX_SIZE_T)
      : raw(mem), offset(offset) {
    if (mem == nullptr) {
      this->length = length;
    } else {
      this->length = MIN(length, mem.size);
    }
  }

  MemoryRef<MemoryBlobType>(const MemoryRef<MemoryBlobType> &other)
      : raw(other.raw), offset(other.offset), length(other.length) {}

  template <typename SecMemoryBlobType>
  MemoryRef<MemoryBlobType>(const MemoryRef<SecMemoryBlobType> &other)
      : raw(std::make_shared<MemoryBlobType>(other.length)),
        offset(other.offset), length(other.length) {
    GEAR_DEBUG_PRINT("Constructing MemoryRef from different memory blob, copy "
                     "memory content.\n")
    memcpy(this->raw->addr, other->raw->addr, length);
  }

  MemoryRef(MemoryRef &&other)
      : raw(std::move(other.raw)), offset(other.offset), length(other.length) {}
};

using CpuMemoryRef = MemoryRef<Memory>;
using SharedMemoryRef = MemoryRef<SharedMemory>;
using UvmMemoryRef = MemoryRef<CudaUVMemory>;

} // namespace gear::memory