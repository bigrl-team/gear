#include <cstdlib>
#include <sstream>

#include "memory/memory.h"

namespace gear::memory {

Memory::Memory(size_t size, MemoryType mtype) : size(size), mtype(mtype) {}

Memory::Memory(Memory &&mem)
    : addr(mem.addr), size(mem.size), mtype(mem.mtype) {
  mem.addr = nullptr;
}

Memory::Memory(const Memory &mem)
    : addr(nullptr), size(mem.size), mtype(mem.mtype) {
  this->alloc();
  memcpy(this->addr, mem.addr, this->size);
}

Memory::Memory(const Uint8Span &span)
    : addr(nullptr), size(span.size), mtype(MemoryType::kCpu) {
  this->alloc();
  memcpy(this->addr, reinterpret_cast<void *>(span.ptr), this->size);
}

Memory::~Memory() { this->free(); }

std::string Memory::to_string() {
  std::stringstream ss;
  ss << "<GearMemory, addr: " << this->addr << ", size: " << this->size
     << ", memory type: " << nameof_mtype(this->mtype) << ">";
  return ss.str();
}

ssize_t Memory::alloc() {
  if (this->addr != nullptr) {
    return 0;
  }
  this->addr = calloc(this->size, sizeof(char));
  return this->size;
}

ssize_t Memory::free() {
  if (this->addr == nullptr) {
    return 0;
  }
  ::free(this->addr);
  return this->size;
}
} // namespace gear::memory