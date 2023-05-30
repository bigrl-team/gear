#pragma once
#include <sstream>
#include <string>

namespace gear::memory {
class MemoryPtr {
public:
  MemoryPtr() : ptr(nullptr) {}

  MemoryPtr(void *ptr) : ptr(ptr) {}

  MemoryPtr(int64_t ptr) : ptr(reinterpret_cast<void *>(ptr)) {}

  MemoryPtr(const MemoryPtr &mptr) : ptr(mptr.ptr) {}

  operator void *() const { return ptr; }

  void *get() const { return ptr; }

  inline std::string to_string() {
    std::stringstream ss;
    ss << reinterpret_cast<void *>(ptr);
    return ss.str();
  }

private:
  void *ptr;
};

} // namespace gear::memory