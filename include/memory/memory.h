#pragma once

#include <array>
#include <string>
#include <type_traits>

#include "common/span.h"

namespace gear::memory {
using gear::common::Uint8Span;

template <typename Enumeration>
auto enum_type_to_value(Enumeration const value) ->
    typename std::underlying_type<Enumeration>::type {
  return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

// TODO: GEAR's private memory pool for mem alloc perf.
struct Memory {
  enum class MemoryType {
    kCpu = 0,
    kUvm = 1,
    kShared = 2,
    kCuda = 3,
  };

  void *addr = nullptr;
  size_t size = 0;
  MemoryType mtype;

  Memory(size_t size, MemoryType mtype = MemoryType::kCpu);

  Memory(Memory &&mem);

  Memory(const Memory &mem);

  Memory(const Uint8Span &span);

  ~Memory();

  std::string to_string();

  virtual ssize_t alloc();

  virtual ssize_t free();
};

template <Memory::MemoryType mtype> struct MemType2Name {
  static std::string name();
};

template <> inline std::string MemType2Name<Memory::MemoryType::kCpu>::name() {
  return "cpu";
};
template <> inline std::string MemType2Name<Memory::MemoryType::kUvm>::name() {
  return "uvm";
};
template <>
inline std::string MemType2Name<Memory::MemoryType::kShared>::name() {
  return "shared";
};
template <> inline std::string MemType2Name<Memory::MemoryType::kCuda>::name() {
  return "cuda";
};

inline std::string nameof_mtype(Memory::MemoryType mtype) {
#define MTYPE_CASE(MTYPE)                                                      \
  case MTYPE: {                                                                \
    return MemType2Name<MTYPE>::name();                                        \
  }

  switch (mtype) {
    MTYPE_CASE(Memory::MemoryType::kCpu);
    MTYPE_CASE(Memory::MemoryType::kCuda);
    MTYPE_CASE(Memory::MemoryType::kUvm);
    MTYPE_CASE(Memory::MemoryType::kShared);
  }

#undef MTYPE_CASE
}
} // namespace gear::memory