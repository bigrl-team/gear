#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "common/align.h"

namespace gear::common {

template <typename Enumeration>
auto enum_type_to_value(Enumeration const value) ->
    typename std::underlying_type<Enumeration>::type {
  return static_cast<typename std::underlying_type<Enumeration>::type>(value);
}

enum class DataType {
  kBool,
  kByte,
  kChar,
  kShort,
  kInt,
  kLong,
  kHalf,
  kFloat,
  kDouble
};

// align with torch
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h
constexpr auto kBool = DataType::kBool;
constexpr auto kUint8 = DataType::kByte;
constexpr auto kInt8 = DataType::kChar;
constexpr auto kInt16 = DataType::kShort;
constexpr auto kInt32 = DataType::kInt;
constexpr auto kInt64 = DataType::kLong;
constexpr auto kFloat16 = DataType::kHalf;
constexpr auto kFloat32 = DataType::kFloat;
constexpr auto kFloat64 = DataType::kDouble;

template <DataType dtype> struct DType2Type;
template <> struct DType2Type<kBool> { using type = bool; };
template <> struct DType2Type<kUint8> { using type = uint8_t; };
template <> struct DType2Type<kInt8> { using type = int8_t; };
template <> struct DType2Type<kInt16> { using type = short; };
template <> struct DType2Type<kInt32> { using type = int32_t; };
template <> struct DType2Type<kInt64> { using type = int64_t; };
// template <> struct DType2Type<kFloat16> { using type = half; };
template <> struct DType2Type<kFloat32> { using type = float; };
template <> struct DType2Type<kFloat64> { using type = double; };

template <DataType dtype> using dtype2type_t = typename DType2Type<dtype>::type;

inline size_t sizeof_dtype(DataType dtype) {
#define DTYPE_CASE(DTYPE)                                                      \
  case DTYPE: {                                                                \
    return sizeof(DType2Type<DTYPE>::type);                                    \
  }

  switch (dtype) {
    DTYPE_CASE(kBool);
    DTYPE_CASE(kUint8);
    DTYPE_CASE(kInt8);
    DTYPE_CASE(kInt16);
    DTYPE_CASE(kInt32);
    DTYPE_CASE(kInt64);
    DTYPE_CASE(kFloat32);
    DTYPE_CASE(kFloat64);
  default: {
    throw std::runtime_error("Unsupported type.");
  }
  }

#undef DTYPE_CASE
}
} // namespace gear::common