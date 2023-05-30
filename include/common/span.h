#pragma once

#include <torch/extension.h>

#include "common/cmp.h"
#include "common/dtypes.h"
#include "common/tensor.h"

#define MACRO_DEFINE_CONVERT_FUNC(FUNC_NAME, SPAN_NAME, TYPE)                  \
  SPAN_NAME FUNC_NAME() {                                                      \
    return SPAN_NAME(reinterpret_cast<TYPE *>(this->ptr),                      \
                     this->size * sizeof(ElemType) / sizeof(TYPE));            \
  }

#define MACRO_DEFINE_SPAN_CONVERSION()                                         \
  MACRO_DEFINE_CONVERT_FUNC(bool8, BoolSpan, bool);                            \
  MACRO_DEFINE_CONVERT_FUNC(int8, Int8Span, int8_t);                           \
  MACRO_DEFINE_CONVERT_FUNC(uint8, Uint8Span, uint8_t);                        \
  MACRO_DEFINE_CONVERT_FUNC(int16, Int16Span, int16_t);                        \
  MACRO_DEFINE_CONVERT_FUNC(int32, Int32Span, int32_t);                        \
  MACRO_DEFINE_CONVERT_FUNC(int64, Int64Span, int64_t);                        \
  MACRO_DEFINE_CONVERT_FUNC(float32, Float32Span, float);                      \
  MACRO_DEFINE_CONVERT_FUNC(float64, Float64Span, double);

#define MACRO_CAST_TENSOR_DTYPE_CASE(DTYPE, TYPE)                              \
  case DTYPE: {                                                                \
    return Span<TYPE>(reinterpret_cast<TYPE *>(this->ptr),                     \
                      this->size * sizeof(ElemType) / sizeof(TYPE))            \
        .tensor();                                                             \
  }

#define MACRO_STATIC_AS_TENSOR_CASTER(SPAN, TORCH_DTYPE)                       \
  template <> inline torch::Tensor SPAN::static_as_tensor(SPAN s) {            \
    return torch::from_blob(reinterpret_cast<void *>(s.ptr),                   \
                            at::IntArrayRef(std::array<int64_t, 1>{{s.size}}), \
                            at::TensorOptions().dtype(TORCH_DTYPE));           \
  }

namespace gear::common {
template <typename ElemType> struct Span;
using BoolSpan = Span<bool>;
using Int8Span = Span<int8_t>;
using Uint8Span = Span<uint8_t>;
using Int16Span = Span<int16_t>;
using Int32Span = Span<int32_t>;
using Int64Span = Span<int64_t>;
// using Float16Span = Span<at::ScalarType::Half>;
using Float32Span = Span<float>;
using Float64Span = Span<double>;

template <typename ElemType> struct Span {
  ElemType *ptr;
  int64_t size;

  Span<ElemType>(torch::Tensor &t)
      : ptr(t.data_ptr<ElemType>()), size(t.numel()) {}

  Span<ElemType>(void *data, int64_t size)
      : ptr(reinterpret_cast<ElemType *>(data)), size(size) {}

  Span<ElemType>(ElemType *data, int64_t size) : ptr(data), size(size) {}

  ElemType &operator[](size_t idx) { return *(ptr + idx); }

  MACRO_DEFINE_SPAN_CONVERSION();

  static Span<ElemType> from_tensor(torch::Tensor &t) {
    return Span<ElemType>{t.data_ptr<ElemType>(), t.numel()};
  }

  static torch::Tensor static_as_tensor(Span<ElemType>);

  void copy(torch::Tensor t) {
    size_t elem_size = 0;
    void *src = nullptr;
    elem_size = convert_tensor_data_pointer(t, &src);
    memcpy(reinterpret_cast<void *>(this->ptr), src,
           MIN(this->size * sizeof(ElemType), elem_size * t.numel()));
  }

  torch::Tensor tensor() { return Span<ElemType>::static_as_tensor(*this); }

  torch::Tensor cast_tensor(DataType dtype) {
    switch (dtype) {
      MACRO_CAST_TENSOR_DTYPE_CASE(kBool, DType2Type<kBool>::type);
      MACRO_CAST_TENSOR_DTYPE_CASE(kUint8, DType2Type<kUint8>::type);
      MACRO_CAST_TENSOR_DTYPE_CASE(kInt8, DType2Type<kInt8>::type);
      MACRO_CAST_TENSOR_DTYPE_CASE(kInt16, DType2Type<kInt16>::type);
      MACRO_CAST_TENSOR_DTYPE_CASE(kInt32, DType2Type<kInt32>::type);
      MACRO_CAST_TENSOR_DTYPE_CASE(kInt64, DType2Type<kInt64>::type);
      MACRO_CAST_TENSOR_DTYPE_CASE(kFloat32, DType2Type<kFloat32>::type);
      MACRO_CAST_TENSOR_DTYPE_CASE(kFloat64, DType2Type<kFloat64>::type);

    default: {
      throw std::runtime_error("Unsupported type.");
    }
    }
  }
};
MACRO_STATIC_AS_TENSOR_CASTER(BoolSpan, at::ScalarType::Bool);
MACRO_STATIC_AS_TENSOR_CASTER(Int8Span, at::ScalarType::Byte);
MACRO_STATIC_AS_TENSOR_CASTER(Uint8Span, at::ScalarType::Char);
MACRO_STATIC_AS_TENSOR_CASTER(Int16Span, at::ScalarType::Short);
MACRO_STATIC_AS_TENSOR_CASTER(Int32Span, at::ScalarType::Int);
MACRO_STATIC_AS_TENSOR_CASTER(Int64Span, at::ScalarType::Long);
MACRO_STATIC_AS_TENSOR_CASTER(Float32Span, at::ScalarType::Float);
MACRO_STATIC_AS_TENSOR_CASTER(Float64Span, at::ScalarType::Double);

inline void *get_raw_ptr(torch::Tensor &t) { return t.data_ptr(); }
} // namespace gear::common

#undef MACRO_STATIC_AS_TENSOR_CASTER
#undef MACRO_CAST_TENSOR_DTYPE_CASE
#undef MACRO_DEFINE_SPAN_CONVERSION
#undef MACRO_DEFINE_CONVERT_FUNC