#pragma once

#include <torch/extension.h>

#include "gear_errors.h"

ssize_t inline convert_tensor_data_pointer(torch::Tensor t, void **ptr) {
  if (t.options().dtype() == torch::kBool) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<bool>());
    return sizeof(uint8_t);
  }
  if (t.options().dtype() == torch::kU8) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<uint8_t>());
    return sizeof(uint8_t);
  } else if (t.options().dtype() == torch::kInt8) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<int8_t>());
    return sizeof(int8_t);
  } else if (t.options().dtype() == torch::kFloat16) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<at::Half>());
    return 2 * sizeof(uint8_t);
  } else if (t.options().dtype() == torch::kFloat32) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<float>());
    return sizeof(float);
  } else if (t.options().dtype() == torch::kFloat64) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<double>());
    return sizeof(double);
  } else if (t.options().dtype() == torch::kInt32) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<int32_t>());
    return sizeof(int32_t);
  } else if (t.options().dtype() == torch::kInt64) {
    *ptr = reinterpret_cast<void *>(t.data_ptr<int64_t>());
    return sizeof(int64_t);
  }
}

template <typename DType>
torch::Tensor inline convert_data_pointer_as_tensor(DType *ptr, size_t size);

template <>
torch::Tensor inline convert_data_pointer_as_tensor<int32_t>(int32_t *ptr,
                                                             size_t size) {
  GEAR_COND_EXCEPT(
      ptr != nullptr /*  expected behavior, otherwise cause exception */,
      std::runtime_error, "Cannot convert nullptr to tensor view");
  return torch::from_blob(
      reinterpret_cast<void *>(ptr),
      at::IntArrayRef(std::array<int64_t, 1>{{static_cast<int64_t>(size)}}),
      at::TensorOptions().dtype(at::ScalarType::Int));
}

template <>
torch::Tensor inline convert_data_pointer_as_tensor<int64_t>(int64_t *ptr,
                                                             size_t size) {
  GEAR_COND_EXCEPT(
      ptr != nullptr /*  expected behavior, otherwise cause exception */,
      std::runtime_error, "Cannot convert nullptr to tensor view");
  return torch::from_blob(
      reinterpret_cast<void *>(ptr),
      at::IntArrayRef(std::array<int64_t, 1>{{static_cast<int64_t>(size)}}),
      at::TensorOptions().dtype(at::ScalarType::Long));
}

template <>
torch::Tensor inline convert_data_pointer_as_tensor<float>(float *ptr,
                                                           size_t size) {
  GEAR_COND_EXCEPT(
      ptr != nullptr /*  expected behavior, otherwise cause exception */,
      std::runtime_error, "Cannot convert nullptr to tensor view");
  return torch::from_blob(
      reinterpret_cast<void *>(ptr),
      at::IntArrayRef(std::array<int64_t, 1>{{static_cast<int64_t>(size)}}),
      at::TensorOptions().dtype(at::ScalarType::Float));
}
