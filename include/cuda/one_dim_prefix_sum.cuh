#pragma once

#include <c10/cuda/CUDAStream.h>

template <typename ValueType> __align__(128) struct PartitionDescriptor {
  ValueType inc_prefix;
  ValueType aggregate;
  int status = 0;
  PartitionDescriptor() {}
};
