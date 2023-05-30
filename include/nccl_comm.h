#pragma once

#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <pybind11/numpy.h>
#include <string>
#include <torch/extension.h>

#include "common/dtypes.h"
#include "common/span.h"

namespace py = pybind11;

using gear::common::DataType;
using gear::common::kBool;
using gear::common::kFloat16;
using gear::common::kFloat32;
using gear::common::kFloat64;
using gear::common::kInt16;
using gear::common::kInt32;
using gear::common::kInt64;
using gear::common::kInt8;
using gear::common::kUint8;

using gear::common::Int64Span;

namespace gear::comm {
inline ncclDataType_t convert_dtype_to_nccl_type(DataType dtype) {
#define MACRO_DTYPE_CASE(DTYPE, NCCL_TYPE)                                     \
  case DTYPE: {                                                                \
    return NCCL_TYPE;                                                          \
  }

  switch (dtype) {
    MACRO_DTYPE_CASE(kBool, ncclUint8);
    MACRO_DTYPE_CASE(kUint8, ncclUint8);
    MACRO_DTYPE_CASE(kInt8, ncclInt8);
    MACRO_DTYPE_CASE(kInt32, ncclInt32);
    MACRO_DTYPE_CASE(kInt64, ncclInt64);
    MACRO_DTYPE_CASE(kFloat16, ncclFloat16);
    MACRO_DTYPE_CASE(kFloat32, ncclFloat32);
    MACRO_DTYPE_CASE(kFloat64, ncclFloat64);
  default: {
    throw std::runtime_error("Unsupported data type");
  }
  }

#undef MACRO_DTYPE_CASE
}

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(res));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

py::bytes create_nccl_id();

struct NcclCommunicator {
  NcclCommunicator(int rank, int ws, py::bytes id);

  NcclCommunicator(const NcclCommunicator &comm);

  int get_rank();

  int get_size();

  int get_device();

  void send(torch::Tensor tensor, int dst);

  void recv(torch::Tensor tensor, int src);

  void allreduce(torch::Tensor tensor);

  // group calls

  void group_send(int peer, const void *base, Int64Span offsets,
                  Int64Span lengths, DataType dtype);

  void group_recv(int peer, void *base, Int64Span offsets, Int64Span lengths,
                  DataType dtype);

  void group_exchange(void *send_buffer, Int64Span send_offsets,
                      Int64Span send_lengths, void *recv_buffer,
                      Int64Span recv_offsets, Int64Span recv_lengths);

  void ptr_type(torch::Tensor tensor, void **ptr, ncclDataType_t *type);
  int rank;
  int size;
  ncclComm_t nccl_comm;
  ncclUniqueId nccl_id;
};

void register_nccl_comm(py::module &m);
} // namespace gear::comm
