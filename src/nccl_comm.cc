#include "nccl_comm.h"

namespace gear::comm {
py::bytes create_nccl_id() {
  ncclUniqueId Id;
  ncclGetUniqueId(&Id);
  std::string temp(reinterpret_cast<const char *>(Id.internal),
                   sizeof(Id.internal));
  return py::bytes(temp);
}

NcclCommunicator::NcclCommunicator(int rank, int ws, py::bytes id)
    : rank(rank), size(ws) {
  std::string id_str = id;
  memcpy(nccl_id.internal, id_str.data(), sizeof(nccl_id.internal));
  NCCL_CHECK(ncclCommInitRank(&nccl_comm, ws, nccl_id, rank));
}

NcclCommunicator::NcclCommunicator(const NcclCommunicator &other)
    : rank(other.rank), size(other.size), nccl_comm(other.nccl_comm),
      nccl_id(other.nccl_id) {}

int NcclCommunicator::get_rank() { return rank; }

int NcclCommunicator::get_size() { return size; }

int NcclCommunicator::get_device() {
  int dev;
  ncclCommCuDevice(nccl_comm, &dev);
  return dev;
}

void NcclCommunicator::send(torch::Tensor tensor, int dst) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  ncclDataType_t type;
  void *ptr;
  ptr_type(tensor, &ptr, &type);
  ncclSend(ptr, tensor.numel(), type, dst, nccl_comm, stream);
}

void NcclCommunicator::recv(torch::Tensor tensor, int src) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  ncclDataType_t type;
  void *ptr;
  ptr_type(tensor, &ptr, &type);
  ncclRecv(ptr, tensor.numel(), type, src, nccl_comm, stream);
}

void NcclCommunicator::allreduce(torch::Tensor tensor) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  ncclDataType_t type;
  void *ptr;
  ptr_type(tensor, &ptr, &type);
  ncclAllReduce(ptr, ptr, tensor.numel(), type, ncclSum, nccl_comm, stream);
}
void NcclCommunicator::group_send(int peer, const void *base, Int64Span offsets,
                                  Int64Span lengths, DataType dtype) {

  ncclGroupStart();
  int64_t num_ops = offsets.size;

  ncclDataType_t nccl_dtype = convert_dtype_to_nccl_type(dtype);
  for (int64_t i = 0; i < num_ops; ++i) {
    ncclSend(base + offsets[i], lengths[i], nccl_dtype, peer, this->nccl_comm,
             c10::cuda::getCurrentCUDAStream());
  }
  ncclGroupEnd();
}

void NcclCommunicator::group_recv(int peer, void *base, Int64Span offsets,
                                  Int64Span lengths, DataType dtype) {
  ncclGroupStart();
  int64_t num_ops = offsets.size;

  ncclDataType_t nccl_dtype = convert_dtype_to_nccl_type(dtype);
  for (int64_t i = 0; i < num_ops; ++i) {
    ncclRecv(base + offsets[i], lengths[i], nccl_dtype, peer, this->nccl_comm,
             c10::cuda::getCurrentCUDAStream());
  }
  ncclGroupEnd();
}

void NcclCommunicator::group_exchange(void *send_buffer, Int64Span send_offsets,
                                      Int64Span send_lengths, void *recv_buffer,
                                      Int64Span recv_offsets,
                                      Int64Span recv_lengths) {}

void NcclCommunicator::ptr_type(torch::Tensor tensor, void **ptr,
                                ncclDataType_t *type) {
  if (tensor.options().dtype() == torch::kFloat16) {
    *type = ncclFloat16;
    *ptr = (void *)tensor.data_ptr<at::Half>();
  }
  if (tensor.options().dtype() == torch::kFloat32) {
    *type = ncclFloat32;
    *ptr = (void *)tensor.data_ptr<float>();
  }
  if (tensor.options().dtype() == torch::kInt64) {
    *type = ncclInt64;
    *ptr = (void *)tensor.data_ptr<int64_t>();
  }
}
void register_nccl_comm(py::module &m) {
  m.def("create_nccl_id", &create_nccl_id);
  py::class_<NcclCommunicator>(m, "NcclCommunicator")
      .def(py::init<int, int, py::bytes>())
      .def("rank", &NcclCommunicator::get_rank)
      .def("size", &NcclCommunicator::get_size)
      .def("device", &NcclCommunicator::get_device)
      .def("send", &NcclCommunicator::send)
      .def("recv", &NcclCommunicator::recv)
      .def("allreduce", &NcclCommunicator::allreduce);
}
} // namespace gear::comm
