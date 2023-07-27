#include <iostream>
#include <pybind11/pybind11.h>
#include <vector>

#include "common/pybind.h"
#include "context.h"
#include "env.h"
#include "index/pybind.h"
#include "kernel_launchers.h"
#include "memory/pybind.h"
#include "nccl_comm.h"
#include "rdma/pybind.h"
#include "storage/pybind.h"

namespace py = pybind11;

PYBIND11_MODULE(libgear, libgear) {

  libgear.doc() = "GEAR's internal C++ library";

  auto libgear_cuda = libgear.def_submodule(
      "cuda", "Cuda kernel launchers");

  // kernel_launchers.h
  register_cuda_init(libgear_cuda);
  register_sum_kernel_launcher(libgear_cuda);
  register_prefix_sum_kernel_launcher(libgear_cuda);
  register_copy_kernerl_launcher(libgear_cuda);

  // register_context(libgear_core); // context.h

  gear::common::register_common_module(libgear);

  auto libgear_rdma = libgear.def_submodule(
      "rdma", "RDMA related classes & methods implementations");
  gear::rdma::register_module(libgear_rdma);

  auto libgear_index = libgear.def_submodule(
      "index", "Data structures and methods for trajectory index management");
  gear::index::register_module(libgear_index);

  gear::env::register_env_detect_functions(libgear);

  auto libgear_storage = libgear.def_submodule(
      "storage", "Storage organizing and methods operated on memory with storage sematics");
  gear::storage::register_module(libgear_storage);

  auto libgear_memory = libgear.def_submodule(
      "memory", "Low level memory abstractions used in gear project");
  gear::memory::register_module(libgear_memory);

  auto libgear_comm = libgear.def_submodule("comm", "Communicator submodule");

  gear::comm::register_nccl_comm(libgear_comm);

}
