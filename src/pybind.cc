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

// binding tools
template <typename Type>
std::vector<Type> PylistToCppVecConverter(const py::list &l) {
  std::vector<Type> vec;
  for (auto v : l) {
    vec.push_back(v.cast<Type>());
  }
  for (auto v : vec) {
    std::cout << v.to_string() << "\n";
  }
  return vec;
};

void register_pylist_converter(py::module &m) {
  m.def("node_list_convert", &PylistToCppVecConverter<gear::rdma::NodeBrief>);
}

struct NArray {
  int values[100];

  NArray() {
    for (size_t i = 0; i < 100; ++i) {
      this->values[i] = (int)i;
    }
  }

  NArray(bool init) {
    if (init) {
      for (size_t i = 0; i < 100; ++i) {
        this->values[i] = (int)i;
      }
    }
  }
};

void register_NArray(py::module &m) {
  py::class_<NArray, std::shared_ptr<NArray>>(m, "NArray")
      .def(py::init<>())
      .def("ivalue", [](NArray &a, size_t idx) { return a.values[idx]; })
      .def(py::pickle(
          [](const NArray &a) {
            return py::make_tuple(py::array_t<uint8_t>({sizeof(int) * 100}, {
              1
            }, reinterpret_cast<const uint8_t *>(a.values)));
          },
          [](py::tuple t) {
            if (t.size() != 1) {
              throw std::runtime_error("tuple size mismatch");
            }
            py::array_t<uint8_t> base = t[0].cast<py::array_t<uint8_t>>();
            auto a = std::make_shared<NArray>(false);
            memcpy(a->values,
                   reinterpret_cast<const void *>(base.request().ptr),
                   sizeof(int) * 100);
            return a;
          }));
}

PYBIND11_MODULE(libgear, libgear) {

  libgear.doc() = "GEAR's internal C++ library";

  auto libgear_core = libgear.def_submodule(
      "core", "Core functional components provided here.");

  // kernel_launchers.h
  register_cuda_init(libgear_core);
  register_sum_kernel_launcher(libgear_core);
  register_prefix_sum_kernel_launcher(libgear_core);
  register_copy_kernerl_launcher(libgear_core);

  register_context(libgear_core); // context.h

  gear::common::register_common_module(libgear_core);

  gear::rdma::register_module(libgear_core);

  gear::index::register_module(libgear_core);

  // register_pylist_converter(m);

  gear::env::register_env_detect_functions(libgear_core);

  gear::storage::register_module(libgear_core);

  gear::memory::register_module(libgear_core);

  auto libgear_comm = libgear.def_submodule("comm", "Communicator submodule");

  gear::comm::register_nccl_comm(libgear_comm);

  register_NArray(libgear_core);
}
