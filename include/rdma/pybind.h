#include <pybind11/pybind11.h>

#include "rdma/rdma_client.h"
#include "rdma/rdma_server.h"

namespace gear::rdma {
namespace py = pybind11;

void register_module(py::module &m);

void register_client(py::module &m);

void register_server(py::module &m);
} // namespace gear::rdma