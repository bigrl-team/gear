#include "rdma/pybind.h"

namespace gear::rdma {
void register_module(py::module &m) {
  register_client(m);
  register_server(m);
}

// ======== submodule-wise pybind11 registration callback =========
void register_client(py::module &m) {
  py::class_<NodeBrief, std::shared_ptr<NodeBrief>>(m, "NodeBrief")
      .def(py::init<>())
      .def(py::init<int, std::string, int>())
      // members
      .def_readwrite("node_rank", &NodeBrief::node_rank)
      .def_readwrite("addr", &NodeBrief::addr)
      .def_readwrite("port", &NodeBrief::port);

  py::class_<Pipe, std::shared_ptr<Pipe>>(m, "Pipe")
      .def(py::init<std::string, int, int, int>())
      .def(py::init<const NodeBrief &, int, int>())
      .def(py::init<const std::shared_ptr<NodeBrief>, int, int>())
      .def("read",
           py::overload_cast<infinity::memory::Buffer *, std::vector<int64_t>,
                             std::vector<int64_t>, int64_t>(&Pipe::read),
           "Vector interface")
      .def("read",
           py::overload_cast<infinity::memory::Buffer *, int64_t, int64_t *,
                             int64_t *, int64_t>(&Pipe::read),
           "Raw pointer interface")
      .def("connect", &Pipe::connect);

  py::class_<RDMAClient, std::shared_ptr<RDMAClient>>(m, "RDMAClient")
      .def(py::init<int, std::vector<NodeBrief>, int>())
      .def("connect", &RDMAClient::connect)
      .def("register_buffer",
           py::overload_cast<void *, size_t>(&RDMAClient::register_buffer),
           "Raw pointer interface")
      .def("register_buffer",
           py::overload_cast<uintptr_t, size_t>(&RDMAClient::register_buffer),
           "Python uinterptr interface")
      .def("register_buffer",
           py::overload_cast<torch::Tensor>(&RDMAClient::register_buffer),
           "Torch tensor interface")

      .def("register_buffer",
           py::overload_cast<std::vector<void *>, std::vector<size_t>>(
               &RDMAClient::register_buffer),
           "Vectorized raw pointer interface")
      .def("register_buffer",
           py::overload_cast<std::vector<uintptr_t>, std::vector<size_t>>(
               &RDMAClient::register_buffer),
           "Vectorized python uinterptrs interface")
      .def("register_buffer",
           py::overload_cast<std::vector<torch::Tensor>>(
               &RDMAClient::register_buffer),
           "Vectorized torch tensors interface")
      .def("deregister_buffer", &RDMAClient::deregister_buffer)
      .def("read",
           py::overload_cast<int, int, std::vector<int64_t>,
                             std::vector<int64_t>, int64_t>(&RDMAClient::read),
           "vector read interface")
      .def("read",
           py::overload_cast<int, int, torch::Tensor, torch::Tensor, int64_t>(
               &RDMAClient::read),
           "tensor read interface")
      .def("get_pipes", &RDMAClient::get_pipes);
}

void register_server(py::module &m) {
  py::class_<RDMAServerConfig, std::shared_ptr<RDMAServerConfig>>(
      m, "RDMAServerConfig")
      .def(py::init<>())
      .def_readwrite("port", &RDMAServerConfig::port)
      .def_readwrite("num_clients", &RDMAServerConfig::num_clients)
      .def_readwrite("qp_per_client", &RDMAServerConfig::qp_per_client);

  py::class_<RDMAServer, std::shared_ptr<RDMAServer>>(m, "RDMAServer")
      .def(py::init<RDMAServerConfig>())
      .def(py::init<void *, size_t, RDMAServerConfig>())
      .def(py::init<uintptr_t, size_t, RDMAServerConfig>())
      .def("register_buffer",
           py::overload_cast<void *, size_t>(&RDMAServer::register_buffer),
           "register via raw pointer")
      .def("register_buffer",
           py::overload_cast<uintptr_t, size_t>(&RDMAServer::register_buffer),
           "register via pyint interface")
      .def("register_buffer",
           py::overload_cast<torch::Tensor>(&RDMAServer::register_buffer),
           "register via torch tensor interface")
      .def("serve", &RDMAServer::serve)
      .def("stop", &RDMAServer::stop);
}
} // namespace gear::rdma