#pragma once

#include <memory>
#include <thread>
#include <torch/extension.h>
#include <vector>

#include <infinity/infinity.h>

#include "common/span.h"

using gear::common::Uint8Span;

namespace gear::rdma {
namespace py = pybind11;

struct RDMAServerConfig {
  int port;
  int num_clients;
  int qp_per_client;
};

class RDMAServer {
public:
  RDMAServer(RDMAServerConfig config);

  RDMAServer(void *buf, size_t size, RDMAServerConfig config);

  RDMAServer(uintptr_t buf, int64_t size, RDMAServerConfig config);

  void register_buffer(void *buf, size_t size);

  void register_buffer(uintptr_t buf, size_t size);

  void register_buffer(Uint8Span s);

  void register_buffer(torch::Tensor t);

  void serve();

  void stop();

  ~RDMAServer();

private:
  bool lazy_init;
  RDMAServerConfig config;

  std::unique_ptr<std::thread> worker;

  infinity::core::Context *context;
  infinity::queues::QueuePairFactory *qpFactory;
  infinity::memory::Buffer *buffer;
  infinity::memory::RegionToken *token;
  std::vector<infinity::queues::QueuePair *> qps;

  static void
  service_wait_conn_loop(std::vector<infinity::queues::QueuePair *> &qps,
                         infinity::queues::QueuePairFactory *qpFactory,
                         infinity::memory::RegionToken *token, int num_conns);

  void inline init_ib(int port);

  void inline init_mr(void *buf, size_t size);
};

void register_rdma_server(py::module &m);
} // namespace gear::rdma