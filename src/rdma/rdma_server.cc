#include <infiniband/verbs.h>
#include <stdexcept>

#include "common/span.h"
#include "common/tensor.h"
#include "debug.h"
#include "rdma/rdma_server.h"

namespace gear::rdma {
void RDMAServer::service_wait_conn_loop(
    std::vector<infinity::queues::QueuePair *> &qps,
    infinity::queues::QueuePairFactory *qpFactory,
    infinity::memory::RegionToken *token, int num_conns) {
  qps.reserve(num_conns);
  for (int i = 0; i < num_conns; ++i) {
    qps.push_back(qpFactory->acceptIncomingConnection(
        reinterpret_cast<void *>(token),
        sizeof(infinity::memory::RegionToken)));
  }
}

void inline RDMAServer::init_ib(int port) {
  this->context = new infinity::core::Context();
  this->qpFactory = new infinity::queues::QueuePairFactory(this->context);
  this->qpFactory->bindToPort(port);
}

void inline RDMAServer::init_mr(void *buf, size_t size) {
  GEAR_ASSERT(this->context != nullptr,
              "<RDMAServer::init_mr> method should be called after "
              "instantiation of buffer.")
  this->buffer = new infinity::memory::Buffer(this->context, buf, size);
  this->buffer->getMemoryRegionType();
  this->buffer->getAddress();
  this->buffer->getLocalKey();
  this->buffer->getRemoteKey();
  this->token = this->buffer->createRegionToken();
}

RDMAServer::RDMAServer(RDMAServerConfig config)
    : lazy_init(true), worker(nullptr), context(nullptr), qpFactory(nullptr),
      buffer(nullptr), token(nullptr) {

  this->config = config;
  this->init_ib(config.port);
}

RDMAServer::RDMAServer(void *buf, size_t size, RDMAServerConfig config)
    : RDMAServer(config) {
  this->lazy_init = false;
  this->init_mr(buf, size);
}

RDMAServer::RDMAServer(uintptr_t buf, int64_t size, RDMAServerConfig config)
    : RDMAServer(reinterpret_cast<void *>(buf), size, config) {}

void RDMAServer::register_buffer(void *buf, size_t size) {
  GEAR_ASSERT(this->lazy_init,
              "<RDMAServer::register_buffer> 'register_buffer' method can "
              "only be called once when RDMAServer is lazy inited(does not "
              "instantiated with buffer pointer and size)")
  this->lazy_init = false;
  this->init_mr(buf, size);
}

void RDMAServer::register_buffer(uintptr_t buf, size_t size) {
  this->register_buffer(reinterpret_cast<void *>(buf), size);
}

void RDMAServer::register_buffer(Uint8Span s) {
  this->register_buffer(reinterpret_cast<void *>(s.ptr),
                        static_cast<size_t>(s.size));
}

void RDMAServer::register_buffer(torch::Tensor t) {
  size_t size = t.nbytes();
  void *buf = nullptr;
  convert_tensor_data_pointer(t, &buf);
  if (buf == nullptr) {
    throw std::runtime_error("<RDMAServer> Error registering data pointer");
  }
  return this->register_buffer(buf, size);
}

void RDMAServer::serve() {
  GEAR_ASSERT(this->buffer != nullptr,
              "<RDMAServer::serve> 'serve' method should be called after "
              "buffer registration");
  GEAR_ASSERT(
      this->worker == nullptr,
      "<RDMAServer::serve> background service thread has already initialized");
  this->worker = std::make_unique<std::thread>(
      this->service_wait_conn_loop, std::ref(this->qps), this->qpFactory,
      this->token, this->config.num_clients * this->config.qp_per_client);
}

void RDMAServer::stop() {
  GEAR_ASSERT(this->worker != nullptr,
              "<RDMAServer::serve> 'stop' method should be called after "
              "server method");
  this->worker->join();
  this->worker = nullptr;
}

RDMAServer::~RDMAServer() {
  this->stop();

  for (size_t i = 0; i < qps.size(); ++i) {
    delete this->qps[i];
  }
  if (this->qpFactory != nullptr)
    delete this->qpFactory;
  if (this->token != nullptr)
    delete this->token;
  if (this->buffer != nullptr)
    delete this->buffer;
  if (this->context != nullptr)
    delete this->context;
}

} // namespace gear::rdma