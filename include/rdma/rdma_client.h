#pragma once

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <infinity/infinity.h>
#include <torch/extension.h>

#include "rdma/ib_context.h"
#include "rdma/queue_pair.h"
#include "rdma/queue_pair_factory.h"

namespace gear::rdma {


struct NodeBrief {
  int node_rank;
  std::string addr;
  int port;

  NodeBrief() = default;

  NodeBrief(const NodeBrief &t) {
    this->node_rank = t.node_rank;
    this->addr = t.addr;
    this->port = t.port;
  }

  NodeBrief(int rank, std::string addr, int port);

  std::string to_string();
};

struct {
  bool operator()(NodeBrief a, NodeBrief b) {
    return a.node_rank < b.node_rank;
  }

  bool operator()(std::shared_ptr<NodeBrief> a, std::shared_ptr<NodeBrief> b) {
    return a->node_rank < b->node_rank;
  }

  bool operator()(NodeBrief *a, NodeBrief *b) {
    return a->node_rank < b->node_rank;
  }
} NodeCmp;

class Pipe {
public:
  Pipe(std::string server_addr, int port, int num_qp, int micro_batch_size);

  Pipe(const NodeBrief &brief, int num_qp, int micro_batch_size);

  Pipe(const std::shared_ptr<NodeBrief> brief, int num_qp,
       int micro_batch_size);

  void release();

  ~Pipe();

  void read(infinity::memory::Buffer *local_buffer,
            std::vector<int64_t> remote_offsets,
            std::vector<int64_t> local_offsets, int64_t stride);

  void read(infinity::memory::Buffer *local_buffer, int64_t batch_size,
            int64_t *remote_offsets, int64_t *local_offsets, int64_t stride);

  void connect(CustomContext *context, CustomQueuePairFactory *qpFactory);

private:
  bool released;
  std::string addr;
  int port;
  int num_qp;
  int micro_batch_size;

  RequestList reqs;

  CustomContext *context;
  CustomQueuePairFactory *qpFactory;
  std::vector<CustomQueuePair *> qps;
  std::vector<infinity::requests::RequestToken *> req_tokens;
  std::vector<infinity::memory::RegionToken *> remote_tokens;
};

class RDMAClient {
public:
  RDMAClient(int local_node_rank, std::vector<NodeBrief> briefs, int num_qp);

  ~RDMAClient();

  void connect();

  size_t register_buffer(void *buf, size_t size);

  size_t register_buffer(uintptr_t buf, size_t size);

  size_t register_buffer(torch::Tensor t);

  size_t register_buffer(std::vector<void *> bufs, std::vector<size_t> sizes);

  size_t register_buffer(std::vector<uintptr_t> bufs,
                         std::vector<size_t> sizes);

  size_t register_buffer(std::vector<torch::Tensor> ts);

  size_t deregister_buffer();

  void read(int pipe, int buffer, std::vector<int64_t> remote_offsets,
            std::vector<int64_t> local_offsets, int64_t stride);

  void read(int pipe, int buffer, torch::Tensor remote_offsets,
            torch::Tensor local_offsets, int64_t stride);

  std::vector<std::shared_ptr<Pipe>> get_pipes();

private:
  int local_node_rank;
  std::vector<std::shared_ptr<NodeBrief>> briefs;
  std::vector<std::shared_ptr<Pipe>> pipes;

  CustomContext *context;
  CustomQueuePairFactory *qpFactory;
  std::vector<infinity::memory::Buffer *> buffers;
  std::vector<infinity::memory::RegionToken *> local_tokens;
};


} // namespace gear::rdma