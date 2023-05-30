#include <algorithm>

#include "common/string_format.h"
#include "common/tensor.h"
#include "debug.h"
#include "macros.h"
#include "rdma/rdma_client.h"

namespace gear::rdma {
// ======== struct NodeBrief =========
NodeBrief::NodeBrief(int rank, std::string addr, int port)
    : node_rank(rank), addr(addr), port(port) {}

std::string NodeBrief::to_string() {
  return string_format(
      "<RDMAClient::NodeBrief, node_rank: %d, addr: %s, port %d>",
      this->node_rank, this->addr, this->port);
}

// ======== class Pipe =========
Pipe::Pipe(std::string server_addr, int port, int num_qp, int micro_batch_size)
    : released(false), context(nullptr), qpFactory(nullptr) {
  this->addr = server_addr;
  this->port = port;
  this->num_qp = num_qp;
  this->micro_batch_size = micro_batch_size;
  this->reqs.resize(1);
}

Pipe::Pipe(const NodeBrief &brief, int num_qp, int micro_batch_size)
    : Pipe(brief.addr, brief.port, num_qp, micro_batch_size) {}

Pipe::Pipe(const std::shared_ptr<NodeBrief> brief, int num_qp,
           int micro_batch_size)
    : Pipe(brief->addr, brief->port, num_qp, micro_batch_size) {}

void Pipe::release() {
  for (size_t i = 0; i < this->qps.size(); ++i)
    delete this->qps[i];
  for (size_t i = 0; i < this->req_tokens.size(); ++i)
    delete this->req_tokens[i];

  this->qps.resize(0);
  this->req_tokens.resize(0);
  this->remote_tokens.resize(0);
  this->released = true;
}

Pipe::~Pipe() {
  if (!this->released)
    this->release();

  // deleted in ~RDMAClient
  qpFactory = nullptr;
  context = nullptr;
}

void Pipe::read(infinity::memory::Buffer *local_buffer,
                std::vector<int64_t> remote_offsets,
                std::vector<int64_t> local_offsets, int64_t stride) {
  size_t batch_size = MIN(remote_offsets.size(), local_offsets.size());
  this->read(local_buffer, batch_size, remote_offsets.data(),
             local_offsets.data(), stride);
}

void Pipe::read(infinity::memory::Buffer *local_buffer, int64_t batch_size,
                int64_t *remote_offsets, int64_t *local_offsets,
                int64_t stride) {
  int iter_stride = this->num_qp * this->micro_batch_size;
  int num_iter = (batch_size + iter_stride - 1) / iter_stride;
  int64_t *strides = reinterpret_cast<int64_t *>(
      malloc(this->micro_batch_size * sizeof(int64_t)));
  for (int i = 0; i < this->micro_batch_size; ++i) {
    strides[i] = stride;
  }

  GEAR_DEBUG_PRINT("<Pipe> iter stride: %d, num_iter: %d.\n", iter_stride,
                   num_iter)

  int expected_signals = 0;
  for (int i = 0; i < num_iter; ++i) {
    for (int qp_idx = 0; qp_idx < num_qp; ++qp_idx) {
      int start_idx = i * iter_stride + qp_idx * this->micro_batch_size;
      int payload = MIN(this->micro_batch_size, MAX(batch_size - start_idx, 0));
      if (payload == 0)
        break;
      GEAR_DEBUG_PRINT("<Pipe> iter: %d, qp_index:%d, start_index: %d, "
                       "payload: %d, stride[0]: %ld,  issue multiRead.\n",
                       i, qp_idx, start_idx, payload, strides[0])
      this->qps[qp_idx]->multiRead(
          payload, local_buffer, this->remote_tokens[qp_idx],
          &local_offsets[start_idx], &remote_offsets[start_idx], strides,
          infinity::queues::OperationFlags(), this->req_tokens[qp_idx],
          this->reqs);
      expected_signals += 1;

      if (expected_signals > 128 || i == num_iter - 1) {
        int polled = this->context->batchPollSend(
            expected_signals,
            i == (num_iter - 1) // non-strict checks for intermediate signals
        );
        expected_signals -= polled;
      }
    }
  }
  GEAR_DEBUG_PRINT("<Pipe> poll end, free space")
  free(reinterpret_cast<void *>(strides));
}

void Pipe::connect(CustomContext *context, CustomQueuePairFactory *qpFactory) {
  this->context = context;
  this->qpFactory = qpFactory;

  this->qps.reserve(num_qp);
  this->req_tokens.reserve(num_qp);
  this->remote_tokens.reserve(num_qp);
  for (int i = 0; i < this->num_qp; ++i) {
    CustomQueuePair *qp =
        this->qpFactory->connectToRemoteHost(this->addr.c_str(), this->port);

    GEAR_DEBUG_PRINT(
        "<Pipe>: (remote addr: %s, remote port %d) qp %d(/%d) connected.\n",
        this->addr.c_str(), this->port, i, num_qp);
    this->qps.push_back(qp);
    this->req_tokens.push_back(new infinity::requests::RequestToken(
        dynamic_cast<infinity::core::Context *>(this->context)));
    this->remote_tokens.push_back(
        reinterpret_cast<infinity::memory::RegionToken *>(qp->getUserData()));
  }

#ifdef GEAR_VERBOSE_DEBUG_ON
  fprintf(stdout, "Enumerating remote tokens...\n");
  fflush(stdout);
  for (size_t i = 0; i < this->remote_tokens.size(); ++i) {
    auto rtoken = this->remote_tokens[i];
    fprintf(stdout, "Remote token %ld(/%ld):", i, this->remote_tokens.size());
    fflush(stdout);

    // fprintf(stdout, "\tremote addr %lld:", rtoken->getAddress());
    // fflush(stdout);

    fprintf(stdout, "\tremote key %u:", rtoken->getRemoteKey());
    fflush(stdout);
  }
#endif
}

// ======== class RDMAClient =========
RDMAClient::RDMAClient(int local_node_rank, std::vector<NodeBrief> briefs,
                       int num_qp)
    : local_node_rank(local_node_rank), context(nullptr) {
  const int micro_batch_size = 32;

  size_t node_world_size = briefs.size();
  this->briefs.reserve(node_world_size);
  this->pipes.reserve(node_world_size);

  for (size_t i = 0; i < node_world_size; ++i) {
    this->briefs.emplace_back(std::make_shared<NodeBrief>(briefs[i]));
  }

  std::sort(this->briefs.begin(), this->briefs.end(), NodeCmp);
  for (size_t i = 0; i < node_world_size; ++i) {
    std::cout << i << std::endl;
    this->pipes.push_back(
        std::make_shared<Pipe>(this->briefs[i], num_qp, micro_batch_size));
  }

  // ib init
  this->context = new CustomContext();
  this->qpFactory = new CustomQueuePairFactory(this->context);
}

RDMAClient::~RDMAClient() {
  for (std::shared_ptr<Pipe> &p : this->pipes) {
    p->release();
  }
  this->deregister_buffer();
  if (this->qpFactory != nullptr) {
    delete this->qpFactory;
  }
  if (this->context != nullptr) {
    delete this->context;
  }
}

void RDMAClient::connect() {
  for (size_t i = 0; i < this->pipes.size(); ++i) {
    // local loop skipped(fetch data directly via shared-memory)
    // if (this->briefs[i]->node_rank != local_node_rank) {
    //   this->pipes[i]->connect(this->context, this->qpFactory);
    // }

    this->pipes[i]->connect(this->context, this->qpFactory);
  }
}

size_t RDMAClient::register_buffer(void *buf, size_t size) {
  this->deregister_buffer();
  this->buffers.emplace_back(
      new infinity::memory::Buffer(this->context, buf, size));
  this->local_tokens.emplace_back(this->buffers[0]->createRegionToken());
  return this->buffers.size();
}

size_t RDMAClient::register_buffer(uintptr_t buf, size_t size) {
  return this->register_buffer(reinterpret_cast<void *>(buf), size);
}

size_t RDMAClient::register_buffer(torch::Tensor t) {
  size_t size = t.nbytes();
  void *buf;
  convert_tensor_data_pointer(t, &buf);
  return this->register_buffer(buf, size);
}

size_t RDMAClient::register_buffer(std::vector<void *> bufs,
                                   std::vector<size_t> sizes) {
  this->deregister_buffer();
  for (size_t i = 0; i < bufs.size(); ++i) {
    auto mr = new infinity::memory::Buffer(this->context, bufs[i], sizes[i]);
    this->buffers.emplace_back(mr);
    this->local_tokens.emplace_back(mr->createRegionToken());
  }
  return this->buffers.size();
}

size_t RDMAClient::register_buffer(std::vector<uintptr_t> bufs,
                                   std::vector<size_t> sizes) {
  this->deregister_buffer();
  for (size_t i = 0; i < bufs.size(); ++i) {
    auto mr = new infinity::memory::Buffer(
        this->context, reinterpret_cast<void *>(bufs[i]), sizes[i]);
    this->buffers.emplace_back(mr);
    this->local_tokens.emplace_back(mr->createRegionToken());
  }
  return this->buffers.size();
}

size_t RDMAClient::register_buffer(std::vector<torch::Tensor> ts) {
  this->deregister_buffer();
  for (size_t i = 0; i < ts.size(); ++i) {
    size_t size = ts[i].nbytes();
    void *buf;
    convert_tensor_data_pointer(ts[i], &buf);
    auto mr = new infinity::memory::Buffer(this->context,
                                           reinterpret_cast<void *>(buf), size);
    this->buffers.emplace_back(mr);
    this->local_tokens.emplace_back(mr->createRegionToken());
  }
  return this->buffers.size();
}

size_t RDMAClient::deregister_buffer() {
  size_t size = this->buffers.size();
  for (auto &token : this->local_tokens) {
    delete token;
  }
  for (auto &buffer : this->buffers) {
    delete buffer;
  }
  this->buffers.resize(0);
  this->local_tokens.resize(0);
  return size;
}

void RDMAClient::read(int pipe, int buffer, std::vector<int64_t> remote_offsets,
                      std::vector<int64_t> local_offsets, int64_t stride) {
  GEAR_ASSERT(this->buffers.size() > static_cast<size_t>(buffer),
              "<RDMAClient> Buffer index out of index");
  this->pipes[pipe]->read(this->buffers[buffer], remote_offsets, local_offsets,
                          stride);
}

void RDMAClient::read(int pipe, int buffer, torch::Tensor remote_offsets,
                      torch::Tensor local_offsets, int64_t stride) {
  GEAR_ASSERT(this->buffers.size() > static_cast<size_t>(buffer),
              "<RDMAClient> Buffer index out of index");
  GEAR_ASSERT(remote_offsets.nbytes() == local_offsets.nbytes(),
              "offsets size mismatch");
  GEAR_ASSERT(remote_offsets.options().dtype() == torch::kInt64 &&
                  local_offsets.options().dtype() == torch::kInt64,
              "offsets dtype should be int64");
  void *remote_offset_ptr, *local_offset_ptr;
  int64_t size = remote_offsets.nbytes() >> 3;
  convert_tensor_data_pointer(remote_offsets, &remote_offset_ptr);
  convert_tensor_data_pointer(local_offsets, &local_offset_ptr);

  this->pipes[pipe]->read(this->buffers[buffer], size,
                          reinterpret_cast<int64_t *>(remote_offset_ptr),
                          reinterpret_cast<int64_t *>(local_offset_ptr),
                          stride);
}

std::vector<std::shared_ptr<Pipe>> RDMAClient::get_pipes() {
  std::vector<std::shared_ptr<Pipe>> ret;
  for (size_t i = 0; i < this->pipes.size(); ++i) {
    ret.push_back(this->pipes[i]);
  }
  return ret;
}

} // namespace gear::rdma