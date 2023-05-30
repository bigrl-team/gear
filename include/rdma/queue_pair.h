#pragma once

#include <vector>

#include <infinity/infinity.h>
#include <infinity/queues/QueuePair.h>

#include "rdma/ib_context.h"

namespace gear::rdma {
class Pipe;
class CustomQueuePairFactory;

struct RequestList {
  std::vector<struct ibv_sge> sges;
  std::vector<struct ibv_send_wr> wrs;

  void resize(int size);

  void reset();
};

class CustomQueuePair : public infinity::queues::QueuePair {

  friend class Pipe;
  friend class CustomQueuePairFactory;

public:
  CustomQueuePair(CustomContext *ctx);

  void multiRead(int num_elem, infinity::memory::Buffer *local_buffer,
                 infinity::memory::RegionToken *remote_token,
                 int64_t *local_offsets, int64_t *remote_offsets,
                 int64_t *strides, infinity::queues::OperationFlags send_flags,
                 infinity::requests::RequestToken *request_token,
                 RequestList &reqs);
};
} // namespace gear::rdma