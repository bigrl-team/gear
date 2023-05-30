#pragma once

#include <infinity/infinity.h>
#include <infinity/queues/QueuePairFactory.h>

#include "rdma/ib_context.h"
#include "rdma/queue_pair.h"

namespace gear::rdma {
class CustomQueuePairFactory : public infinity::queues::QueuePairFactory {
public:
  CustomQueuePairFactory(CustomContext *ctx);

  CustomQueuePair *connectToRemoteHost(const char *hostAddress, uint16_t port,
                                       void *userData = NULL,
                                       uint32_t userDataSizeInBytes = 0);

protected:
  CustomContext *custom_context;
};
} // namespace gear::rdma