#pragma once

#include <infinity/core/Context.h>

namespace gear::rdma {
class CustomQueuePairFactory;

class CustomContext : public infinity::core::Context {
  friend class CustomQueuePairFactory;

public:
  CustomContext() = default;

  int batchPollSend(int expected_num, bool blocking);
};
} // namespace gear::rdma