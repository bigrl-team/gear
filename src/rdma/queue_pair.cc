#include "debug.h"
#include "macros.h"
#include "rdma/queue_pair.h"

namespace gear::rdma {
void RequestList::resize(int size) {
  this->sges.resize(size);
  this->wrs.resize(size);
}

void RequestList::reset() {
  memset(reinterpret_cast<void *>(sges.data()), 0,
         sges.size() * sizeof(ibv_sge));
  memset(reinterpret_cast<void *>(wrs.data()), 0,
         wrs.size() * sizeof(ibv_send_wr));
}

CustomQueuePair::CustomQueuePair(CustomContext *ctx)
    : QueuePair(dynamic_cast<infinity::core::Context *>(ctx)) {}

void CustomQueuePair::multiRead(int num_elem,
                                infinity::memory::Buffer *local_buffer,
                                infinity::memory::RegionToken *remote_token,
                                int64_t *local_offsets, int64_t *remote_offsets,
                                int64_t *strides,
                                infinity::queues::OperationFlags send_flags,
                                infinity::requests::RequestToken *request_token,
                                RequestList &reqs) {
  if (request_token != nullptr) {
    request_token->reset();
    request_token->setRegion(local_buffer);
  }
  reqs.resize(num_elem);
  reqs.reset();

  struct ibv_send_wr *bad_wr;

  GEAR_DEBUG_PRINT("<CustomQueuePair> num_elem: %d.\n", num_elem)

  for (int i = 0; i < num_elem; ++i) {
    GEAR_DEBUG_PRINT("<CustomQueuePair> building %d-th(/%d) wr ......\n", i,
                       num_elem)

    reqs.sges[i].addr = local_buffer->getAddress() + local_offsets[i];
    reqs.sges[i].length = strides[i];
    reqs.sges[i].lkey = local_buffer->getLocalKey();
    GEAR_DEBUG_PRINT("<CustomQueuePair> building %d-th(/%d) wr - local field "
                       "set(laddr: %ld, stride: %d, lkey: %d).\n",
                       i, num_elem, reqs.sges[i].addr, reqs.sges[i].length,
                       reqs.sges[i].lkey)

    reqs.wrs[i].sg_list = reqs.sges.data() + i;
    reqs.wrs[i].num_sge = 1;
    reqs.wrs[i].opcode = ibv_wr_opcode::IBV_WR_RDMA_READ;
    reqs.wrs[i].next = (i == num_elem - 1) ? nullptr : &reqs.wrs[i + 1];
    reqs.wrs[i].send_flags = send_flags.ibvFlags();
    // reqs.wrs[i].next = (i == num_elem - 1) ? nullptr : &reqs.wrs[i + 1];
    GEAR_DEBUG_PRINT("<CustomQueuePair> building %d-th(/%d) wr - wr "
                       "attributes set (%p, %d, %ld, %p).\n",
                       i, num_elem, reqs.wrs[i].sg_list, reqs.wrs[i].num_sge,
                       local_offsets[i], reqs.wrs[i].next)

    reqs.wrs[i].wr.rdma.remote_addr =
        remote_token->getAddress() + remote_offsets[i];
    reqs.wrs[i].wr.rdma.rkey = remote_token->getRemoteKey();
    GEAR_DEBUG_PRINT("<CustomQueuePair> building %d-th(/%d) wr - remote "
                       "fields set (%ld, %ld, %u).\n",
                       i, num_elem, remote_token->getAddress(),
                       remote_offsets[i], remote_token->getRemoteKey())
  }
  if (request_token != nullptr) {
    GEAR_DEBUG_PRINT("<CustomQueuePair> signaling request.\n")
    reqs.wrs[num_elem - 1].wr_id = reinterpret_cast<uint64_t>(request_token);
    reqs.wrs[num_elem - 1].send_flags |= ibv_send_flags::IBV_SEND_SIGNALED;
  }

  GEAR_DEBUG_PRINT("<CustomQueuePair> post_send.\n")
  int returnValue = ibv_post_send(this->ibvQueuePair, &reqs.wrs[0], &bad_wr);

  GEAR_DEBUG_PRINT("<CustomQueuePair> post_send end.\n")
  GEAR_ASSERT(
      returnValue == 0,
      "<CustomQueuePair::multiRead> Posting read request failed. %s.\n",
      strerror(errno));
}
} // namespace gear::rdma