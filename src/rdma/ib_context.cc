#include "debug.h"
#include "rdma/ib_context.h"

namespace gear::rdma {
int CustomContext::batchPollSend(int expected_num, bool blocking) {
  struct ibv_wc wc;
  int succ = 0;
  while (succ < expected_num) {
    GEAR_DEBUG_PRINT("BatchPollSend succ count %d expected %d, cq %p.\n", succ,
                     expected_num, this->ibvSendCompletionQueue);
    int num;
    do {
      num = ibv_poll_cq(this->ibvSendCompletionQueue, 1, &wc);
    } while (num == 0);

    if (num < 0) {
      GEAR_ERROR(
          "<RDMA::batch_poll_send> Batch Polling failed with return %d\n",
          succ);
      continue;
    }

    GEAR_DEBUG_PRINT("BatchPollSend status check %d.\n", succ);
    if (wc.status != ibv_wc_status::IBV_WC_SUCCESS) {
      GEAR_ERROR("<RDMA::batch_poll_send> Request %lu failed with code %d\n",
                 wc.wr_id, wc.status);
      continue;
    }

    GEAR_DEBUG_PRINT("BatchPollSend status check %d passed.\n", succ);
    succ += num;
    if (!blocking) {
      break;
    }
  }

  return succ;
}
} // namespace gear::rdma