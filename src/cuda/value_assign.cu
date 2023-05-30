#include "cuda/value_assign.cuh"
#include "macros.h"

const int default_value_assign_unroll = 16; // float4

template <int elem_size, int unroll>
__global__ void value_assign(int num_elem, int workload, void *src, void *dst,
                             int src_stride, size_t dst_stride) {
  int num_thread = (num_elem + workload - 1) / workload;
  CUDA_1D_KERNEL_LOOP(idx, num_thread) {
    int start_idx = workload * idx;
    int end_idx = MIN(start_idx + workload, num_elem);
    int valid_workload = end_idx - start_idx;

    for (int i = 0; i < valid_workload; ++i) {
      char *assign_from =
          reinterpret_cast<char *>(src) + (start_idx + i) * src_stride;
      char *assign_to =
          reinterpret_cast<char *>(dst) + (start_idx + i) * dst_stride;

      for (int j = 0; j < elem_size / unroll; ++j) {
        reinterpret_cast<float4 *>(assign_to)[j] =
            reinterpret_cast<float4 *>(assign_from)[j];
      }
      for (int j = (elem_size / unroll) * unroll; j < elem_size; ++j) {
        assign_to[j] = assign_from[j];
      }
    }
  }
}

namespace gear::cuda {
void launch_value_assign() {}

} // namespace gear::cuda