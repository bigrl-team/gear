#include <cooperative_groups.h>
#include <math.h>

#include <stdio.h>

#include "config.h"
#include "kernel_launchers.h"
#include "macros.h"

namespace cg = cooperative_groups;

template <int tileSize>
__device__ __forceinline__ float
reduction_tile_sum(cg::thread_block_tile<tileSize> g, float val) {
  for (int mask = tileSize / 2; mask > 0; mask >>= 1) {
    val += g.shfl_xor(val, mask);
  }
  return val;
}

// one-dimension array sum kernel
__global__ void one_dimension_sum(int length, float *vals, float *sum) {
  __shared__ float partial_sums[MAX_WARP_NUM];

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int warp_num = (blockDim.x + (int)WARP_SIZE - 1) >> WARP_SIZE_BITS;
  int warp_id = threadIdx.x >> WARP_SIZE_BITS;
  int lane = threadIdx.x & 0x1f;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  float4 *val_cast4 = reinterpret_cast<float4 *>(vals);

  float partial_sum = 0;
  CUDA_1D_KERNEL_LOOP(j, length / unroll_factor) {
    float4 data = val_cast4[j];
    partial_sum += (data.x + data.y + data.z + data.w);
  }

  int stride = blockDim.x * gridDim.x;
  int high_index =
      ((length / unroll_factor / stride) * stride + idx) * unroll_factor;
  if (length < high_index + 4) {
    for (int i = high_index; i < length; ++i) {
      partial_sum += vals[i];
    }
  }

  g.sync();
  partial_sum = reduction_tile_sum<WARP_SIZE>(g, partial_sum);

  if (warp_num > 1) {
    if (lane == 0) {
      partial_sums[warp_id] = partial_sum;
    }
    b.sync();

    if (warp_id == 0) {
      partial_sum = 0;
      if (lane < warp_num) {
        partial_sum = partial_sums[lane];
      }
      g.sync();

      for (int i = 1; i < warp_num; i <<= 1) {
        partial_sum += g.shfl_xor(partial_sum, i);
      }
    }
  }
  if (threadIdx.x == 0) {
    atomicAdd(sum, partial_sum);
  }
}

namespace gear::cuda {
inline dim3 get_sum_grid_dim(int length, int stride, int threads) {
  int length4 = length / unroll_factor;
  return (int)std::max(
      std::min(((length4 + threads - 1) / threads) / stride, MAX_NUM_BLOCKS),
      1);
}

void launch_sum(torch::Tensor val_tensor, torch::Tensor sum_tensor,
                int stride = 1) {
  const int max_block_threads = 512;
  int length = val_tensor.size(0);
  int length4 = length / unroll_factor;
  float *vals = val_tensor.data_ptr<float>();
  float *sum = sum_tensor.data_ptr<float>();
  CUDA_CHECK(cudaMemset(sum, 0, sizeof(float)));

  int threads = std::max(
      std::min((int)pow(2.0, ceil(log2((float)length4))), max_block_threads),
      1);
  dim3 block_dim = threads;
  dim3 grid_dim = get_sum_grid_dim(length, stride, threads);
  one_dimension_sum<<<grid_dim, block_dim>>>(length, vals, sum);
}
} // namespace gear::cuda

void register_sum_kernel_launcher(pybind11::module &m) {
  m.def("sum", &gear::cuda::launch_sum);
}
