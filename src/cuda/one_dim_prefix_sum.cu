#include <cooperative_groups.h>
#include <math.h>

#include "config.h"
#include "cuda/one_dim_prefix_sum.cuh"
#include "debug.h"
#include "kernel_launchers.h"
#include "macros.h"

namespace cg = cooperative_groups;
constexpr int max_block_threads = 512;
const int STATUS_X = 0;
const int STATUS_A = 1;
const int STATUS_P = 2;

template <int tileSize>
__device__ __forceinline__ float
reduction_tile_exclusive_prefix_sum(cg::thread_block_tile<tileSize> g,
                                    float val) {
  float inclusive_prefix = val;
  int lane = g.thread_rank();
  float tmp;
  for (int i = 1; i < g.size(); i <<= 1) {
    tmp = g.shfl_up(inclusive_prefix, i);
    if (lane >= i) {
      inclusive_prefix += tmp;
    }
  }
  return inclusive_prefix;
}

__device__ __forceinline__ float
scan_float_prefix_sum(int stride, float *inputs, float *outputs) {
  if (stride <= 0) {
    return 0;
  }
  float prefix_sum = 0;
  float4 *inputs_cast4 = reinterpret_cast<float4 *>(inputs);
  float4 *outputs_cast4 = reinterpret_cast<float4 *>(outputs);
  for (int i = 0; i < stride / 4; ++i) {
    float4 val4 = inputs_cast4[i];
    val4.x += prefix_sum;
    val4.y += val4.x;
    val4.z += val4.y;
    val4.w += val4.z;

    prefix_sum = val4.w;
    outputs_cast4[i] = val4;
  }

  for (int i = (stride / 4) * 4; i < stride; ++i) {
    float val = inputs[i];
    prefix_sum += val;
    outputs[i] = prefix_sum;
  }
  return prefix_sum;
}

__device__ __forceinline__ void vectorized_add(int stride, float *inputs,
                                               float operand, float *outputs) {
  if (stride <= 0)
    return;

  for (int i = 0; i < stride / 4; ++i) {
    float4 val4 = reinterpret_cast<float4 *>(inputs)[i];
    val4.x += operand;
    val4.y += operand;
    val4.z += operand;
    val4.w += operand;
    reinterpret_cast<float4 *>(outputs)[i] = val4;
  }

  for (int i = (stride / 4) * 4; i < stride; ++i) {
    outputs[i] = inputs[i] + operand;
  }
}

__device__ __forceinline__ bool is_last_thread(dim3 thread_index,
                                               dim3 block_dim) {
  return thread_index.x == (block_dim.x - 1);
}

__device__ __forceinline__ bool is_first_partition(int pidx) {
  return pidx == 0;
}

__device__ __forceinline__ void
release_aggregate(float agg, volatile PartitionDescriptor<float> *descs,
                  int pidx) {
  volatile PartitionDescriptor<float> *desc = descs + pidx;
  desc->aggregate = agg;
  __threadfence();

  desc->inc_prefix = agg;
  __threadfence();

  desc->status = is_first_partition(pidx) ? STATUS_P : STATUS_A;
}

__device__ __forceinline__ float
lookback(float agg, volatile PartitionDescriptor<float> *descs, int pidx,
         dim3 thread_index, dim3 block_dim) {
  float exclusive_prefix = 0;
  if (pidx > 0) {
    int prev = pidx - 1;
    volatile PartitionDescriptor<float> *prev_desc = descs + prev;
    int status = 0;
    do {
      status = prev_desc->status;
      __threadfence();
    } while (status < STATUS_P);
    exclusive_prefix = prev_desc->inc_prefix;
    __threadfence();

    volatile PartitionDescriptor<float> *desc = descs + pidx;
    desc->inc_prefix = agg + exclusive_prefix;
    __threadfence();

    desc->status = STATUS_P;
    __threadfence();
  }
  return exclusive_prefix;
}

__global__ void one_dimension_prefix_sum(int length, float *vals,
                                         float *outputs,
                                         volatile void *d_temp_storage) {

  __shared__ float temp_results[max_block_threads * unroll_factor];
  __shared__ float warp_prefix[MAX_WARP_NUM + 1];
  warp_prefix[0] = 0;

  volatile PartitionDescriptor<float> *descs =
      reinterpret_cast<volatile PartitionDescriptor<float> *>(d_temp_storage);

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int warp_num = (blockDim.x + (int)WARP_SIZE - 1) >> WARP_SIZE_BITS;
  int warp_id = threadIdx.x >> WARP_SIZE_BITS;
  int lane = threadIdx.x & 0x1f;
  int partition_index = blockIdx.x;

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  int base_offset = idx * unroll_factor;
  int valid_workload = (length - base_offset) > 0 ? length - base_offset : 0;
  int workload =
      valid_workload > unroll_factor ? unroll_factor : valid_workload;
  float inclusive_prefix = scan_float_prefix_sum(workload, vals + base_offset,
                                                 temp_results + base_offset);
  // float inclusive_prefix = 0;
  // float4 *inputs_cast4 = reinterpret_cast<float4 *>(vals + base_offset);
  // float4 *outputs_cast4 =
  //     reinterpret_cast<float4 *>(temp_results + base_offset);
  // for (int i = 0; i < workload / 4; ++i) {
  //   float4 val4 = inputs_cast4[i];
  //   val4.x += inclusive_prefix;
  //   val4.y += val4.x;
  //   val4.z += val4.y;
  //   val4.w += val4.z;

  //   inclusive_prefix = val4.w;
  //   outputs_cast4[i] = val4;
  // }

  // for (int i = (workload / 4) * 4; i < workload; ++i) {
  //   float val = vals[base_offset + i];
  //   inclusive_prefix += val;
  //   temp_results[base_offset + i] = inclusive_prefix;
  // }

  // inner-warp reduction
  g.sync();
  float inclusive_thread_prefix_in_tile =
      reduction_tile_exclusive_prefix_sum<WARP_SIZE>(g, inclusive_prefix);
  vectorized_add(workload, temp_results + base_offset,
                 inclusive_thread_prefix_in_tile - inclusive_prefix,
                 temp_results + base_offset);
  printf("lane %d thread prefix %f tile prefix %f\n", lane, inclusive_prefix,
         inclusive_thread_prefix_in_tile);

  if (lane == (int)WARP_SIZE - 1 || lane == blockDim.x - 1) {
    if (warp_id < warp_num) {
      warp_prefix[warp_id + 1] = inclusive_thread_prefix_in_tile;
    } else {
      warp_prefix[warp_id + 1] = 0;
    }
  }

  b.sync();
  // warp-level reduction
  if (warp_id == 0) {
    float exclusive_parition_prefix = 0;
    float inclusive_prefix_in_block = warp_prefix[lane + 1];
    printf("lane %d prefix %f\n", lane, inclusive_prefix_in_block);

    g.sync();
    inclusive_prefix_in_block = reduction_tile_exclusive_prefix_sum<WARP_SIZE>(
        g, inclusive_prefix_in_block);
    warp_prefix[lane + 1] = inclusive_prefix_in_block;
    g.sync();
    printf("after lane %d prefix %f\n", lane, inclusive_prefix_in_block);

    g.sync();
    if (lane == (int)WARP_SIZE - 1) {
      // release block aggregate
      release_aggregate(inclusive_prefix_in_block, descs, partition_index);

      // look-back
      exclusive_parition_prefix =
          lookback(inclusive_prefix_in_block, descs, partition_index, threadIdx,
                   blockDim);
    }

    // broadcast exclusive prefix to all lanes in first warp
    exclusive_parition_prefix =
        __shfl_sync(0xFFFFFFFF, exclusive_parition_prefix, (int)WARP_SIZE - 1,
                    (int)WARP_SIZE);
    warp_prefix[lane] += exclusive_parition_prefix;

    g.sync();
    if (lane < warp_num) {
      int workload_per_thread = (int)WARP_SIZE * unroll_factor;
      int offset = lane * workload_per_thread;
      int workload = (workload_per_thread <= (length - offset))
                         ? workload_per_thread
                         : (length - offset);
      printf("lane %d warp prefix %f workload %d\n", lane, warp_prefix[lane],
             workload);
      vectorized_add(workload, temp_results + offset, warp_prefix[lane],
                     outputs + offset);
    }
  }
}

namespace gear::cuda {
int get_prefix_sum_block_dim(int threads, dim3 block_dim) {
  int num_blocks =
      std::max((int)((threads + block_dim.x - 1) / block_dim.x), 1);
  GEAR_ASSERT(num_blocks <= MAX_NUM_BLOCKS,
                "Required CUDA thread-block exceeds MAX_NUM_BLOCKS, consider "
                "reset the latter to fit the application");
  return num_blocks;
}

void launch_prefix_sum(torch::Tensor input_tensor,
                       torch::Tensor output_tensor) {
  int length = input_tensor.size(0);
  int threads =
      std::max((int)pow(2.0, ceil(log2(((float)length / unroll_factor)))), 1);

  dim3 block_dim =
      std::max(std::min(threads, max_block_threads), (int)WARP_SIZE);
  dim3 grid_dim = get_prefix_sum_block_dim(threads, block_dim);

  void *d_temp_storage_ptr;
  CUDA_CHECK(cudaMalloc(&d_temp_storage_ptr, grid_dim.x * sizeof(float)));

  float *inputs = input_tensor.data_ptr<float>();
  float *outputs = output_tensor.data_ptr<float>();

  auto stream = c10::cuda::getCurrentCUDAStream();
  int shared_memory_size =
      (max_block_threads * unroll_factor + MAX_WARP_NUM + 1) * sizeof(float);
  one_dimension_prefix_sum<<<grid_dim, block_dim, shared_memory_size, stream>>>(
      length, inputs, outputs, d_temp_storage_ptr);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
} // namespace gear::cuda

void register_prefix_sum_kernel_launcher(pybind11::module &m) {
  m.def("prefix_sum", &gear::cuda::launch_prefix_sum);
}