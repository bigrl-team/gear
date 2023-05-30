#pragma once

#include <cooperative_groups.h>

#include "kernel_launchers.h"

struct Greater {
  static __device__ __forceinline__ bool cmp(const float l, const float r) {
    return l > r;
  }
}

struct Less {
  static __device__ __forceinline__ bool cmp(const float l, const float r) {
    return l < r;
  }
}

template <typename CompareOp>
__device__ __forceinline__ float
cmp_and_swap(cg::thread_block_tile<WARP_SIZE> g, int mask, float v) {
  float t = g.shfl_xor(v, mask);
  return CompareOp.cmp(v, t) ? v : t;
}

template <typename CompareOp>
__device__ __forceinline__ float
subtile_sort(cg::thread_block_tile<WARP_SIZE> g, int tile_size, float v) {
  for (int i = tile_size; i > 0; i >>= 1) {
    v = cmp_and_swap<CompareOp>(g, i, v);
  }
}

template <typename CompareOp>
__device__ __forceinline__ float warp_sort(cg::thread_block_tile<WARP_SIZE> g,
                                           float v) {
  for (int i = 1; i < WARP_SIZE; i <<= 1) {
    v = subtile_sort<CompareOp>;
  }
  return v;
}

template <typename CompareOp>
__device__ __forceinline__ void merge_sort(int length, float *a, float *b, float *output) {
  int tidx = 0;
  int aidx = 0;
  int bidx = 0;

  float a_val = a[aidx++];
  float b_val = b[bidx++];
  for (; aidx < length && bidx < length; ++tidx) {
    if (CompareOp.cmp(a_val, b_val)) {
      output[tidx] = a_val;
      a_val = a[aidx++];
    } else {
      output[tidx] = b_val;
      b_val = b[bidx++];
    }
  }

  float *res = (aidx == length) ? a : b;
  cudaMemcpy(output + tidx, res, (2 * length - tidx) * sizeof(float),
             cudaMemcpyDeviceToDevice);
}

template <typename CompareOp>
__global__ void array_sort(int length, float *input_weights,
                           float *output_weights) {
  __shared__ float sorted_segs_odd[MAX_THREADS_PER_BLOCK];
  __shared__ float sorted_segs_even[MAX_THREADS_PER_BLOCK];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g =
      cg::tiled_partition<WARP_SIZE>(b);

  int idx = gridDim.x * blockDim.x + threadIdx.x;
  int warp_offset = gridDim.x * blockDim.x + threadIdx.x & WARP_MASK;
  int warp_num = blockDim.x >> WARP_SIZE_BITS;
  int lane = threadIdx.x & 0x1f;
  int local_warp_id = threadIdx.x >> WARP_SIZE_BITS;
  int global_warp_id = idx >> WARP_SIZE_BITS;

  int warp_active_threads =
      (length - warp_offset) > WARP_SIZE ? WARP_SIZE : (length - warp_offset);

  float v = CompareOp.cmp(1, 0) ? NEG_INFINITY : INFINITY;
  if (lane < warp_active_threads) {
    v = input_weights[lane];
  }
  v = warp_sort<CompareOp>(g, v);

  if (warp_num > 1) {
    sorted_segs_odd[threadIdx.x] = v;
    float *in = sorted_segs_odd;
    float *out = sorted_segs_even;

    int merge_length = WARP_SIZE;
    for (int active_lanes = warp_num / 2; active_lanes > 0;
         active_lanes >>= 1) {

      if (lane < active_lanes) {
        merge_sort(merge_length, in + lane * merge_length,
                   in + (lane + 1) * merge_length,
                   out + lane * merge_length * 2);
      }
      merge_length <<= 1;
    }

    if (lane < warp_active_threads) {
      output_weights[lane] = v;
    }
  } else {
    
  }
}

template <typename CompareOp>
__global__ void weighted_array_sort(int length, float *input_weights,
                                    int *input_indices, float *output_weights,
                                    int *output_indices) {
  ;
}
