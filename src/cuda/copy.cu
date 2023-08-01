#include <cstdint>
#include <math.h>
#include <stdio.h>

#include "common/tensor.h"
#include "config.h"
#include "cuda/copy.cuh"
#include "kernel_launchers.h"
#include "macros.h"

const int copy_unroll_factor = 16;
__global__ void fix_stride_copy_collect(void *src, void *dst, int length,
                                        int64_t *src_offsets,
                                        int64_t *dst_offsets, int64_t stride,
                                        int stride_group_size, int avg_load) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int group_index = idx / stride_group_size;
  if (group_index >= length)
    return;

  int group_rank = idx % stride_group_size;

  int group_offset = group_rank * avg_load;
  int load =
      (stride - group_offset) < avg_load ? (stride - group_offset) : avg_load;
  if (load <= 0)
    return;

  // printf("threadIdx %d load %d ", idx, load);
  char *copy_from =
      reinterpret_cast<char *>(src) + src_offsets[group_index] + group_offset;
  char *copy_to =
      reinterpret_cast<char *>(dst) + dst_offsets[group_index] + group_offset;

  for (int i = 0; i < load / copy_unroll_factor; ++i) {
    reinterpret_cast<float4 *>(copy_to)[i] =
        reinterpret_cast<float4 *>(copy_from)[i];
  }

  int base = (load / copy_unroll_factor) * copy_unroll_factor;
  for (int i = base; i < load; ++i) {
    copy_to[i] = copy_from[i];
  }
}

__global__ void vary_stride_copy_collect(void *src, void *dst, int length,
                                         int64_t *src_offsets,
                                         int64_t *dst_offsets, int64_t *strides,
                                         int stride_group_size) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int group_index = idx / stride_group_size;
  if (group_index >= length)
    return;

  int group_rank = idx % stride_group_size;
  // printf("threadIdx %d gindex %d grank %d", idx, group_index, group_rank);
  int stride = strides[group_index];
  int avg_load = (((stride + stride_group_size - 1) / stride_group_size) +
                  copy_unroll_factor - 1) /
                 copy_unroll_factor;
  avg_load = avg_load * copy_unroll_factor;
  int group_offset = group_rank * avg_load;
  int load =
      (stride - group_offset) < avg_load ? (stride - group_offset) : avg_load;
  if (load <= 0)
    return;

  // printf("threadIdx %d load %d ", idx, load);
  char *copy_from =
      reinterpret_cast<char *>(src) + src_offsets[group_index] + group_offset;
  char *copy_to =
      reinterpret_cast<char *>(dst) + dst_offsets[group_index] + group_offset;

  for (int i = 0; i < load / copy_unroll_factor; ++i) {
    reinterpret_cast<float4 *>(copy_to)[i] =
        reinterpret_cast<float4 *>(copy_from)[i];
  }

  for (int i = int(load / copy_unroll_factor) * copy_unroll_factor / 4;
       i < int(load / 4); ++i) {
    reinterpret_cast<float *>(copy_to)[i] =
        reinterpret_cast<float *>(copy_from)[i];
  }
  for (int i = int(load / 4) * 4; i < load; ++i) {
    copy_to[i] = copy_from[i];
  }
}

__global__ void fused_subcopy_collect(
    void *src /* host data ptr */, void *dst /* device data ptr */,
    int num /* number of subscriptions */,
    int64_t *idxs /* subscribed trajectory indices */,
    int64_t *subs /* subscribed trajectory timesteps */,
    int64_t *lens /* subscribed trajectory lengths */,
    bool pad /* align to the tail */, int64_t ofst /* timestep sub offset */,
    int64_t span /* timestep sub len */,
    int split /* #threads co-work on a sub */,
    int load /* max nbytes copied by a thread */,
    int idx_interval /* trajectory stride */,
    int sub_interval /* timestep stride of the column */,
    int sub_baseofst /* relative offset of a column */,
    int cpy_interval /* copied column slice stride */) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int gidx = idx / split, grank = idx % split;
  if (gidx >= num)
    return;

  int sub = subs[gidx], len = lens[gidx];
  int lsub = MAX(sub + ofst, 0);
  int rsub = MIN(lsub + span, len);
  int stride = sub_interval * (rsub - lsub);
  int npad = (cpy_interval - stride) * int(pad);
  int thread_ofst = grank * load;
  int mload = (stride - thread_ofst) < load ? (stride - thread_ofst) : load;
  if (mload < 0)
    return;

  char *copy_from = reinterpret_cast<char *>(src) + idx_interval * idxs[gidx] +
                    sub_baseofst + lsub * sub_interval + thread_ofst;
  char *copy_to =
      reinterpret_cast<char *>(dst) + cpy_interval * gidx + npad + thread_ofst;

  for (int i = 0; i < mload / copy_unroll_factor; ++i) {
    reinterpret_cast<float4 *>(copy_to)[i] =
        reinterpret_cast<float4 *>(copy_from)[i];
  }

  int base = (mload / copy_unroll_factor) * copy_unroll_factor;
  for (int i = base; i < mload; ++i) {
    copy_to[i] = copy_from[i];
  }
}

namespace gear::cuda {

void launch_fix_stride_copy_collect(gear::memory::MemoryPtr src_mptr,
                                    torch::Tensor dst,
                                    torch::Tensor src_offsets,
                                    torch::Tensor dst_offsets, int64_t stride) {
  const int max_bytes_accessed_per_thread = 64;
  // const int min_bytes_accessed_per_thread = 64;

  int num_src_offsets = src_offsets.size(0);
  int num_dst_offsets = dst_offsets.size(0);
  int num_slices = std::min(num_src_offsets, num_dst_offsets);

  int64_t *src_offsets_ptr = src_offsets.data_ptr<int64_t>();
  int64_t *dst_offsets_ptr = dst_offsets.data_ptr<int64_t>();

  void *src_ptr = src_mptr.get();
  void *dst_ptr;
  convert_tensor_data_pointer(dst, &dst_ptr);

  int num_threads_perform_per_stride =
      (stride + max_bytes_accessed_per_thread - 1) /
      max_bytes_accessed_per_thread;
  // num_threads_perform_per_stride =
  //     (int)to_power_of_two((uint32_t)num_threads_perform_per_stride);
  int avg_load_in_bytes = (((stride + num_threads_perform_per_stride - 1) /
                            num_threads_perform_per_stride) +
                           copy_unroll_factor - 1) /
                          copy_unroll_factor;
  avg_load_in_bytes *= copy_unroll_factor;
  int block_dim = MAX_THREADS_PER_BLOCK;
  int grid_dim =
      (num_slices * num_threads_perform_per_stride + block_dim - 1) / block_dim;
  auto stream = c10::cuda::getCurrentCUDAStream();
  fix_stride_copy_collect<<<grid_dim, block_dim, 0, stream>>>(
      src_ptr, dst_ptr, num_slices, src_offsets_ptr, dst_offsets_ptr, stride,
      num_threads_perform_per_stride, avg_load_in_bytes);
}

void launch_vary_stride_copy_collect(gear::memory::MemoryPtr &src_mptr,
                                     torch::Tensor &dst,
                                     torch::Tensor &src_offsets,
                                     torch::Tensor &dst_offsets,
                                     torch::Tensor &strides,
                                     int64_t max_stride) {
  const int max_bytes_accessed_per_thread = 64;
  int num_src_offsets = src_offsets.size(0);
  int num_dst_offsets = dst_offsets.size(0);
  int num_strides = strides.size(0);
  int num_slices =
      std::min(std::min(num_src_offsets, num_dst_offsets), num_strides);

  void *src_ptr = src_mptr.get();
  void *dst_ptr;
  convert_tensor_data_pointer(dst, &dst_ptr);

  int64_t *src_offsets_ptr = src_offsets.data_ptr<int64_t>();
  int64_t *dst_offsets_ptr = dst_offsets.data_ptr<int64_t>();
  int64_t *strides_ptr = strides.data_ptr<int64_t>();

  int num_threads_perform_per_stride =
      (max_stride + max_bytes_accessed_per_thread - 1) /
      max_bytes_accessed_per_thread;
  int block_dim = MAX_THREADS_PER_BLOCK;
  int grid_dim =
      (num_slices * num_threads_perform_per_stride + block_dim - 1) / block_dim;
  auto stream = c10::cuda::getCurrentCUDAStream();

  // printf("Block Dim %d ,Grid dim %d, max_stride %d, num_slices %d, "
  //        "num_threads_per_stride %d, "
  //        "src count %d, offset %d, "
  //        "dst_count %d, "
  //        "offset %d",
  //        block_dim, grid_dim, max_stride, num_slices,
  //        num_threads_perform_per_stride, num_src_offsets, src_offsets_ptr[0],
  //        num_dst_offsets, dst_offsets_ptr[0]);
  vary_stride_copy_collect<<<grid_dim, block_dim, 0, stream>>>(
      src_ptr, dst_ptr, num_slices, src_offsets_ptr, dst_offsets_ptr,
      strides_ptr, num_threads_perform_per_stride);
}

void launch_fused_subcopy_collect(
    void *src /* host data ptr */, void *dst /* device data ptr */,
    int num /* number of subscriptions */,
    int64_t *idxs /* subscribed trajectory indices */,
    int64_t *subs /* subscribed trajectory timesteps */,
    int64_t *lens /* subscribed trajectory lengths */,
    bool pad /* align to the tail */, int64_t ofst /* timestep sub offset */,
    int64_t span /* timestep sub len */,
    int idx_interval /* trajectory stride */,
    int sub_interval /* timestep stride of the column */,
    int sub_baseofst /* relative offset of a column */,
    int cpy_interval /* copied column slice stride */) {
  const int max_bytes_accessed_per_thread = 64;
  int maxload = span * sub_interval;
  int split = (maxload + max_bytes_accessed_per_thread - 1) /
              max_bytes_accessed_per_thread;

  int block_dim = MAX_THREADS_PER_BLOCK;
  int grid_dim = (num * split + block_dim - 1) / block_dim;
  auto stream = c10::cuda::getCurrentCUDAStream();

  fused_subcopy_collect<<<grid_dim, block_dim, 0, stream>>>(
      src, dst, num, idxs, subs, lens, pad, ofst, span, split,
      max_bytes_accessed_per_thread, idx_interval, sub_interval, sub_baseofst,
      cpy_interval);
}
} // namespace gear::cuda

void register_copy_kernerl_launcher(py::module &m) {
  m.def("copy", &gear::cuda::launch_fix_stride_copy_collect,
        /*TODO: add docstring*/ R"mydelimiter(
          
          
          )mydelimiter");
  m.def("vcopy", &gear::cuda::launch_vary_stride_copy_collect,
        /*TODO: add docstring*/ R"mydelimiter(
          
          
          )mydelimiter");

  m.def("copy_debug", [](torch::Tensor &src, torch::Tensor &dst,
                         torch::Tensor &src_offsets, torch::Tensor &dst_offsets,
                         torch::Tensor &strides, int64_t max_stride) {
    void *src_ptr = nullptr;
    convert_tensor_data_pointer(src, &src_ptr);

    auto src_mptr = gear::memory::MemoryPtr(src_ptr);
    gear::cuda::launch_vary_stride_copy_collect(
        src_mptr, dst, src_offsets, dst_offsets, strides, max_stride);
  });
}