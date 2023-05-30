#pragma once

#include <cuda.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "common/span.h"
#include "memory/memory_ptr.h"

namespace gear::cuda {
void init(int device_id);

void launch_sum(torch::Tensor val_tensor, torch::Tensor sum_tensor, int stride);

void launch_prefix_sum(torch::Tensor input, torch::Tensor output);

void launch_fix_stride_copy_collect(gear::memory::MemoryPtr src,
                                    torch::Tensor dst,
                                    torch::Tensor src_offsets,
                                    torch::Tensor dst_offsets, int64_t stride);

void launch_vary_stride_copy_collect(gear::memory::MemoryPtr &ssrc_mptr,
                                     torch::Tensor &dst,
                                     torch::Tensor &src_offsets,
                                     torch::Tensor &dst_offsets,
                                     torch::Tensor &strides,
                                     int64_t max_stride);

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
    int cpy_interval /* copied column slice stride */);
} // namespace gear::cuda

void register_cuda_init(pybind11::module &m);

void register_sum_kernel_launcher(pybind11::module &m);

void register_prefix_sum_kernel_launcher(pybind11::module &m);

// void launch_copy_collect(pybind11::array_t<double> &src, torch::Tensor dst,
//                          int offset, int stride);

void register_copy_kernerl_launcher(pybind11::module &m);
