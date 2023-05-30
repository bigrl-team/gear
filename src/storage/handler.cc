#include <sstream>

#include "common/cmp.h"
#include "debug.h"
#include "gear_errors.h"
#include "kernel_launchers.h"
#include "memory/memory.h"
#include "storage/handler.h"

using gear::comm::NcclCommunicator;
using gear::memory::Memory;
namespace gear::storage {

SubscribePattern::SubscribePattern(int offset, size_t length, PadOption option)
    : offset(offset), length(length), pad_option(option) {}

SubscribePattern::SubscribePattern(const SubscribePattern &other)
    : offset(other.offset), length(other.length), pad_option(other.pad_option) {
}

std::string
SubscribePattern::pad_option_to_string(const SubscribePattern::PadOption &opt) {
  switch (opt) {
  case SubscribePattern::PadOption::kInactive:
    return "PadOption::kInactive";
  case SubscribePattern::PadOption::kHead:
    return "PadOption::kHead";
  case SubscribePattern::PadOption::kTail:
    return "PadOption::kTail";
  default:
    return "PadOption: kUnknown";
  }
}

std::string SubscribePattern::to_string() {
  std::stringstream ss;
  ss << "<SubscribePattern: offset " << this->offset << ", length "
     << this->length << ", " << pad_option_to_string(this->pad_option) << ">";
  return ss.str();
}

CpuTrajectoryStorageHandler::CpuTrajectoryStorageHandler(
    const TrajectoryTable *table, size_t global_capacity, Range wregion,
    Range rregion)
    : global_capacity(global_capacity), writable_region(wregion),
      readable_region(rregion) {
  this->table_inst = table;
  this->patterns.resize(table->ncolumns());
}

void CpuTrajectoryStorageHandler::set(
    std::vector<size_t> cids, const std::vector<SubscribePattern> &patterns) {
  size_t count = 0;
  for (size_t cid : cids) {
    GEAR_DEBUG_PRINT("Setting subscribed column: cid :%zu, no: %zu.\n", cid,
                     count);
    this->patterns[cid] = patterns[count];
    count++;
  }
}

void inline _presub_addressing_bound_check(
    Int64Span target_trajectory_indices, Int64Span target_trajectory_timesteps,
    Int64Span target_trajectory_lengths, Int64Span offsets, Int64Span lengths) {
  GEAR_COND_EXCEPT(
      target_trajectory_indices.size == target_trajectory_timesteps.size &&
          target_trajectory_indices.size == target_trajectory_lengths.size,
      std::runtime_error,
      "Spans describing target propoties has inconsistent size!");

  GEAR_COND_EXCEPT(target_trajectory_indices.size <= offsets.size &&
                       target_trajectory_indices.size <= lengths.size,
                   std::runtime_error,
                   "Spans describing target propoties larger than "
                   "destination spans, potential overrun!");
}

int64_t CpuTrajectoryStorageHandler::sub(
    Int64Span target_trajectory_indices, Int64Span target_trajectory_timesteps,
    Int64Span target_trajectory_lengths, size_t target_column_id,
    Int64Span src_offsets, Int64Span dst_offsets, Int64Span lengths) {
  // _presub_addressing_bound_check(
  //     target_trajectory_indices, target_trajectory_timesteps,
  //     target_trajectory_lengths, src_offsets, lengths);

  const std::optional<SubscribePattern> &opt_pattern =
      this->patterns[target_column_id];
  if (!opt_pattern.has_value()) {
    throw std::runtime_error(
        "Subscribing a column without setting its SubscribePattern.");
  }
  const SubscribePattern &pattern = *opt_pattern;
  size_t accu_stride = this->table_inst->accu_strides[target_column_id];
  size_t column_ofst =
      accu_stride * this->table_inst->table_spec.trajectory_length;
  size_t stride = this->table_inst->column_strides[target_column_id];
  size_t num_op = target_trajectory_indices.size;
  int64_t expected_length = static_cast<int64_t>(pattern.length * stride);
  // #pragma unroll
  for (size_t i = 0; i < num_op; ++i) {
    int64_t tidx = target_trajectory_indices[i];
    int64_t ts = target_trajectory_timesteps[i];
    int64_t len = target_trajectory_lengths[i];

    int64_t base_ofst =
        tidx * this->table_inst->trajectory_stride + column_ofst;
    int64_t start = MAX(ts + static_cast<int64_t>(pattern.offset), 0);
    int64_t end = MIN(start + static_cast<int64_t>(pattern.length), len);

    start *= stride;
    end *= stride;
    int64_t copy_len = MAX(end - start, 0);

    int64_t dst_ofst = pattern.pad_option == SubscribePattern::PadOption::kHead
                           ? expected_length - copy_len
                           : 0;
    dst_ofst += expected_length * i;

    lengths[i] = copy_len;
    src_offsets[i] = base_ofst + start;
    dst_offsets[i] = dst_ofst;

    GEAR_DEBUG_PRINT(
        "xxxx>>>>> Trajectory stride %zu column offset %zu, base offset %zu, "
        "start %ld, index %ld.\n",
        this->table_inst->trajectory_stride, column_ofst, base_ofst, start,
        tidx);
  }

  return expected_length;
}

void CpuTrajectoryStorageHandler::fused_subcopy(Int64Span indices,
                                                Int64Span timesteps,
                                                Int64Span lengths,
                                                size_t column, void *dst) {
  const std::optional<SubscribePattern> &opt_pattern = this->patterns[column];
  if (!opt_pattern.has_value()) {
    throw std::runtime_error(
        "Subscribing a column without setting its SubscribePattern.");
  }
  const SubscribePattern &pat = *opt_pattern;
  int64_t column_ofst =
      static_cast<int64_t>(this->table_inst->accu_strides[column] *
                           this->table_inst->table_spec.trajectory_length);
  int64_t expected_length = static_cast<int64_t>(
      pat.length * this->table_inst->column_strides[column]);
  gear::cuda::launch_fused_subcopy_collect(
      this->table_inst->shm_blk.addr /* src */, dst, indices.size, indices.ptr,
      timesteps.ptr, lengths.ptr,
      pat.pad_option == SubscribePattern::PadOption::kHead, pat.offset,
      pat.length, this->table_inst->trajectory_stride,
      this->table_inst->column_strides[column], column_ofst, expected_length);
}

void CpuTrajectoryStorageHandler::connect(int rank, int worldsize, py::bytes id) {
  this->ncomm = NcclCommunicator(rank, worldsize, std::string(id));
}

void CpuTrajectoryStorageHandler::subsend(int peer, Int64Span indices,
                                          Int64Span timesteps,
                                          Int64Span lengths, size_t column) {

  const std::optional<SubscribePattern> &opt_pattern = this->patterns[column];
  if (!opt_pattern.has_value()) {
    throw std::runtime_error(
        "Subscribing a column without setting its SubscribePattern.");
  }
  const SubscribePattern &pat = *opt_pattern;

  int64_t column_ofst =
      static_cast<int64_t>(this->table_inst->accu_strides[column] *
                           this->table_inst->table_spec.trajectory_length);

  ncclDataType_t nccl_dtype = gear::comm::convert_dtype_to_nccl_type(
      this->table_inst->get_column_spec(column).dtype);

  ncclGroupStart();
  for (size_t i = 0; i < static_cast<size_t>(indices.size); ++i) {
    const void *src = const_cast<const void *>(reinterpret_cast<void *>(
        reinterpret_cast<char *>(this->table_inst->shm_blk.addr) +
        this->table_inst->trajectory_stride * indices[i] + column_ofst));
    int64_t start = MAX(0, timesteps[i] + pat.offset);
    int64_t end = MIN(lengths[i], static_cast<int64_t>(start + pat.length));
    int64_t count =
        MAX(end - start, 0) * this->table_inst->column_strides[column];
    printf("timestep %ld length %ld start %ld, end %ld, count %ld.\n",timesteps[i], lengths[i], start, end, count);
    ncclSend(src, count, nccl_dtype, peer, this->ncomm->nccl_comm,
             c10::cuda::getCurrentCUDAStream());
  }
  ncclGroupEnd();

}

void CpuTrajectoryStorageHandler::subrecv(int peer, void *dst,
                                          Int64Span indices,
                                          Int64Span timesteps,
                                          Int64Span lengths, size_t column) {
  const std::optional<SubscribePattern> &opt_pattern = this->patterns[column];
  if (!opt_pattern.has_value()) {
    throw std::runtime_error(
        "Subscribing a column without setting its SubscribePattern.");
  }
  const SubscribePattern &pat = *opt_pattern;

  ncclDataType_t nccl_dtype = gear::comm::convert_dtype_to_nccl_type(
      this->table_inst->get_column_spec(column).dtype);
  ncclGroupStart();
  for (size_t i = 0; i < static_cast<size_t>(indices.size); ++i) {
    void *wptr = reinterpret_cast<void *>(
        reinterpret_cast<char *>(dst) + i*
        pat.length * this->table_inst->column_strides[column]);
    int64_t start = MAX(0, timesteps[i] + pat.offset);
    int64_t end = MIN(lengths[i], static_cast<int64_t>(start + pat.length));
    int64_t count =
        MAX(end - start, 0) * this->table_inst->column_strides[column];
    ncclRecv(wptr, count, nccl_dtype, peer, this->ncomm->nccl_comm,
             c10::cuda::getCurrentCUDAStream());
  }
  ncclGroupEnd();
}

Uint8Span CpuTrajectoryStorageHandler::view(size_t index, size_t column_id) {
  size_t accu_stride = this->table_inst->accu_strides[column_id];
  size_t columnn_ofst =
      accu_stride * this->table_inst->table_spec.trajectory_length;
  size_t base_ofst = index * this->table_inst->trajectory_stride + columnn_ofst;
  size_t stride = this->table_inst->column_strides[column_id];
  size_t length = stride * this->table_inst->table_spec.trajectory_length;
  return Uint8Span(
      reinterpret_cast<void *>(
          reinterpret_cast<char *>(this->table_inst->shm_blk.addr) + base_ofst),
      length);
}

Uint8Span CpuTrajectoryStorageHandler::raw(size_t offset, size_t length) {
  return Uint8Span(
      reinterpret_cast<void *>(
          reinterpret_cast<char *>(this->table_inst->shm_blk.addr) + offset),
      length);
}

// void CpuTrajectoryStorageHandler::lcollect(int64_t *src_offsets, int64_t
// *src_lengths,
//                                      void *dst) {
//   GEAR_COND_EXCEPT(table_inst != nullptr, std::runtime_error,
//                    "Calling copy method with an uninstatiated
//                    TrajectoryTable!")
//   void *src_base = this->table_inst->shm_blk.addr;

// }
} // namespace gear::storage