#pragma once

#include <string>

#include "common/range.h"
#include "common/span.h"
#include "nccl_comm.h"
#include "storage/specs.h"
#include "storage/table.h"

using gear::common::Int64Span;
using gear::common::Range;
using gear::common::Uint8Span;

using gear::comm::NcclCommunicator;
namespace gear::storage {
struct SubscribePattern {
  int offset = 0;
  size_t length = 1;

  enum class PadOption { kInactive, kHead, kTail, kUnknown };
  static std::string pad_option_to_string(const PadOption &opt);

  PadOption pad_option;

  SubscribePattern(int offset, size_t length, PadOption option);

  SubscribePattern(const SubscribePattern &other);

  std::string to_string();
};

class TrajectoryStorageHandler {

public:
  virtual void set(std::vector<size_t> cids,
                   const std::vector<SubscribePattern> &patterns) = 0;

  virtual int64_t sub(Int64Span target_trajectory_indices,
                      Int64Span target_trajectory_timesteps,
                      Int64Span target_trajectory_lengths,
                      size_t target_column_id, Int64Span src_offsets,
                      Int64Span dst_offsets, Int64Span lengths) = 0;

protected:
  const TrajectoryTable *table_inst = nullptr;
  std::vector<std::optional<SubscribePattern>> patterns;
};

class CpuTrajectoryStorageHandler : public virtual TrajectoryStorageHandler {

public:
  CpuTrajectoryStorageHandler(const TrajectoryTable *table, Range wregion,
                              Range rregion);

  void set(std::vector<size_t> cids,
           const std::vector<SubscribePattern> &patterns) override;

  int64_t sub(Int64Span target_trajectory_indices,
              Int64Span target_trajectory_timesteps,
              Int64Span target_trajectory_lengths, size_t target_column_id,
              Int64Span src_offsets, Int64Span dst_offsets,
              Int64Span lengths) override;

  void fused_subcopy(Int64Span indices, Int64Span timesteps, Int64Span lengths,
                     size_t column, void *dst);

  void connect(int rank, int worldsize, py::bytes id);

  void subrecv(int peer, void *dst, Int64Span indices, Int64Span timesteps,
               Int64Span lengths, size_t column);

  void subsend(int peer, Int64Span indices, Int64Span timesteps,
               Int64Span lengths, size_t column);

  Uint8Span view(size_t index, size_t column_id);

  Uint8Span raw(size_t offset, size_t length);

private:
  size_t global_capacity;
  Range writable_region;
  Range readable_region;

  std::optional<NcclCommunicator> ncomm;
};

// class CudaTrajectoryStorageHandler : public TrajectoryStorageHandler {
// public:
//   CudaTrajectoryStorageHandler(const TrajectoryTable* table);

//   void set(std::vector<std::shared_ptr<SubscribePattern>> patterns) override;

//   void sub(Int64Span target_trajectory_indices,
//            Int64Span target_trajectory_timesteps,
//            Int64Span target_trajectory_lengths, size_t target_column_id,
//            Int64Span offsets, Int64Span lengths) override;

//   void fused_subcopy(Int64Span indices, Int64Span timesteps, Int64Span
//   lengths, size_t column)
// };

} // namespace gear::storage