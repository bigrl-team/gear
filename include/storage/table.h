#pragma once

#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

#include "memory/shared_memory.h"
#include "storage/specs.h"

namespace gear::storage {

class TrajectoryStorageHandler;
class CpuTrajectoryStorageHandler;
class CudaTrajectoryStorageHandler;

class TrajectoryTable {
  friend class TrajectoryStorageHandler;
  friend class CpuTrajectoryStorageHandler;
  friend class CudaTrajectoryStorageHandler;

public:
  // TODO: current impl. relies on the consensus of table underlying memory
  // layout both on the reader(server)'s side and clients' side, which may be an
  // overkill for the possible applications and potential disagreements may
  // cause memory corruption. Implement it in the way that server serialize the
  // table meta onto the shared memory while the clients retrieve meta,
  // deserialize it and reinterpret the memory layout.
  TrajectoryTable(const TableSpec &spec, key_t key, bool create);

  TrajectoryTable(const TrajectoryTable &table) = delete;

  TrajectoryTable &operator=(const TrajectoryTable &table) = delete;

  TrajectoryTable &operator=(TrajectoryTable &&table) = delete;

  ssize_t connect();

  size_t ncolumns() const;

  const TableSpec &get_table_spec() const;

  const ColumnSpec &get_column_spec(size_t column_id) const;

  void *get_address() const;

  size_t get_size() const;

  key_t get_key() const;

  bool is_create() const;

private:
  TableSpec table_spec;

  gear::memory::SharedMemory shm_blk;

  size_t trajectory_stride;
  std::vector<size_t> column_strides;
  std::vector<size_t> accu_strides;
};

} // namespace gear::storage