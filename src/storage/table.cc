#include <numeric>

#include "storage/table.h"

using gear::memory::SharedMemory;

namespace gear::storage {
TrajectoryTable::TrajectoryTable(const TableSpec &spec, key_t key, bool create)
    : table_spec(spec) {
  this->column_strides.reserve(this->table_spec.num_columns);
  for (const ColumnSpec &cspec : this->table_spec.column_specs) {
    this->column_strides.push_back(cspec.size());
  }
  this->trajectory_stride = std::accumulate(std::begin(this->column_strides),
                                            std::end(this->column_strides), 0) *
                            this->table_spec.trajectory_length;

  size_t table_size = this->trajectory_stride * this->table_spec.capacity;
  this->shm_blk = std::move(SharedMemory(key, table_size, create));

  this->accu_strides.resize(this->column_strides.size() + 1);
  std::partial_sum(std::begin(this->column_strides),
                   std::end(this->column_strides),
                   std::begin(this->accu_strides) + 1);
}

ssize_t TrajectoryTable::connect() { return this->shm_blk.alloc(); }

size_t TrajectoryTable::ncolumns() const{ return this->table_spec.num_columns; }

const TableSpec &TrajectoryTable::get_table_spec() const { return this->table_spec; }

const ColumnSpec &TrajectoryTable::get_column_spec(size_t column_id) const {
  return this->table_spec.column_specs[column_id];
}

void *TrajectoryTable::get_address() const { return this->shm_blk.addr; }

size_t TrajectoryTable::get_size() const { return this->shm_blk.size; }

key_t TrajectoryTable::get_key() const { return this->shm_blk.key; }

bool TrajectoryTable::is_create() const { return this->shm_blk.create; }
} // namespace gear::storage