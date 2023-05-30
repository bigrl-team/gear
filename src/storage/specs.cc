#include "common/dtypes.h"
#include "debug.h"
#include "storage/specs.h"

namespace gear::storage {
ColumnSpec::ColumnSpec(std::vector<size_t> shape, DataType dtype,
                       std::string name)
    : shape(shape), dtype(dtype), name(name) {}

size_t ColumnSpec::size() const {
  size_t elem_size = gear::common::sizeof_dtype(this->dtype);
  return std::accumulate(std::begin(this->shape), std::end(this->shape),
                         elem_size, std::multiplies<size_t>());
}

TableSpec::TableSpec(size_t rank, size_t worldsize, size_t trajectory_length,
                     size_t capacity, size_t num_columns,
                     std::vector<ColumnSpec> column_specs)
    : rank(rank), worldsize(worldsize), trajectory_length(trajectory_length),
      capacity(capacity), num_columns(num_columns), column_specs(column_specs) {
  GEAR_ASSERT(column_specs.size() == num_columns,
              "column specs size mismatch with declared number of columns: "
              "spec num %ld, expected %ld",
              column_specs.size(), num_columns);
  size_t total_size = 0;
  for (const ColumnSpec &cspec : this->column_specs) {
    total_size += cspec.size();
  }
  this->nbytes = total_size * this->trajectory_length * this->capacity;
}

ssize_t TableSpec::index(const std::string &name) {
  ssize_t cid = 0;
  for (const ColumnSpec &cspec : this->column_specs) {
    if (cspec.name == name)
      return cid;
    else
      cid++;
  }
  return -1;
}

size_t TableSpec::size() { return this->nbytes; }

} // namespace gear::storage