#pragma once

#include <memory>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include "common/dtypes.h"
using gear::common::DataType;

namespace gear::storage {

struct ColumnSpec {
  std::vector<size_t> shape;
  DataType dtype;
  std::string name;

  ColumnSpec(std::vector<size_t> shape, DataType dtype, std::string name = "");

  size_t size() const;
};

struct TableSpec {
  size_t rank;
  size_t worldsize;
  size_t trajectory_length;
  size_t capacity;
  size_t num_columns;
  std::vector<ColumnSpec> column_specs;
  size_t nbytes;

  TableSpec(size_t rank, size_t worldsize, size_t trajectory_length,
            size_t capacity, size_t num_columns,
            std::vector<ColumnSpec> column_specs);

  ssize_t index(const std::string &name);

  size_t size();
};

} // namespace gear::storage