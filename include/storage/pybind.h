#pragma once

#include <pybind11/pybind11.h>

#include "storage/handler.h"
#include "storage/specs.h"
#include "storage/table.h"

namespace gear::storage {
namespace py = ::pybind11;

void register_module(py::module &m);

void register_specs(py::module &m);

void register_handler(py::module &m);

void register_table(py::module &m);

} // namespace gear::storage