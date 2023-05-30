#pragma once

#include <pybind11/pybind11.h>

#include "common/dtypes.h"
#include "common/range.h"
#include "common/span.h"

namespace gear::common {
namespace py = pybind11;

void register_common_module(py::module &m);

void register_span(py::module &m);

void register_dtypes(py::module &m);

void register_range(py::module &m);

} // namespace gear::common