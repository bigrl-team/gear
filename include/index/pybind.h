#pragma once

#include <pybind11/pybind11.h>

#include "index/client.h"
#include "index/request.h"
#include "index/server.h"
#include "index/set.h"

namespace gear::index {
namespace py = pybind11;

void register_module(py::module &m);

void register_client(py::module &m);

void register_request(py::module &m);

void register_server(py::module &m);

void register_indexset(py::module &m);
} // namespace gear::index