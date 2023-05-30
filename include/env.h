#pragma once

#include <pybind11/pybind11.h>

namespace gear::env {
namespace py = pybind11;

const int PRINT_ALIGNMENT = 30;

void print_compile_options();

void print_environ();

void register_env_detect_functions(py::module &m);
} // namespace gear::env
