#pragma once

#include <pybind11/pybind11.h>

#include "memory/cuda_memory.h"
#include "memory/cuda_uvm.h"
#include "memory/memory.h"
#include "memory/memory_ptr.h"
#include "memory/memory_ref.h"
#include "memory/shared_memory.h"

namespace gear::memory {
namespace py = pybind11;

void register_module(py::module &m);

void register_memory_ptr(py::module &m);

void register_memory(py::module &m);

void register_shared_memory(py::module &m);

void register_cuda_memory(py::module &m);

void register_cuda_uvm(py::module &m);

void register_memory_ref(py::module &m);
} // namespace gear::memory