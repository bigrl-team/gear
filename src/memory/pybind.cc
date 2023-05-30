#include "memory/pybind.h"

namespace gear::memory {
void register_module(py::module &m) {
  register_memory(m);
  register_memory_ptr(m);
  register_shared_memory(m);
  register_cuda_memory(m);
  register_cuda_uvm(m);
  register_memory_ref(m);
}

void register_memory_ptr(py::module &m) {
  py::class_<MemoryPtr, std::shared_ptr<MemoryPtr>>(m, "MemoryPtr")
      .def(py::init<int64_t>())
      .def("__repr__", &MemoryPtr::to_string);
}

void register_memory(py::module &m) {
  py::class_<Memory, std::shared_ptr<Memory>> memory_class(m, "Memory");

  py::enum_<Memory::MemoryType>(memory_class, "MemoryType")
      .value("cpu", Memory::MemoryType::kCpu)
      .value("uvm", Memory::MemoryType::kUvm)
      .value("shared", Memory::MemoryType::kShared)
      .value("cuda", Memory::MemoryType::kCuda)
      .export_values();

  memory_class
      .def(py::init([](size_t size) {
             return std::make_shared<Memory>(size, Memory::MemoryType::kCpu);
           }),
           py::arg("size"))
      .def("__repr__", &Memory::to_string)
      .def_property_readonly(
          "addr",
          [](const Memory &m) { return std::make_shared<MemoryPtr>(m.addr); })
      .def_readonly("size", &Memory::size)
      .def_readonly("type", &Memory::mtype)
      .def("alloc", &Memory::alloc)
      .def("free", &Memory::free);
}

void register_shared_memory(py::module &m) {
  py::class_<SharedMemory, std::shared_ptr<SharedMemory>, Memory>(
      m, "SharedMemory")
      .def(py::init<key_t, size_t, bool>(), py::arg("key"), py::arg("size"),
           py::arg("create"))
      .def("alloc", &SharedMemory::alloc)
      .def("free", &SharedMemory::free)
      .def_property_readonly("addr",
                             [](const SharedMemory &m) {
                               return std::make_shared<MemoryPtr>(m.addr);
                             })
      .def_readonly("size", &SharedMemory::size)
      .def_readonly("type", &SharedMemory::mtype)
      .def_readonly("key", &SharedMemory::key)
      .def_readonly("create", &SharedMemory::create)
      .def_readonly("shmid", &SharedMemory::shmid);
}

void register_cuda_memory(py::module &m) {
  py::class_<CudaMemory, std::shared_ptr<CudaMemory>, Memory>(m, "CudaMemory")
      .def(py::init<size_t>(), py::arg("size"))
      .def("alloc", &CudaMemory::alloc)
      .def("free", &CudaMemory::free)
      .def_property_readonly("addr",
                             [](const CudaMemory &m) {
                               return std::make_shared<MemoryPtr>(m.addr);
                             })
      .def_readonly("size", &SharedMemory::size)
      .def_readonly("type", &SharedMemory::mtype);
}

void register_cuda_uvm(py::module &m) {
  py::class_<CudaUVMemory, std::shared_ptr<CudaUVMemory>, Memory>(
      m, "CudaUVMemory")
      .def(py::init<size_t>(), py::arg("size"))
      .def("alloc", &CudaUVMemory::alloc)
      .def("free", &CudaUVMemory::free);
}

void register_memory_ref(py::module &m) {

#define MACRO_REGISTER_MEMREF(MEMREF_TYPE, MEMREF_NAME)                        \
  py::class_<MEMREF_TYPE, std::shared_ptr<MEMREF_TYPE>>(m, #MEMREF_NAME)       \
      .def_readonly("raw", &MEMREF_TYPE::raw)                                  \
      .def_readonly("length", &MEMREF_TYPE::length)                            \
      .def_readonly("offset", &MEMREF_TYPE::offset);

  MACRO_REGISTER_MEMREF(CpuMemoryRef, "CpuMemoryRef");
  MACRO_REGISTER_MEMREF(SharedMemoryRef, "SharedMemoryRef");
  MACRO_REGISTER_MEMREF(UvmMemoryRef, "UvmMemoryRef");

#undef MACRO_REGISTER_MEMREF
}

} // namespace gear::memory