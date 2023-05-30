#include <pybind11/numpy.h>
#include <torch/extension.h>

#include "common/dtypes.h"
#include "common/span.h"
#include "index/request.h"
#include "memory/memory_ptr.h"
#include "storage/pybind.h"

using gear::common::get_raw_ptr;
using gear::common::Int64Span;
namespace gear::storage {

void register_module(py::module &m) {
  register_specs(m);
  register_handler(m);
  register_table(m);
}

void register_specs(py::module &m) {
  using gear::common::DataType;
  py::class_<ColumnSpec, std::shared_ptr<ColumnSpec>>(m, "ColumnSpec")
      .def(py::init<std::vector<size_t>, DataType, std::string>())
      .def_readwrite("dtype", &ColumnSpec::dtype)
      .def_readwrite("shape", &ColumnSpec::shape)
      .def_readwrite("name", &ColumnSpec::name)
      .def("size", &ColumnSpec::size)
      .def(py::pickle(
          // __getstate__ method
          [](const ColumnSpec &spec) {
            return py::make_tuple(spec.shape, spec.dtype, spec.name);
          }, // __setstate__ method
          [](py::tuple t) {
            if (t.size() != 3)
              throw std::runtime_error(
                  "Invalid state for ColumnSpec unpickling!");
            return std::make_shared<ColumnSpec>(
                t[0].cast<std::vector<size_t>>(), t[1].cast<DataType>(),
                t[2].cast<std::string>());
          }));

  py::class_<TableSpec, std::shared_ptr<TableSpec>>(m, "TableSpec")
      .def(py::init<size_t, size_t, size_t, size_t, size_t,
                    std::vector<ColumnSpec>>())
      .def_readwrite("rank", &TableSpec::rank)
      .def_readwrite("worldsize", &TableSpec::worldsize)
      .def_readwrite("trajectory_length", &TableSpec::trajectory_length)
      .def_readwrite("capacity", &TableSpec::capacity)
      .def_readonly("num_columns", &TableSpec::num_columns)
      .def_readonly("column_specs", &TableSpec::column_specs)
      .def("index", &TableSpec::index)
      .def("size", &TableSpec::size)
      .def(py::pickle(
          // __getstate__ method
          [](const TableSpec &spec) {
            return py::make_tuple(spec.rank, spec.worldsize,
                                  spec.trajectory_length, spec.capacity,
                                  spec.num_columns, spec.column_specs);
          }, // __setstate__ method
          [](py::tuple t) {
            if (t.size() != 6) {
              throw std::runtime_error(
                  "Invalid state for TableSpec unpickling!");
            }
            return std::make_shared<TableSpec>(
                t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<size_t>(),
                t[3].cast<size_t>(), t[4].cast<size_t>(),
                t[5].cast<std::vector<ColumnSpec>>());
          }));
}

void register_handler(py::module &m) {
  py::class_<SubscribePattern, std::shared_ptr<SubscribePattern>>
      subscribe_pattern_class(m, "SubscribePattern");

  subscribe_pattern_class
      .def(py::init<int, size_t, SubscribePattern::PadOption>())
      .def(py::init([](int offset, size_t length) {
        return SubscribePattern{offset, length,
                                SubscribePattern::PadOption::kTail};
      }))
      .def_readonly("offset", &SubscribePattern::offset)
      .def_readonly("length", &SubscribePattern::length)
      .def("__repr__", &SubscribePattern::to_string)
      .def(py::pickle(
          // __getstate__ method
          [](const SubscribePattern &p) {
            return py::make_tuple(p.offset, p.length, p.pad_option);
          }, // __setstate__ method
          [](py::tuple t) {
            if (t.size() != 3) {
              throw std::runtime_error(
                  "Invalid state for SubscribePattern unpickling");
            }

            return std::make_shared<SubscribePattern>(
                t[0].cast<int>(), t[1].cast<size_t>(),
                t[2].cast<SubscribePattern::PadOption>());
          }));

  py::enum_<SubscribePattern::PadOption>(subscribe_pattern_class, "PadOption")
      .value("inactive", SubscribePattern::PadOption::kInactive)
      .value("head", SubscribePattern::PadOption::kHead)
      .value("tail", SubscribePattern::PadOption::kTail)
      .export_values();

  py::class_<TrajectoryStorageHandler,
             std::shared_ptr<TrajectoryStorageHandler>>(
      m, "TrajectoryStorageHandler")
      .def("set", &TrajectoryStorageHandler::set)
      .def("sub", &TrajectoryStorageHandler::sub);

  py::class_<CpuTrajectoryStorageHandler,
             std::shared_ptr<CpuTrajectoryStorageHandler>>(
      m, "CpuTrajectoryStorageHandler")
      .def("set", &CpuTrajectoryStorageHandler::set)
      .def("sub", &CpuTrajectoryStorageHandler::sub)
      .def("sub_",
           [](CpuTrajectoryStorageHandler &self, torch::Tensor &indices,
              torch::Tensor &timesteps, torch::Tensor &lengths,
              size_t column_id, torch::Tensor &src_offsets,
              torch::Tensor &dst_offsets, torch::Tensor &copy_lengths) {
             return self.sub(Int64Span(indices), Int64Span(timesteps),
                             Int64Span(lengths), column_id,
                             Int64Span(src_offsets), Int64Span(dst_offsets),
                             Int64Span(copy_lengths));
           })
      .def("sub_infer_cache",
           [](CpuTrajectoryStorageHandler &self,
              gear::index::CachedRequest &cache, size_t batch_size,
              size_t column_id, torch::Tensor &src_offsets,
              torch::Tensor &dst_offsets, torch::Tensor &lengths) {
             gear::index::InferenceRequestArray &arr = cache.iarr;
             self.sub(Int64Span(arr.idxs, arr.count),
                      Int64Span(arr.tss, arr.count),
                      Int64Span(arr.tss, arr.count), column_id,
                      Int64Span(src_offsets), Int64Span(dst_offsets),
                      Int64Span(lengths));
           })
      .def("subcopy",
           [](CpuTrajectoryStorageHandler &self, torch::Tensor idxs,
              torch::Tensor tss, torch::Tensor lens, size_t column,
              torch::Tensor dst) {
             self.fused_subcopy(Int64Span(idxs), Int64Span(tss),
                                Int64Span(lens), column, get_raw_ptr(dst));
           })
      .def("connect", &CpuTrajectoryStorageHandler::connect)
      .def("subsend", &CpuTrajectoryStorageHandler::subsend)
      .def("subrecv", [](CpuTrajectoryStorageHandler& self, int peer, torch::Tensor dst, Int64Span indices, Int64Span timesteps , Int64Span lengths, size_t column){
        void* ptr = nullptr;
        convert_tensor_data_pointer(dst, &ptr);
        self.subrecv(peer, ptr, indices, timesteps, lengths, column);
      })
      .def("view", &CpuTrajectoryStorageHandler::view)
      .def("raw", &CpuTrajectoryStorageHandler::raw);

  m.def("get_cpu_handler",
        [](const TrajectoryTable &table, size_t global_capacity, Range wregion,
           Range rregion) {
          return std::make_shared<CpuTrajectoryStorageHandler>(
              &table, global_capacity, wregion, rregion);
        }

  );
}
void register_table(py::module &m) {
  py::class_<TrajectoryTable, std::shared_ptr<TrajectoryTable>>(
      m, "TrajectoryTable")
      .def(py::init<const TableSpec &, key_t, bool>())
      .def_property_readonly("ncolumns", &TrajectoryTable::ncolumns)
      .def("get_table_spec", &TrajectoryTable::get_table_spec)
      .def("get_column_spec", &TrajectoryTable::get_column_spec)
      .def("get_address",
           [](TrajectoryTable &self) {
             return gear::memory::MemoryPtr(self.get_address());
           })
      .def("connect", &TrajectoryTable::connect)
      .def(py::pickle(
          // __getstate__ method
          [](const TrajectoryTable &t) {
            return py::make_tuple(
                t.get_table_spec(), t.get_key(), t.get_size(), t.is_create(),
                py::array_t<uint8_t>(
                    {static_cast<int64_t>(t.get_size())}, {1},
                    reinterpret_cast<uint8_t *>(t.get_address())));
          },
          // __setstate__ method
          [](py::tuple t) {
            if (t.size() != 5)
              throw std::runtime_error("Invalid state!");

            key_t key = t[1].cast<key_t>();
            int size = t[2].cast<int>();
            bool create = t[3].cast<bool>();

            auto table = std::make_shared<TrajectoryTable>(
                t[0].cast<TableSpec>(), key, create);
            table->connect();
            memcpy(table->get_address(),
                   reinterpret_cast<const void *>(
                       t[4].cast<py::array_t<uint8_t>>().request().ptr),
                   static_cast<size_t>(size));
            return table;
          }));
}
} // namespace gear::storage