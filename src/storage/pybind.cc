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
  py::class_<ColumnSpec, std::shared_ptr<ColumnSpec>>(
      m, "ColumnSpec",
      R"mydelimiter(Describing all the necessary attributes/info about a table column, 
      which includes the column entry shape, data type and possibly the name. 

      :type shape: std::vector<size_t> 
      :param shape:
        Describe the entry shape in int array.

      :type dtype: :py:class:`libgear.DataType`.
      :param dtype:
        Data type of the column. Used to calculate memory offset and interpretation.

      :type name: std::string
      :param name:
        Column name. Referenced when indexing columns with name.)mydelimiter")
      .def(py::init<std::vector<size_t>, DataType, std::string>())
      .def_readwrite("dtype", &ColumnSpec::dtype, "Column data type.")
      .def_readwrite("shape", &ColumnSpec::shape,
                     "Column entry shape as int array")
      .def_readwrite("name", &ColumnSpec::name, "Column name as a string.")
      .def("size", &ColumnSpec::size,
           "Get the entry size of the column in bytes.")
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

  py::class_<TableSpec, std::shared_ptr<TableSpec>>(
      m, "TableSpec",
      R"mydelimiter(Attributes and info of table, together with serveral utility methods related.
      
      :type rank: size_t
      :param rank: 
        The rank of the current table among all the tables, which is referenced in index translation.
      
      :type worldsize: size_t
      :param worldsize: 
        Total number of tables, referenced when making inference on global capacity.


      :type trajectory_length: size_t
      :param trajectory_length:
        Maximum sequence length of the table.

      :type capacity: size_t
      :param capacity:
        Capacity of local table.
        
      :type num_columns: size_t 
      :param num_columns:
        Number of columns in the table.
      )mydelimiter")
      .def(py::init<size_t, size_t, size_t, size_t, size_t,
                    std::vector<ColumnSpec>>())
      .def_readwrite("rank", &TableSpec::rank)
      .def_readwrite("worldsize", &TableSpec::worldsize)
      .def_readwrite("trajectory_length", &TableSpec::trajectory_length)
      .def_readwrite("capacity", &TableSpec::capacity)
      .def_readonly("num_columns", &TableSpec::num_columns)
      .def_readonly("column_specs", &TableSpec::column_specs)
      .def(
          "index", &TableSpec::index,
          R"mydelimiter(Return the id of the first column whose name is equal to the given request. 
          
          :param name:
            Desired column name.

          :rtype: ssize_t 
          :return:
            -1 if no match column, else the column id.

          )mydelimiter")
      .def("size", &TableSpec::size,
           R"mydelimiter(
            Return the estimated size of the table storage space in bytes.
          )mydelimiter")
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
      subscribe_pattern_class(
          m, "SubscribePattern",
          R"mydelimiter(Column-wise data request patterns that support time-ranged subscribing, referenced when fetching data by a storage handler.

          :type offset: int
          :param offset:
            The offset with respect to the current length of the trajectory to begin subscribing, usually negative. That is, for a trajectory that has `k` steps(whether terminated or non-terminated), if an offset of `s` is set for subscribe, then the fetched data starts with the step `max(0, k + s)`.

          :type length: int
          :param length:
            The desired length of the pulled subtrajectory. That is, That is, for a trajectory that has `k` steps(whether terminated or non-terminated), if an offset of `s` is set for subscribe and also a length of "l", then the fetched subtrajectory starts with the step `max(0, k + s)` and ends with the step `min(k, k + s + l)`.

          :type option: :py:class:`libgear.storage.SubscribePattern.PadOption`.
          :param option:
            Dertermine the padding position. Default value is :py:attr:`libgear.storage.SubscribePattern.PadOption.tail` if not specified.  When using the storage handler's API, note that range clipping may be applied to the sub-trajectory to prevent oversubscription. As a result, the returned sequence could be shorter than the set length. If the subscribed data returned to the user is shorter, zero padding is added either at the beginning or the end of the sub-trajectory.


          .. code-block:: python

            # usage
            p1 = SubscribePattern(-100, 100, SubscribePattern.PadOption.head)
            p2 = SubscribePattern(-100, 100, SubscribePattern.PadOption.head)
            p3 = SubscribePattern(p2)
            
          )mydelimiter");

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

  py::enum_<SubscribePattern::PadOption>(subscribe_pattern_class, "PadOption",
                                         R"mydelimiter(
          inactive:
            Default value for an empty construction and should not be taken other than testing. Potential errors are expected if an `inactive` value is set in practice.
          
          head:
            Apply zero padding to the head/begining of the trajectory.

          tail:
            Apply zero padding to the tail/ending of the trajectory.

          )mydelimiter")
      .value("inactive", SubscribePattern::PadOption::kInactive)
      .value("head", SubscribePattern::PadOption::kHead)
      .value("tail", SubscribePattern::PadOption::kTail)
      .export_values();

  py::class_<TrajectoryStorageHandler,
             std::shared_ptr<TrajectoryStorageHandler>>(
      m, "TrajectoryStorageHandler", R"mydelimiter(
          The abstract class the for the `StorageHandler` class family, whose main role is the translation from high-level indexing interface(i.e. trajectory id & column ids & steps) to the low-level memory offset indexing.)mydelimiter")
      .def("set", &TrajectoryStorageHandler::set)
      .def("sub", &TrajectoryStorageHandler::sub);

  py::class_<CpuTrajectoryStorageHandler,
             std::shared_ptr<CpuTrajectoryStorageHandler>>(
      m, "CpuTrajectoryStorageHandler", R"mydelimiter(
          An implementation of the 'StorageHandler' class whose index translation computation mainly locates on host CPU. If the index translation computation are proved to be expensive, a CUDA version of it may be necessary.
          
          Usage:
            .. seealso::
            
             :py:func:`libgear.storage.get_cpu_handler`.
          
          )mydelimiter")
      .def("set", &CpuTrajectoryStorageHandler::set,
           /*TODO: add docstring*/ R"mydelimiter(
        
          
          )mydelimiter")
      .def("sub", &CpuTrajectoryStorageHandler::sub,
           /*TODO: add docstring*/ R"mydelimiter(
          
          
          )mydelimiter")
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
      .def(
          "subcopy",
          [](CpuTrajectoryStorageHandler &self, torch::Tensor idxs,
             torch::Tensor tss, torch::Tensor lens, size_t column,
             torch::Tensor dst) {
            self.fused_subcopy(Int64Span(idxs), Int64Span(tss), Int64Span(lens),
                               column, get_raw_ptr(dst));
          },
          /*TODO: add docstring*/ R"mydelimiter(
          
          
          )mydelimiter")
      .def("connect", &CpuTrajectoryStorageHandler::connect,
           /*TODO: add docstring*/ R"mydelimiter(
          
          
          )mydelimiter")
      .def("subsend", &CpuTrajectoryStorageHandler::subsend,
           /*TODO: add docstring*/ R"mydelimiter(
          
          
          )mydelimiter")
      .def(
          "subrecv",
          [](CpuTrajectoryStorageHandler &self, int peer, torch::Tensor dst,
             Int64Span indices, Int64Span timesteps, Int64Span lengths,
             size_t column) {
            void *ptr = nullptr;
            convert_tensor_data_pointer(dst, &ptr);
            self.subrecv(peer, ptr, indices, timesteps, lengths, column);
          },
          /*TODO: add docstring*/ R"mydelimiter(
           /* your docstrings here */ 
           )mydelimiter")
      .def("view", &CpuTrajectoryStorageHandler::view,
           /*TODO: add docstring*/ R"mydelimiter(
      /* your docstrings here */ 
      )mydelimiter")
      .def("raw", &CpuTrajectoryStorageHandler::raw,
           /*TODO: add docstring*/ R"mydelimiter(
      /* your docstrings here */ 
      )mydelimiter");

  m.def(
      "get_cpu_handler",
      [](const TrajectoryTable &table, Range wregion, Range rregion) {
        return std::make_shared<CpuTrajectoryStorageHandler>(&table, wregion,
                                                             rregion);
      },
      R"mydelimiter(
          The factory function of :py:class:`CpuTrajectoryStorageHandler`.

          :type table: :py:class:`libgear.storage.TrajectoryTable`.
          :param table:
            The subscribed table.
          
          :type wregion: libgear.Range
          :param wregion:
            Writable index region.

          :type rregion: libgear.Range
          :param rregion:
            Readable index region

          
          )mydelimiter");
}
void register_table(py::module &m) {
  py::class_<TrajectoryTable, std::shared_ptr<TrajectoryTable>>(
      m, "TrajectoryTable", /*TODO: add docstring*/ R"mydelimiter(
      /* your docstrings here */ 
      )mydelimiter")
      .def(py::init<const TableSpec &, key_t, bool>())
      .def_property_readonly("ncolumns", &TrajectoryTable::ncolumns,
                             /*TODO: add docstring*/ R"mydelimiter(
      /* your docstrings here */ 
      )mydelimiter")
      .def("get_table_spec", &TrajectoryTable::get_table_spec,
           /*TODO: add docstring*/ R"mydelimiter(
      /* your docstrings here */ 
      )mydelimiter")
      .def("get_column_spec", &TrajectoryTable::get_column_spec,
           /*TODO: add docstring*/ R"mydelimiter(
      /* your docstrings here */ 
      )mydelimiter")
      .def(
          "get_address",
          [](TrajectoryTable &self) {
            return gear::memory::MemoryPtr(self.get_address());
          },
          /*TODO: add docstring*/ R"mydelimiter(
           /* your docstrings here */ 
           )mydelimiter")
      .def("connect", &TrajectoryTable::connect,
           /*TODO: add docstring*/ R"mydelimiter(
      /* your docstrings here */ 
      )mydelimiter")
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