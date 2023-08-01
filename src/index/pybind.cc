#include <pybind11/numpy.h>

#include "common/span.h"
#include "common/tensor.h"
#include "index/pybind.h"
#include "memory/memory_ptr.h"

using gear::common::Float32Span;
using gear::common::Int64Span;
using gear::common::Uint8Span;

namespace py = pybind11;

namespace gear::index {
py::tuple indexset_state_serialization_helper(const IndexsetState &state) {
  GEAR_DEBUG_PRINT(
      "begin serialization helper, total mem length %zu %zu %zu.\n",
      calc_state_mem_size(state.local_capacity), state.mem.length,
      state.mem.raw->size);
  GEAR_DEBUG_PRINT("first timestep %ld.\n",
                   reinterpret_cast<int64_t *>(state.mem.raw->addr)[0]);
  return py::make_tuple(
      (int64_t)state.index_offset, (int64_t)state.local_capacity,
      (int64_t)state.global_capacity,
      py::array_t<uint8_t>(
          {calc_state_mem_size(state.local_capacity)}, {1},
          reinterpret_cast<const uint8_t *>(state.mem.raw->addr)));
}

std::shared_ptr<IndexsetState>
indexset_state_deserialization_helper(py::tuple t) {
  if (t.size() != 4) {
    throw std::runtime_error("Invalid state for unpickling IndexsetState");
  }
  size_t index_offset = (size_t)t[0].cast<int64_t>();
  size_t local_capacity = (size_t)t[1].cast<int64_t>();
  size_t global_capacity = (size_t)t[2].cast<int64_t>();
  GEAR_DEBUG_PRINT("Deserialized IndexsetState description: <IndexOffset: %zu, "
                   "LocalCapacity: %zu, GlobalCapacity: %zu>.\n",
                   index_offset, local_capacity, global_capacity);
  py::array_t<uint8_t> mem = t[3].cast<py::array_t<uint8_t>>();
  const int64_t *timesteps =
      reinterpret_cast<const int64_t *>(mem.request().ptr);
  const float *weights = reinterpret_cast<const float *>(
      reinterpret_cast<const char *>(mem.request().ptr) +
      local_capacity * sizeof(int64_t));

  return std::make_shared<IndexsetState>(global_capacity, local_capacity,
                                         index_offset, timesteps, weights);
}

py::tuple
index_buffer_state_serialization_helper(const IndexBufferState &state) {
  return py::make_tuple(
      state.capacity, state.head, state.tail, state.length,
      py::array_t<uint8_t>(
          {sizeof(int64_t) * state.capacity}, {1},
          reinterpret_cast<const uint8_t *>(state.data.raw->addr)));
}

std::shared_ptr<IndexBufferState>
index_buffer_state_deserialization_helper(py::tuple t) {
  if (t.size() != 5) {
    throw std::runtime_error("Invalid state for unpickling IndexBufferState");
  }
  return std::make_shared<IndexBufferState>(
      t[1].cast<size_t>(), t[2].cast<size_t>(), t[3].cast<size_t>(),
      t[0].cast<size_t>(),
      reinterpret_cast<const int64_t *>(
          t[4].cast<py::array_t<uint8_t>>().request().ptr));
}

py::tuple index_server_state_serialization_helper(
    const SharedMemoryIndexServerState &state) {
  GEAR_DEBUG_PRINT("========> pre-serialization info: \n\t key: %d, "
                   "\n\tnum_clients %zu, \n\tcapacity %zu.\n",
                   state.key, state.num_clients, state.capacity);
  py::tuple ret =
      py::make_tuple(
          state.key, state.num_clients, state.capacity, state.index_offset,
          py::array_t<uint8_t>(
              {sizeof(sim_status_t) * state.num_clients}, {1},
              reinterpret_cast<const uint8_t *>(state.status.raw->addr)),
          py::array_t<uint8_t>(
              {sizeof(float) * state.capacity}, {1},
              reinterpret_cast<const uint8_t *>(state.weights.raw->addr)),
          py::array_t<uint8_t>(
              {sizeof(int64_t) * state.capacity}, {1},
              reinterpret_cast<const uint8_t *>(state.timesteps.raw->addr)),
          py::array_t<uint8_t>(
              {sizeof(record_t) * state.num_clients}, {1},
              reinterpret_cast<const uint8_t *>(state.records.raw->addr))) +
      index_buffer_state_serialization_helper(state.buffer_state);
  return ret;
}

std::shared_ptr<SharedMemoryIndexServerState>
index_server_state_deserialization_helper(py::tuple t) {
  if (t.size() != 13) {
    throw std::runtime_error(
        "Invalid state for unpickling SharedMemoryIndexServer!");
  }
  py::tuple buffer_tuple(5);
  for (size_t i = 0; i < 5; ++i) {
    buffer_tuple[i] = t[i + 8];
  }
  key_t key = t[0].cast<key_t>();
  size_t num_clients = t[1].cast<size_t>();
  size_t capacity = t[2].cast<size_t>();
  size_t index_offset = t[3].cast<size_t>();
  GEAR_DEBUG_PRINT("========> pre-deserialization info: \n\t key: %d, "
                   "\n\tnum_clients %zu, \n\tcapacity %zu.\n",
                   key, num_clients, capacity);
  auto buffer_state = index_buffer_state_deserialization_helper(buffer_tuple);
  return std::make_shared<SharedMemoryIndexServerState>(
      capacity, num_clients, index_offset, key,
      t[3].cast<py::array_t<uint8_t>>().request().ptr,
      t[4].cast<py::array_t<uint8_t>>().request().ptr,
      t[5].cast<py::array_t<uint8_t>>().request().ptr,
      t[6].cast<py::array_t<uint8_t>>().request().ptr, *buffer_state);
}

void register_module(py::module &m) {
  register_client(m);
  register_request(m);
  register_server(m);
  register_indexset(m);
}

void register_client(py::module &m) {
  py::class_<RawSimulStatus, std::shared_ptr<RawSimulStatus>>(m,
                                                              "RawSimulStatus")
      .def_readonly("ownership", &RawSimulStatus::ownership)
      .def_readonly("terminated", &RawSimulStatus::terminated)
      .def_readonly("alloc", &RawSimulStatus::alloc)
      .def_readonly("valid", &RawSimulStatus::valid)
      .def_readonly("tindex", &RawSimulStatus::tindex)
      .def_readonly("timestep", &RawSimulStatus::timestep)
      .def_readonly("weight", &RawSimulStatus::weight);

  py::class_<SharedMemoryIndexClient, std::shared_ptr<SharedMemoryIndexClient>>(
      m, "SharedMemoryIndexClient")
      .def(py::init<key_t, size_t, size_t>(), py::arg("key"),
           py::arg("client_rank"), py::arg("num_clients"),
           "SharedMemoryIndexClient constructor")
      .def_property_readonly("num_clients",
                             [](const SharedMemoryIndexClient &self) {
                               return self.get_num_clients();
                             })
      .def("connect", &SharedMemoryIndexClient::connect)
      .def("release", &SharedMemoryIndexClient::release)
      .def("acquire", &SharedMemoryIndexClient::acquire)
      .def("wait", &SharedMemoryIndexClient::wait)
      .def("get_index", &SharedMemoryIndexClient::get_index)
      .def("writeback", &SharedMemoryIndexClient::writeback)
      .def("get_timestep", &SharedMemoryIndexClient::get_timestep)
      .def("get_status_unsafe", &SharedMemoryIndexClient::get_status_unsafe)
      .def("step_inc", &SharedMemoryIndexClient::step_inc);
}

void register_request(py::module &m) {
  py::class_<InferenceRequestArray, std::shared_ptr<InferenceRequestArray>>(
      m, "InferenceRequestArray")
      .def_readonly("count", &InferenceRequestArray::count)
      .def("source_element_view",
           [](InferenceRequestArray &self, size_t i) { return self.srcs[i]; })
      .def("index_element_view",
           [](InferenceRequestArray &self, size_t i) { return self.idxs[i]; })
      .def("timestep_element_view",
           [](InferenceRequestArray &self, size_t i) { return self.tss[i]; })
      .def("source_tensor_view",
           [](InferenceRequestArray &self) {
             return convert_data_pointer_as_tensor<int32_t>(self.srcs,
                                                            self.count);
           })
      .def("index_tensor_view",
           [](InferenceRequestArray &self) {
             return convert_data_pointer_as_tensor<int32_t>(self.srcs,
                                                            self.count);
           })
      .def("timestep_tensor_view", [](InferenceRequestArray &self) {
        return convert_data_pointer_as_tensor<int64_t>(self.tss, self.count);
      });

  py::class_<WeightUpdateRequestArray,
             std::shared_ptr<WeightUpdateRequestArray>>(
      m, "WeightUpdateRequestArray")
      .def_readonly("count", &WeightUpdateRequestArray::count)
      .def("updated_element_view", [](WeightUpdateRequestArray &self,
                                      size_t i) { return self.updated[i]; })
      .def("timestep_element_view", [](WeightUpdateRequestArray &self,
                                       size_t i) { return self.timesteps[i]; })
      .def("weight_element_view", [](WeightUpdateRequestArray &self, size_t i) {
        return self.weights[i];
      });

  py::class_<CachedRequest, std::shared_ptr<CachedRequest>>(m, "CachedRequest")
      .def(py::init<size_t>())
      .def_readonly("capacity", &CachedRequest::capacity)
      .def_readonly("iarr", &CachedRequest::iarr)
      .def_readonly("warr", &CachedRequest::warr)
      .def("continuous", [](CachedRequest &self, torch::Tensor idxs,
                            torch::Tensor weights) {
        int64_t num;
        Int64Span idxs_span_interface = Int64Span::from_tensor(idxs);
        Float32Span weights_span_interface = Float32Span::from_tensor(weights);
        self.warr.continuous(num, idxs_span_interface, weights_span_interface);
        return num;
      });
}

void register_server(py::module &m) {

  py::class_<IndexBufferState, std::shared_ptr<IndexBufferState>>(
      m, "IndexBufferState")
      .def_readonly("head", &IndexBufferState::head)
      .def_readonly("tail", &IndexBufferState::tail)
      .def_readonly("length", &IndexBufferState::length)
      .def_readonly("capacity", &IndexBufferState::capacity)
      .def_property(
          "data",
          [](const IndexBufferState &state) {
            if (state.data.raw->addr == nullptr) {
              return gear::memory::MemoryPtr(reinterpret_cast<void *>(0));
            }
            return gear::memory::MemoryPtr(
                reinterpret_cast<void *>(state.data.raw->addr));
          },
          nullptr)
      .def(py::pickle(
          // __getstate__ method
          [](const IndexBufferState &state) {
            return index_buffer_state_serialization_helper(state);
          },
          // __setstate__ method
          [](const py::tuple t) {
            return index_buffer_state_deserialization_helper(t);
          }));

  py::class_<SharedMemoryIndexServerState,
             std::shared_ptr<SharedMemoryIndexServerState>>(
      m, "SharedMemoryIndexServerState")
      .def(py::pickle(
          [](const SharedMemoryIndexServerState &state) {
            return index_server_state_serialization_helper(state);
          },
          [](py::tuple t) {
            return index_server_state_deserialization_helper(t);
          }));

  py::class_<SharedMemoryIndexServer, std::shared_ptr<SharedMemoryIndexServer>>(
      m, "SharedMemoryIndexServer")
      .def(py::init<key_t, size_t, size_t, size_t>())
      .def_property_readonly("capacity", &SharedMemoryIndexServer::get_capacity)
      .def_property_readonly("weights",
                             [](SharedMemoryIndexServer &self) {
                               return convert_data_pointer_as_tensor<float>(
                                   self.get_weights(), self.get_capacity());
                             })
      .def_property_readonly("timesteps",
                             [](SharedMemoryIndexServer &self) {
                               return convert_data_pointer_as_tensor<int64_t>(
                                   self.get_timesteps(), self.get_capacity());
                             })
      .def_property_readonly(
          "num_clients",
          [](SharedMemoryIndexServer &self) { return self.get_num_clients(); })
      .def("connect", &SharedMemoryIndexServer::connect)
      .def("scan", &SharedMemoryIndexServer::scan)
      .def("callback", &SharedMemoryIndexServer::callback)
      .def("get_state", &SharedMemoryIndexServer::get_state)
      .def("set_state", &SharedMemoryIndexServer::set_state)
      .def(py::pickle(
          // __getstate__ method
          [](const SharedMemoryIndexServer &s) {
            auto state = s.get_state();
            return index_server_state_serialization_helper(state);
          }, // __setstate__ method
          [](py::tuple t) {
            auto state = index_server_state_deserialization_helper(t);
            auto s = std::make_shared<SharedMemoryIndexServer>(
                state->key, state->num_clients, state->index_offset,
                state->capacity);
            GEAR_DEBUG_PRINT(
                "SharedMemoryIdexServer got deserialized state.\n");
            s->set_state(*state);
            return s;
          }

          ));
}

void register_indexset(py::module &m) {
  py::class_<IndexsetState, std::shared_ptr<IndexsetState>>(m, "IndexsetState")
      .def_readonly("global_capacity", &IndexsetState::global_capacity)
      .def_readonly("local_capacity", &IndexsetState::local_capacity)
      .def_readonly("index_offset", &IndexsetState::index_offset)
      .def_readonly("mem", &IndexsetState::mem)
      .def(py::pickle(
          // __getstate__ method
          [](const IndexsetState &s) {
            return indexset_state_serialization_helper(s);
          }, // __setstate__ method
          [](py::tuple t) {
            return indexset_state_deserialization_helper(t);
          }));
  py::class_<Indexset, std::shared_ptr<Indexset>>(
      m, "Indexset", /*TODO: add docstring*/ R"mydelimiter(
  /* your docstrings here */ 
  )mydelimiter")
      .def(py::init<size_t, size_t, size_t, bool, key_t, bool>(),
           py::arg("global_capacity"), py::arg("local_capacity"),
           py::arg("index_offet"), py::arg("shared") = false,
           py::arg("key") = 0, py::arg("create") = false)
      .def_static(
          "load_state",
          [](IndexsetState &state, bool shared, key_t key, bool create) {
            return std::make_shared<Indexset>(state, shared, key, create);
          },
          /*TODO: add docstring*/ R"mydelimiter(
          /* your docstrings here */ 
          )mydelimiter")
      .def_property(
          "timesteps",
          [](Indexset &self) {
            return convert_data_pointer_as_tensor<int64_t>(
                self.get_timesteps(), self.get_local_capacity());
          },
          &Indexset::set_timestep, /*TODO: add docstring*/ R"mydelimiter(
          /* your docstrings here */ 
          )mydelimiter")
      .def_property(
          "weights",
          [](Indexset &self) {
            return convert_data_pointer_as_tensor<float>(
                self.get_weights(), self.get_local_capacity());
          },
          &Indexset::set_weight, /*TODO: add docstring*/ R"mydelimiter(
          /* your docstrings here */ 
          )mydelimiter")
      .def_property_readonly(
          "global_capacity",
          [](const Indexset &s) { return s.get_global_capacity(); })
      .def_property_readonly(
          "local_capacity",
          [](const Indexset &s) { return s.get_local_capacity(); })
      .def_property_readonly(
          "index_offset",
          [](const Indexset &s) { return s.get_index_offset(); })
      .def_property_readonly("is_sharing",
                             [](const Indexset &s) { return s.is_sharing(); })
      .def_property_readonly("shm_key",
                             [](const Indexset &s) { return s.get_shm_key(); })
      .def("get_state", &Indexset::get_state)
      // .def(py::pickle(
      //     // __getstate__ method
      //     [](const Indexset &s) {
      //       IndexsetState state = s.get_state();
      //       return indexset_state_serialization_helper(state);
      //     }, // __setstate__ method
      //     [](py::tuple t) {
      //       IndexsetState state = indexset_state_deserialization_helper(t);
      //       return Indexset(state);
      //     }))
      ;
}

} // namespace gear::index