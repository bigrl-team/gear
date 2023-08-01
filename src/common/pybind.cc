#include <cstdio>

#include "common/pybind.h"
namespace gear::common {

void register_common_module(py::module &m) {
  register_span(m);
  register_dtypes(m);
  register_range(m);
}

void register_span(py::module &m) {
#define BIND_SPAN_CLASS(CLS)                                                   \
  py::class_<CLS>(m, #CLS)                                                     \
      .def_static("Tensor", &CLS::static_as_tensor)                            \
      .def_static("from_tensor", &CLS::from_tensor)                            \
      .def("copy", &CLS::copy)                                                 \
      .def("as_bool", &CLS::bool8)                                             \
      .def("as_int8", &CLS::int8)                                              \
      .def("as_uint8", &CLS::uint8)                                            \
      .def("as_int16", &CLS::int16)                                            \
      .def("as_int32", &CLS::int32)                                            \
      .def("as_int64", &CLS::int64)                                            \
      .def("as_float32", &CLS::float32)                                        \
      .def("as_float64", &CLS::float64)                                        \
      .def("cast_tensor", &CLS::cast_tensor)                                   \
      .def("tensor", &CLS::tensor);                                            \
  py::implicitly_convertible<torch::Tensor, CLS>();

  BIND_SPAN_CLASS(BoolSpan);
  BIND_SPAN_CLASS(Int8Span);
  BIND_SPAN_CLASS(Uint8Span);
  BIND_SPAN_CLASS(Int16Span);
  BIND_SPAN_CLASS(Int32Span);
  BIND_SPAN_CLASS(Int64Span);
  BIND_SPAN_CLASS(Float32Span);
  BIND_SPAN_CLASS(Float64Span);

#undef BIND_SPAN_CLASS

  m.def("debug_check_implicit_longtensor_to_int64span", [](Int64Span &span) {
    printf("Int64Span: <ptr: %p, size: %ld>.\n", span.ptr, span.size);
  });
  m.def("debug_check_implicit_uint8tensor_to_uint8span", [](Uint8Span &span) {
    printf("Uint8Span: <ptr: %p, size: %ld>.\n", span.ptr, span.size);
  });

  m.def("debug_check_implicit_uint8tensor_to_uint8span", [](Uint8Span &span) {
    printf("Uint8Span: <ptr: %p, size: %ld>.\n", span.ptr, span.size);
  });
}

void register_dtypes(py::module &m) {
  py::enum_<DataType>(m, "DataType", py::arithmetic(),
                      "Meta-class for data-type")
      .value("bool", DataType::kBool, "Bool(1-byte underlying memoryview) type")
      .value("int8", DataType::kByte, "Byte(1-byte underlying memoryview) type")
      .value("uint8", DataType::kChar,
             "Char(1-byte underlying memoryview) type")
      .value("short", DataType::kShort,
             "Int16(2-byte underlying memoryview) type")
      .value("int16", DataType::kShort,
             "Int16(2-byte underlying memoryview) type")
      .value("int", DataType::kInt, "Int32(4-byte underlying memoryview) type")
      .value("int32", DataType::kInt,
             "Int32(4-byte underlying memoryview) type")
      .value("long", DataType::kLong,
             "Int64(8-byte underlying memoryview) type")
      .value("int64", DataType::kLong,
             "Int64(8-byte underlying memoryview) type")
      .value("half", DataType::kHalf,
             "Float16(2-byte underlying memoryview) type")
      .value("float16", DataType::kHalf,
             "Float16(2-byte underlying memoryview) type")
      .value("float", DataType::kFloat,
             "Float32(4-byte underlying memoryview) type")
      .value("float32", DataType::kFloat,
             "Float32(4-byte underlying memoryview) type")
      .value("double", DataType::kDouble,
             "Float64(8-byte underlying memoryview) type")
      .value("float64", DataType::kDouble,
             "Float64(8-byte underlying memoryview) type")
      .export_values();
}

void register_range(py::module &m) {
  py::class_<Range>(m, "Range", /*TODO: add docstring*/ R"mydelimiter(
  /* your docstrings here */ 
  )mydelimiter")
      .def(py::init<size_t, size_t>())
      .def_readwrite("lb", &Range::lb)
      .def_readwrite("hb", &Range::hb);
}
} // namespace gear::common