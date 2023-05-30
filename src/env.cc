#include <cstdio>
#include <pybind11/pybind11.h>
#include <string>

#include "debug.h"
#include "env.h"
#include "macros.h"

namespace gear::env {
char *create_char_array(size_t length, std::string s, bool left_align = true,
                        char fill_in = '.') {
  char *ptr = new char[length + 1];
  assert(s.length() <= length); // message too long
  if (left_align) {
    for (size_t i = 0; i < s.length(); ++i) {
      ptr[i] = s[i];
    }
    for (size_t i = s.length(); i < length; ++i) {
      ptr[i] = fill_in;
    }
  } else {
    for (size_t i = 0; i < (length - s.length()); ++i) {
      ptr[i] = fill_in;
    }
    for (size_t i = 0; i < s.length(); ++i) {
      ptr[i + length - s.length()] = s[i];
    }
  }
  return ptr;
}

void print_key_value_pair(std::string key, std::string value) {
  char *key_char_arr = create_char_array(PRINT_ALIGNMENT, key, true, '.');
  char *val_char_arr = create_char_array(PRINT_ALIGNMENT, value, false, '.');
  fprintf(stdout, "CHECK %s%s.\n", key_char_arr, val_char_arr);
  delete[] key_char_arr;
  delete[] val_char_arr;
}

void print_compile_options() {
#ifdef GEAR_VERBOSE_DEBUG_ON
  print_key_value_pair("gear_verbose_debug", "enable");
#else
  print_key_value_pair("gear_verbose_debug", "disable");
#endif
}

void print_environ() {
  // print device count
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count))
  print_key_value_pair("visible_device_count", std::to_string(device_count));
}

void register_env_detect_functions(py::module &m) {
  m.def("print_compile_options", &print_compile_options);
  m.def("print_environ", &print_environ);
}

} // namespace gear::env
