#include "debug.h"
#include "kernel_launchers.h"
#include "macros.h"

namespace gear::cuda {
void init(int device_id) {
  int dev = -1;
  CUDA_CHECK(cudaGetDevice(&dev));

  CUDA_CHECK(cudaSetDevice(device_id));
  GEAR_DEBUG_PRINT("<GEAR::init> Default device id %d, set to %d", dev,
                   device_id);
}
} // namespace gear::cuda

void register_cuda_init(pybind11::module &m) {
  m.def("init", &gear::cuda::init);
}