#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t error_code = callstr;                                          \
    if (error_code != cudaSuccess) {                                           \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":"    \
                << __LINE__ << "\n";                                           \
      assert(0);                                                               \
    }                                                                          \
  }

#define CUDA_1D_KERNEL_LOOP(i, n)                                              \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);              \
       i += blockDim.x * gridDim.x)

#define CUDA_2D_KERNEL_LOOP(i, n, j, m)                                        \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);              \
       i += blockDim.x * gridDim.x)                                            \
    for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < (m);            \
         j += blockDim.y * gridDim.y)
