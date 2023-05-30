#pragma once

#include <torch/extension.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

// #define TENSOR_DPTR(dtype, tensor) tensor.data_ptr<dtype>()




