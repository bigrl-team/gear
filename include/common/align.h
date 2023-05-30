#pragma once

// https://stackoverflow.com/questions/12778949/cuda-memory-alignment
#if defined(___CUDACC__)
#define GEAR_STRUCT_ALIGN(n) __align__(n)
#elif defined(__GNUC__)
#define GEAR_STRUCT_ALIGN(n) __attribute__((aligned(n)))
#else
#error "GEAR_ALIGN macro definition missed for current compiler"
#endif