#pragma once

#include <cstdint>
#include <limits>

// constexpr float INFINITY = std::numeric_limits<double>::infinity();
// constexpr float NEG_INFINITY = -std::numeric_limits<double>::infinity();

// ==== CUDA Device Properties ======
constexpr int GPU_PAGE_SIZE = 64 * 1024; // 64KB page size
constexpr uint32_t WARP_SIZE = 32;
constexpr int WARP_SIZE_BITS = 5;
constexpr int WARP_MASK = -1 ^ 0x1f;

constexpr int MAX_THREADS_PER_BLOCK = 1024;
constexpr int MAX_WARP_NUM = 32;
constexpr int MAX_NUM_BLOCKS = 1048576;
constexpr int MAX_THREAD_DIM[3] = {1024, 1024, 64};
constexpr int MAX_GRID_SIZE[3] = {2147483647, 65535, 65535};

constexpr int unroll_factor = 4;

