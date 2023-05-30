cmake_minimum_required(VERSION 3.20)
project(libgear LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(NCCL_LIBNAME "nccl")
find_path(NCCL_INCLUDE_DIR
    NAMES ${NCCL_LIBNAME}.h
    HINTS
    $ENV{NCCL_HOME}/include
    $ENV{CUDA_ROOT}/include)

find_library(NCCL_LIBRARY
    NAMES ${NCCL_LIBNAME}
    HINTS
    $ENV{NCCL_HOME}/lib/x86_64-linux-gnu
    $ENV{NCCL_HOME}/lib
    $ENV{CUDA_ROOT}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

find_package(Python REQUIRED COMPONENTS Interpreter Development)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python #libtorch_python.so
    HINTS
    ${TORCH_INSTALL_PREFIX}/lib
)
find_package(pybind11 REQUIRED)

add_subdirectory(${PROJECT_SOURCE_DIR}/third-party/infinity)
# set(INFINITY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party/infinity)
# set(INFINITY_SRCS ${INFINITY_SOURCE_DIR}/core/Context.cpp
# ${INFINITY_SOURCE_DIR}/memory/Atomic.cpp
# ${INFINITY_SOURCE_DIR}/memory/Buffer.cpp
# ${INFINITY_SOURCE_DIR}/memory/Region.cpp
# ${INFINITY_SOURCE_DIR}/memory/RegionToken.cpp
# ${INFINITY_SOURCE_DIR}/memory/RegisteredMemory.cpp
# ${INFINITY_SOURCE_DIR}/queues/QueuePair.cpp
# ${INFINITY_SOURCE_DIR}/queues/QueuePairFactory.cpp
# ${INFINITY_SOURCE_DIR}/requests/RequestToken.cpp
# ${INFINITY_SOURCE_DIR}/utils/Address.cpp)

include_directories(
    include
    third-party
    ${NCCL_INCLUDE_DIR}
    ${pybind11_INCLUDE_DIR}
)


file(GLOB_RECURSE SRCS src/**/*.c*)

pybind11_add_module(libgear
    ${SRCS}
)

# target_compile_features(pipedt PRIVATE cxx_std_17)
set_target_properties(libgear PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_features(libgear PRIVATE cuda_std_14)
target_link_directories(libgear PUBLIC ${NCCL_LIBRARY} pybind11::headers)


target_link_libraries(libgear PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(libgear PUBLIC Python::Python ibverbs infinity)
