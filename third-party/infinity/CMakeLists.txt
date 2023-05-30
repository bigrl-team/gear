cmake_minimum_required(VERSION 3.16)
project(infinity LANGUAGES CXX)

message("infinity project directory" ${PROJECT_SOURCE_DIR})

get_filename_component(INFINITY_INCLUDE_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
include_directories(${INFINITY_INCLUDE_DIR})

set(INFINITY_SRCS ${PROJECT_SOURCE_DIR}/core/Context.cpp
${PROJECT_SOURCE_DIR}/memory/Atomic.cpp
${PROJECT_SOURCE_DIR}/memory/Buffer.cpp
${PROJECT_SOURCE_DIR}/memory/Region.cpp
${PROJECT_SOURCE_DIR}/memory/RegionToken.cpp
${PROJECT_SOURCE_DIR}/memory/RegisteredMemory.cpp
${PROJECT_SOURCE_DIR}/queues/QueuePair.cpp
${PROJECT_SOURCE_DIR}/queues/QueuePairFactory.cpp
${PROJECT_SOURCE_DIR}/requests/RequestToken.cpp
${PROJECT_SOURCE_DIR}/utils/Address.cpp)

add_library(infinity SHARED
    ${INFINITY_SRCS}
)

target_compile_features(infinity PRIVATE cxx_std_17)
target_link_libraries(infinity PUBLIC ibverbs)
