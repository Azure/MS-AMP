# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)

project(msamp_dist LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/)

find_package(CUDA REQUIRED)
find_package(NCCL REQUIRED)

add_library(msamp_dist SHARED dist.cpp)

target_include_directories(msamp_dist PUBLIC ${CUDA_INCLUDE_DIRS} ${NCCL_INCLUDE_DIRS})
target_link_libraries(msamp_dist PUBLIC ${CUDA_LIBRARIES} ${CUDA_LIBRARIES})

install(TARGETS msamp_dist)
