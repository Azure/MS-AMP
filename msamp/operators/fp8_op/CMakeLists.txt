# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)

project(msampfp8 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/)

find_package(CUDA REQUIRED)
find_package(NCCL REQUIRED)

add_library(msampfp8 SHARED fp8.cpp)

target_include_directories(msampfp8 PUBLIC ${CUDA_INCLUDE_DIRS} ${NCCL_INCLUDE_DIRS})
target_link_libraries(msampfp8 PUBLIC ${CUDA_LIBRARIES} ${CUDA_LIBRARIES})

install(TARGETS msampfp8)
