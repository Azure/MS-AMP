# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)

project(msampdist LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/)

find_package(CUDA REQUIRED)
find_package(NCCL REQUIRED)

add_library(msampdist SHARED dist.cpp)

target_include_directories(msampdist PUBLIC ${CUDA_INCLUDE_DIRS} ${NCCL_INCLUDE_DIRS})
target_link_libraries(msampdist PUBLIC ${CUDA_LIBRARIES} ${CUDA_LIBRARIES})

install(TARGETS msampdist)
