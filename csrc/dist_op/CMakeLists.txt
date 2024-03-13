# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.10)

project(msamp_dist LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/)

find_package(CUDA QUIET)

if (CUDA_FOUND)
    find_package(NCCL REQUIRED)
    add_library(msamp_dist SHARED dist.cpp)
    target_include_directories(msamp_dist PUBLIC ${CUDA_INCLUDE_DIRS} ${NCCL_INCLUDE_DIR})
    target_link_libraries(msamp_dist PUBLIC ${CUDA_LIBRARIES} ${NCCL_LIBRARY})
else()
    enable_language(HIP)
    find_package(HIP REQUIRED)
    find_package(RCCL REQUIRED)
    add_library(msamp_dist SHARED dist.cpp)

    target_include_directories(msamp_dist PUBLIC ${HIP_INCLUDE_DIRS} ${RCCL_INCLUDE_DIR})
    target_link_libraries(msamp_dist PUBLIC ${HIP_LIBRARIES} ${RCCL_LIBRARY})
    set_source_files_properties(dist.cpp PROPERTIES LANGUAGE HIP)

endif()

install(TARGETS msamp_dist)
