# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Find the RCCL libraries
#
# The following variables are optionally searched for defaults
#  ROCM_PATH:    Base directory where all RCCL components are found
#
# The following are set after configuration is done:
#  RCCL_FOUND
#  RCCL_INCLUDE_DIR
#  RCCL_LIBRARY

if(NOT DEFINED ENV{ROCM_PATH})
    # Run hipconfig -p to get ROCm path
    execute_process(
      COMMAND hipconfig -R
      RESULT_VARIABLE HIPCONFIG_RESULT
      OUTPUT_VARIABLE ROCM_PATH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Check if hipconfig was successful
    if(NOT HIPCONFIG_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to run hipconfig -p. Make sure ROCm is installed and hipconfig is available.")
    endif()
else()
  set(ROCM_PATH $ENV{ROCM_PATH})
endif()

find_path(RCCL_INCLUDE_DIR NAMES rccl.h
        PATHS ${ROCM_PATH}/include/rccl /usr/local/include/rccl)

find_library(RCCL_LIBRARY NAMES rccl
        PATHS ${ROCM_PATH}/lib /usr/local/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RCCL DEFAULT_MSG RCCL_INCLUDE_DIR RCCL_LIBRARY)

if (RCCL_FOUND)
    message(STATUS "Found RCCL (include: ${RCCL_INCLUDE_DIR}, library: ${RCCL_LIBRARY})")
    mark_as_advanced(RCCL_INCLUDE_DIR RCCL_LIBRARY)

endif ()
