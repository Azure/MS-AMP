
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// The file is from https://github.com/microsoft/mscclpp/blob/main/include/mscclpp/poll.hpp.

#ifndef MSAMP_POLL_H_
#define MSAMP_POLL_H_

#include <cstdint>

extern "C" __device__ void __assert_fail(const char *__assertion, const char *__file, unsigned int __line,
                                         const char *__function) __THROW;

// If a spin is stuck, print a warning and keep spinning.
#define POLL_MAYBE_JAILBREAK(__cond, __max_spin_cnt)                     \
  do {                                                                   \
    int64_t __spin_cnt = 0;                                              \
    while (__cond) {                                                     \
      if (__max_spin_cnt >= 0 && __spin_cnt++ == __max_spin_cnt) {       \
        __assert_fail(#__cond, __FILE__, __LINE__, __PRETTY_FUNCTION__); \
      }                                                                  \
    }                                                                    \
  } while (0);

#endif  // MSAMP_POLL_H_