// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
// The file is from https://github.com/microsoft/mscclpp/blob/main/include/mscclpp/concurrency.hpp.

#ifndef MSAMP_CONCURRENCY_H_
#define MSAMP_CONCURRENCY_H_

#include "poll.h"

namespace msamp {

/// A device-wide barrier.
struct DeviceSyncer {
 public:
  /// Construct a new DeviceSyncer object.
  DeviceSyncer() = default;

  /// Destroy the DeviceSyncer object.
  ~DeviceSyncer() = default;

#ifdef __CUDACC__
  /// Synchronize all threads inside a kernel. Guarantee that all previous work of all threads in cooperating blocks is
  /// finished.
  /// @param blockNum The number of blocks that will synchronize.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  __forceinline__ __device__ void sync(int blockNum, int64_t maxSpinCount = 100000000) {
    unsigned int maxOldCnt = blockNum - 1;
    __syncthreads();
    if (blockNum == 1) return;
    if (threadIdx.x == 0) {
      // Need a `__threadfence()` before to flip `flag`.
      __threadfence();
      int tmp = isIncFlag_ ^ 1;
      if (tmp) {
        if (atomicInc(&count_, maxOldCnt) == maxOldCnt) {
          flag_ = 1;
        }
        POLL_MAYBE_JAILBREAK(!flag_, maxSpinCount);
      } else {
        if (atomicInc(&count_, maxOldCnt) == maxOldCnt) {
          flag_ = 0;
        }
        POLL_MAYBE_JAILBREAK(flag_, maxSpinCount);
      }
      isIncFlag_ = tmp;
    }
    // We need sync here because only a single thread is checking whether
    // the flag is flipped.
    __syncthreads();
  }
#endif

 private:
  /// The flag to indicate whether the barrier is reached by the latest thread.
  volatile int flag_;
  /// The counter of synchronized blocks.
  unsigned int count_;
  /// The flag to indicate whether to increase or decrease @ref flag_.
  int isIncFlag_;
};

}  // namespace msamp

#endif  // MSAMP_CONCURRENCY_H_