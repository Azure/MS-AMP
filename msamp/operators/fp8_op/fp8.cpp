#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>

enum FP8ModeType {kFP8Disabled, kFP8E4M3, kFp8E5M2};
static enum FP8ModeType gFP8Mode = kFP8Disabled;

extern "C"
void disable_fp8() {
  gFP8Mode = kFP8Disabled;
}

extern "C"
void enable_fp8_e4m3() {
  gFP8Mode = kFP8E4M3;
}

extern "C"
void enable_fp8_e5m2() {
  gFP8Mode = kFp8E5M2;
}

/**
 * It will override the ncclAllReduce function in nccl library if this library is set to LD_PRELOAD.
*/
#undef ncclAllReduce
extern "C"
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream) {
  using ncclAllReduceFuncType = ncclResult_t (*)
    (const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t);
  ncclAllReduceFuncType real_nccl_all_reduce = nullptr;
  if (!real_nccl_all_reduce) {
    real_nccl_all_reduce = reinterpret_cast<ncclAllReduceFuncType>(dlsym(RTLD_NEXT, "ncclAllReduce"));
  }
  if (gFP8Mode == kFP8E4M3) {
    return real_nccl_all_reduce(sendbuff, recvbuff, count, ncclFp8E4M3, op, comm, stream);
  } else if (gFP8Mode == kFp8E5M2) {
    return real_nccl_all_reduce(sendbuff, recvbuff, count, ncclFp8E5M2, op, comm, stream);
  } else {
    return real_nccl_all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
  }

}