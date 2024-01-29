// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <stdio.h>
#include <dlfcn.h>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <rccl.h>
#define stream_t hipStream_t
// TODO: Remove ncclFp8E4M3 and ncclFp8E5M2 after rccl supports FP8
#define ncclFp8E4M3 ncclUint8
#define ncclFp8E5M2 ncclInt8
#else
#include <nccl.h>
#define stream_t cudaStream_t

#endif

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

void f() {

}

/**
 * It will override the ncclAllReduce function in nccl library if this library is set to LD_PRELOAD.
*/
#undef ncclAllReduce
extern "C"
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
  ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, stream_t stream) {
  using ncclAllReduceFuncType = ncclResult_t (*)
    (const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, stream_t);
  ncclAllReduceFuncType real_nccl_all_reduce = reinterpret_cast<ncclAllReduceFuncType>(dlsym(RTLD_NEXT, "ncclAllReduce"));
  if (real_nccl_all_reduce == nullptr) {
    printf("MSAMP_DistOp: Failed to find ncclAllReduce symbol");
    return ncclSystemError;
  }
  if (gFP8Mode == kFP8E4M3) {
    return real_nccl_all_reduce(sendbuff, recvbuff, count, ncclFp8E4M3, op, comm, stream);
  } else if (gFP8Mode == kFp8E5M2) {
    return real_nccl_all_reduce(sendbuff, recvbuff, count, ncclFp8E5M2, op, comm, stream);
  } else {
    return real_nccl_all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
  }
  return real_nccl_all_reduce(sendbuff, recvbuff, count, datatype, op, comm, stream);

}

/**
 * It will override the ncclReduce function in nccl library if this library is set to LD_PRELOAD.
*/
#undef ncclReduce
extern "C"
ncclResult_t  ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
  ncclRedOp_t op, int root, ncclComm_t comm, stream_t stream)
{
  using ncclReduceFuncType = ncclResult_t (*)
    (const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, int, ncclComm_t, stream_t);
  ncclReduceFuncType real_nccl_reduce = reinterpret_cast<ncclReduceFuncType>(dlsym(RTLD_NEXT, "ncclReduce"));
  if (real_nccl_reduce == nullptr) {
    printf("MSAMP_DistOp: Failed to find ncclReduce symbol");
    return ncclSystemError;
  }
  if (gFP8Mode == kFP8E4M3) {
    return real_nccl_reduce(sendbuff, recvbuff, count, ncclFp8E4M3, op, root, comm, stream);
  } else if (gFP8Mode == kFp8E5M2) {
    return real_nccl_reduce(sendbuff, recvbuff, count, ncclFp8E5M2, op, root, comm, stream);
  } else {
    return real_nccl_reduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
  }
}

/**
 * It will override the ncclReduceScatter function in nccl library if this library is set to LD_PRELOAD.
*/
#undef ncclReduceScatter
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, stream_t stream) {
  using ncclReduceScatterFuncType = ncclResult_t (*)
    (const void*, void*, size_t, ncclDataType_t, ncclRedOp_t, ncclComm*, stream_t);
  ncclReduceScatterFuncType real_nccl_reduce_scatter = reinterpret_cast<ncclReduceScatterFuncType>(dlsym(RTLD_NEXT, "ncclReduceScatter"));
  if (real_nccl_reduce_scatter == nullptr) {
    printf("MSAMP_DistOp: Failed to find ncclReduceScatter symbol");
    return ncclSystemError;
  }
  if (gFP8Mode == kFP8E4M3) {
    return real_nccl_reduce_scatter(sendbuff, recvbuff, recvcount, ncclFp8E4M3, op, comm, stream);
  } else if (gFP8Mode == kFp8E5M2) {
    return real_nccl_reduce_scatter(sendbuff, recvbuff, recvcount, ncclFp8E5M2, op, comm, stream);
  } else {
    return real_nccl_reduce_scatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream);
  }
}
