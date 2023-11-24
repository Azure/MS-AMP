// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// The file is adapted from https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/util/vectorized_pointwise.h.

#ifndef MSAMP_VECTORIZED_POINTWISE_H
#define MSAMP_VECTORIZED_POINTWISE_H

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../../common/include/common.h"
#include "../../common/include/utils.cuh"
#include "../../common/include/concurrency.h"

namespace msamp {
/* \brief Helper class that enables storing multiple values of type DType
          as 1 value of type LType.
*/
template <typename DType, int n>
class VectorizedStorage {
 public:
  using LType = typename transformer_engine::BytesToType<sizeof(DType) * n>::Type;
  constexpr static int nvec = n;
  union vectorized_storage {
    LType aligned;
    DType separate[nvec];  // NOLINT(*)

    inline __device__ vectorized_storage() {}
    inline __device__ ~vectorized_storage() {}
  } scratch_;

  inline __device__ VectorizedStorage() {}
  inline __device__ VectorizedStorage(const VectorizedStorage<DType, n>& y2) {
      scratch_.aligned = y2.scratch_.aligned;
  }
  inline __device__ VectorizedStorage(const LType &y2) {
      scratch_.aligned = y2;
  }
  inline __device__ VectorizedStorage<DType, n>& operator+=(
      const VectorizedStorage<DType, n>& rhs) {
    #pragma unroll
    for (int i = 0; i < nvec; ++i) {
      scratch_.separate[i] = add_elem(scratch_.separate[i], rhs.scratch_.separate[i]);
    }
    return *this;
  }
  inline __device__ ~VectorizedStorage() {}
};

// Returns const LType is DType is const
template <typename DType, typename LType>
struct select_const {
  using type = LType;
};

template <typename DType, typename LType>
struct select_const<const DType, LType> {
  using type = const LType;
};


/* \brief Helper class that enables accessing multiple values of type DType
          as 1 value of type LType. Additional aligned template argument
          allows performance optimizations if the pointer and the size of
          the allocation is aligned to sizeof(LType) / sizeof(DType) elements.
*/
template <typename DType, int nvec, bool aligned = false>
class VectorizedAccessor {
 public:
  using StorageType = VectorizedStorage<typename std::remove_const<DType>::type,
                                        nvec>;
  using LType = typename select_const<DType, typename StorageType::LType>::type;
  StorageType storage_;

  LType* aligned_ptr_;
  DType* unaligned_ptr_;
  int alignment_;
  size_t n_elems_;

  inline __device__ VectorizedAccessor(DType* const ptr, const size_t size) {
    unaligned_ptr_ = ptr;
    if (aligned) {
      alignment_ = 0;
      aligned_ptr_ = reinterpret_cast<LType*>(ptr);
      n_elems_ = (size + nvec - 1) / nvec;
    } else {
      size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
      alignment_ = (ptr_as_number % sizeof(LType)) / sizeof(DType);
      aligned_ptr_ = reinterpret_cast<LType*>(ptr - alignment_);
      n_elems_ = (size + alignment_ + nvec - 1) / nvec;
    }
  }

  /* \brief Alignment of the input pointer in elements. */
  inline __device__ int alignment() const {
    return alignment_;
  }

  /* \brief Access to separate elements. */
  inline __device__ DType* separate() {
    return storage_.scratch_.separate;
  }

  /* \brief Number of aligned elements that span the entire input tensor. */
  inline __device__ size_t num_aligned_elements() const {
    return n_elems_;
  }

  /* \brief Load values from the input.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
  inline __device__ void load(const size_t id, const size_t N) {
    if (aligned) {
      storage_.scratch_.aligned = aligned_ptr_[id];
    } else {
      if (id > 0 && id < n_elems_ - 1) {
        storage_.scratch_.aligned = aligned_ptr_[id];
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          DType* ptr = reinterpret_cast<DType*>(&(aligned_ptr_[id])) + j;
          if (reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(unaligned_ptr_) &&
              reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(unaligned_ptr_ + N)) {
            storage_.scratch_.separate[j] = *ptr;
          } else {
            storage_.scratch_.separate[j] = DType();
          }
        }
      }
    }
  }
};

/* \brief Class used for vectorized read-only access. */
template <typename DType, int nvec, bool aligned = false>
class VectorizedLoader : public VectorizedAccessor<const DType, nvec, aligned> {
 public:
  inline __device__ VectorizedLoader(const DType* ptr, const size_t N) :
    VectorizedAccessor<const DType, nvec, aligned>(ptr, N) {
  }
};

/* \brief Class used for vectorized writable access. */
template <typename DType, int nvec, bool aligned = false>
class VectorizedStorer : public VectorizedAccessor<DType, nvec, aligned> {
 public:
  inline __device__ VectorizedStorer(DType* ptr, const size_t N) :
    VectorizedAccessor<DType, nvec, aligned>(ptr, N) {
  }

  /* \brief Store values to the output.
     \param id Aligned index of the element.
     \param N size of the tensor.
  */
  inline __device__ void store(const size_t id, const size_t N) {
    if (aligned) {
      this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
    } else {
      if (id > 0 && id < this->n_elems_ - 1) {
        this->aligned_ptr_[id] = this->storage_.scratch_.aligned;
      } else {
#pragma unroll
        for (int j = 0; j < nvec; ++j) {
          DType* ptr = reinterpret_cast<DType*>(&(this->aligned_ptr_[id])) + j;
          if (reinterpret_cast<size_t>(ptr) >= reinterpret_cast<size_t>(this->unaligned_ptr_) &&
              reinterpret_cast<size_t>(ptr) < reinterpret_cast<size_t>(this->unaligned_ptr_ + N)) {
            *ptr = this->storage_.scratch_.separate[j];
          }
        }
      }
    }
  }
};


constexpr int unary_kernel_threads = 512;
constexpr float e4m3_max = 448.0;
constexpr float e5m2_max = 57344.0;

extern __device__ msamp::DeviceSyncer device_syncer;

template <int nvec, bool aligned,
          typename ComputeType,
          typename InputType,
          typename OutputType>
__launch_bounds__(unary_kernel_threads)
__global__ void add_to_fp8_kernel(InputType *input,
                             OutputType *output,
                             ComputeType *scale,
                             ComputeType *scale_inv,
                             ComputeType *amax,
                             const size_t N,
                             const size_t num_aligned_elements) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *amax = 0;
  }
  device_syncer.sync(gridDim.x, -1);

  // input is high precision, output is fp8
  VectorizedStorer<OutputType, nvec, aligned> output_storer(output, N);
  VectorizedStorer<InputType, nvec, aligned> input_storer(input, N);

  ComputeType max = 0;
  ComputeType s = 0;
  if constexpr (is_fp8<OutputType>::value) {
      if (scale_inv != nullptr) s = *scale_inv;
  }
  const int warp_id = threadIdx.x / THREADS_PER_WARP;

  const size_t M = num_aligned_elements;

  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    input_storer.load(tid, N);
    output_storer.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const InputType val1 = static_cast<InputType>(input_storer.separate()[i]);
      const ComputeType val2 = static_cast<ComputeType>(output_storer.separate()[i]);

      InputType temp = static_cast<InputType>(val2 * s);

      if constexpr (is_half<InputType>::value) {
          temp = static_cast<ComputeType>(__hadd(temp, val1));
      } else {
          temp += val1;
      }

      if constexpr (is_fp8<OutputType>::value) {
        __builtin_assume(max >= 0);
        max = fmaxf(fabsf(temp), max);
      }
    }
  }

  if constexpr (is_fp8<OutputType>::value) {
    /* warp tile amax reduce*/
    max =  transformer_engine::reduce_max<unary_kernel_threads / THREADS_PER_WARP>(max, warp_id);

    if (threadIdx.x == 0 && amax != nullptr) {
        static_assert(std::is_same<ComputeType, float>::value);
        transformer_engine::atomicMaxFloat(amax, max);
    }
  }

  device_syncer.sync(gridDim.x, -1);

  /* Compute scaling factor, translate the following python code to c++:
    exp = torch.floor(torch.log2(fp_max / amax)) - margin
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(exp < 0, 1 / sf, sf)
  */
  ComputeType amax_value = *amax;

  ComputeType fp_max = std::is_same<OutputType, fp8e4m3>::value ? e4m3_max : e5m2_max;

  ComputeType exp = floorf(log2f(fp_max/(amax_value)));
  ComputeType sf = roundf(powf(2, fabsf(exp)));

  if (amax_value <= 0 || !isfinite(amax_value)) {
    sf = *scale;
  }

  if (exp < 0) {
    sf = 1 / sf;
  }

  // using new scaling factor to quantize the input
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
       tid < M;
       tid += gridDim.x * blockDim.x) {
    input_storer.load(tid, N);
    output_storer.load(tid, N);
#pragma unroll
    for (int i = 0; i < nvec; ++i) {
      const InputType val1 = static_cast<InputType>(input_storer.separate()[i]);
      const ComputeType val2 = static_cast<ComputeType>(output_storer.separate()[i]);
    
      InputType temp1 = static_cast<InputType>(val2 * s);
      
      if constexpr (is_half<InputType>::value) {
          temp1 = static_cast<ComputeType>(__hadd(temp1, val1));
      } else {
          temp1 += val1;
      }
      ComputeType temp2 = sf * static_cast<ComputeType>(temp1);
      output_storer.separate()[i] = static_cast<OutputType>(temp2);
    }
    output_storer.store(tid, N);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
      *scale = sf;
      *scale_inv = 1.0 / sf;
  }
}


namespace {

inline size_t get_num_aligned_elements(const void *ptr, const size_t lead_dim,
                                        const int nvec, const int size) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  int alignment = (ptr_as_number % (nvec * size)) / size;
  return DIVUP(lead_dim + alignment, static_cast<size_t>(nvec));
}

enum class Alignment {
  SAME_ALIGNED,  // All tensors aligned
  SAME_UNALIGNED,  // All tensors have the same misalignment
  DIFFERENT  // Tensors have different alignment
};

inline int CalcAlignment(const void *ptr, const int size) {
  size_t ptr_as_number = reinterpret_cast<size_t>(ptr);
  return ptr_as_number % size;
}

/* \brief Check alignment of the inputs and outputs when using vectorized accesses.
   \param lead_dim Leading dimension of the tensors.
   \param other_dim The size of the other dimensions of the tensors.
   \param nvec Length of the vector.
   \param ptrs Inputs and Outputs to the operator.
*/
template <typename... T>
Alignment CheckAlignment(const size_t lead_dim,
                         const int nvec,
                         const T... ptrs
                        ) {
  std::vector<int> alignments;
  alignments.reserve(sizeof...(T));

  // calculate the alignments of all ptrs and store them into alignments
  (..., alignments.push_back(CalcAlignment(ptrs, sizeof(*ptrs) * nvec)));

  bool all_same = std::all_of(alignments.cbegin(), alignments.cend(),
    [alignments](int val) {return val == alignments.front();});
  if (!all_same) {
    return Alignment::DIFFERENT;
  }

  if (alignments.front() == 0 &&
      lead_dim % nvec == 0) {
    // all alignment are 0
    return Alignment::SAME_ALIGNED;
  } else {
    return Alignment::SAME_UNALIGNED;
  }
}

}

template <int nvec,
          typename InputType,
          typename OutputType>
void VectorizedAddToFp8KernelLauncher(InputType *input,
                                      OutputType *output,
                                      fp32 *scale,
                                      fp32 *scale_inv,
                                      fp32 *amax,
                                      const size_t N,
                                      cudaStream_t stream) {
  if (N != 0) {
    auto align = CheckAlignment(N, nvec, input, output);

    size_t num_aligned_elements = get_num_aligned_elements(input, N, nvec,
                                                           sizeof(InputType));
    constexpr size_t threads = unary_kernel_threads;
    size_t num_blocks = DIVUP(num_aligned_elements, threads);

    // We use DeviceSyncer to sync the amax value between blocks, the block number should be less than 
    // (SMCount*MaxThreadsPerSM)/unary_kernel_threads, which is 132*2048/512 = 528 on H100 SXM. We set 
    // max_blocks to half of 528 to make sure it works on other H100 GPUs.  
    // constexpr size_t max_blocks = 65535;
    constexpr size_t max_blocks = 264;
    num_blocks = std::min(num_blocks, max_blocks);

    switch (align) {
      case Alignment::SAME_ALIGNED:
        add_to_fp8_kernel<nvec, true, fp32><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, N, num_aligned_elements);
        break;
      case Alignment::SAME_UNALIGNED:
        add_to_fp8_kernel<nvec, false, fp32><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, N, num_aligned_elements);
        break;
      case Alignment::DIFFERENT: {
        // If the pointers are aligned differently we cannot vectorize
        add_to_fp8_kernel<1, true, fp32><<<num_blocks, threads, 0, stream>>>(
            input, output, scale, scale_inv, amax, N, num_aligned_elements);
        break;
      }
    }
  }
}

} // namespace msamp

#endif // MSAMP_VECTORIZED_POINTWISE_H