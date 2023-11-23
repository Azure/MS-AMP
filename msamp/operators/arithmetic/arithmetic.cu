// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "../../common/include/common.h"
#include "../../common/include/utils.cuh"
#include "../../common/include/concurrency.h"
#include "vectorized_pointwise.h"

namespace msamp {
void add_to_fp8(at::Tensor fp8_tensor,
                at::Tensor scale,
                at::Tensor scale_inv,
                at::Tensor amax,
                const at::Tensor& other,
                bool is_e4m3) {
  const size_t N = other.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  TORCH_DTYPE_SWITCH_INPUT(other.scalar_type(), IType,
    SELECT_FP8_TYPE(is_e4m3, OType,
    
      constexpr int nvec = 32 / sizeof(IType);
      
      VectorizedAddToFp8KernelLauncher<nvec>(
        reinterpret_cast<IType*>(other.data_ptr()),
        reinterpret_cast<OType*>(fp8_tensor.data_ptr()),
        reinterpret_cast<fp32*>(scale.data_ptr()),
        reinterpret_cast<fp32*>(scale_inv.data_ptr()),
        reinterpret_cast<fp32*>(amax.data_ptr()),
        N,
        stream
      );
    );
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_to_fp8", &add_to_fp8, "Add to fp8");
}

} // namespace msamp
