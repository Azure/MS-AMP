#include <torch/extension.h>

namespace msamp {
void add_to_fp8(at::Tensor fp8_tensor,
                at::Tensor scale,
                at::Tensor scale_inv,
                at::Tensor amax,
                const at::Tensor& other,
                bool is_e4m3);

void adamw_fp8_stage1_compute(at::Tensor param, at::Tensor grad, at::Tensor exp_avg_value, float exp_avg_factor,
                              at::Tensor exp_avg_amax, float beta1, at::Tensor exp_avg_sq_value, float exp_avg_sq_factor,
                              at::Tensor exp_avg_sq_amax, float beta2, float eps, int step, float lr,
                              bool bias_correction);

void adamw_fp8_stage2_compute(at::Tensor grad, at::Tensor exp_avg_value, float exp_avg_factor, float new_exp_avg_factor,
                              float beta1, at::Tensor exp_avg_sq_value, float exp_avg_sq_factor,
                              float new_exp_avg_sq_factor, float beta2, int step, bool bias_correction);

}
