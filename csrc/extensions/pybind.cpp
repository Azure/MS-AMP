// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <string>
#include <torch/extension.h>

#include "include/extensions.h"

std::string version() { return "1.0.0"; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_to_fp8", &msamp::add_to_fp8, "Add to fp8");
    m.def("adamw_fp8_stage1_compute", &msamp::adamw_fp8_stage1_compute, "Adamw FP8 Stage1 Compute");
    m.def("adamw_fp8_stage2_compute", &msamp::adamw_fp8_stage2_compute, "Adamw FP8 Stage2 Compute");
    m.def("version", &version);
}
