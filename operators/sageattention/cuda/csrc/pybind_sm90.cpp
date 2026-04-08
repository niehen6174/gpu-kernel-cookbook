/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "attn_cuda_sm90.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("qk_int8_sv_f8_accum_f32_attn_inst_buf", &qk_int8_sv_f8_accum_f32_attn_inst_buf,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("query_scale"), py::arg("key_scale"),
        py::arg("tensor_layout"), py::arg("is_causal"), py::arg("qk_quant_gran"),
        py::arg("sm_scale"), py::arg("return_lse"), py::arg("skip_threshold") = 0.0f);
  m.def("qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf", &qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf,
        py::arg("query"), py::arg("key"), py::arg("value"), py::arg("output"),
        py::arg("query_scale"), py::arg("key_scale"), py::arg("value_scale"),
        py::arg("tensor_layout"), py::arg("is_causal"), py::arg("qk_quant_gran"),
        py::arg("sm_scale"), py::arg("return_lse"), py::arg("skip_threshold") = 0.0f);
}
