// ##########################################################################
// Copyright (C) 2022 COAI @ Tsinghua University

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//         http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ###########################################################################

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> dag_loss(const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, bool require_gradient, int config);
std::tuple<torch::Tensor, torch::Tensor> dag_loss_backward(const torch::Tensor &grad_output, const torch::Tensor &alpha, const torch::Tensor &beta, const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, int config1, int config2);
std::tuple<torch::Tensor, torch::Tensor> dag_best_alignment(const torch::Tensor &match_all, const torch::Tensor &links, const torch::Tensor &output_length, const torch::Tensor &target_length, int config);
torch::Tensor logsoftmax_gather(torch::Tensor word_ins_out, const torch::Tensor &select_idx, bool require_gradient);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dag_loss", &dag_loss, "DAG Loss");
  m.def("dag_loss_backward", &dag_loss_backward, "DAG Loss Backward");
  m.def("dag_best_alignment", &dag_best_alignment, "DAG Best Alignment");
  m.def("logsoftmax_gather", &logsoftmax_gather, "logsoftmax + gather");
}
