// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include "helper.h"
#include "utils.cuh"

template <typename T>
void DecodeMLAAttentionKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor &q, // [token_num, num_heads, head_dim]
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& shift_bias,
    const paddle::optional<paddle::Tensor>& smooth_weight,
    const paddle::Tensor &seq_lens_q, // q_seq_len is 1
    const paddle::Tensor &seq_lens_kv,
    const paddle::Tensor &padding_offsets,
    const paddle::Tensor &cum_offsets,
    const paddle::Tensor &block_table,
    int max_seq_len,
    int max_dec_len,
    float softmax_scale,
    float in_scale,
    bool causal,
    cudaStream_t &stream,
    paddle::Tensor *out);
