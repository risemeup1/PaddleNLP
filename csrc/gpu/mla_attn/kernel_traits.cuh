// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */

#ifndef ATTENTION_HOPPER_KERNEL_TRAITS_CUH_
#define ATTENTION_HOPPER_KERNEL_TRAITS_CUH_

#include <type_traits>

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace mla_attn {

using namespace cute;

template <typename MainloopPipeline, typename MainloopPipelineQ, class DTypeQ, class DTypeKV, class DTypeQKAccum, class DTypeOut, class IdType,
          int BLOCK_SHAPE_KV, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutP, class SmemLayoutRow, class SmemLayoutO>
struct alignas(16) SharedStorageQKVO {
  alignas(16) cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutQ>> smem_q;
  alignas(16) cute::array_aligned<DTypeQ, cute::cosize_v<SmemLayoutP>> smem_p;
  alignas(16) cute::array_aligned<DTypeQKAccum, cute::cosize_v<SmemLayoutRow>> smem_scale;
  union {
    alignas(16) cute::array_aligned<DTypeKV, cute::cosize_v<SmemLayoutK>> smem_kv;
    alignas(16) cute::array_aligned<DTypeOut, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    alignas(16) typename MainloopPipelineQ::SharedStorage pipeline_q;
    alignas(16) typename MainloopPipeline::SharedStorage pipeline_kv;
  };
};

template <bool USE_TMA_LOAD_KV_, int HEAD_DIM_QK_, int HEAD_DIM_VO_, int GROUP_SIZE_, int BLOCK_SHAPE_Q_, int BLOCK_SHAPE_KV_,
          int NUM_STAGES_, typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_, typename NV_TYPE_>
struct AttentionKernelTraits {

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;
  using DTypePVAccum = float;
  using NV_TYPE = NV_TYPE_;
  
  
  static constexpr bool USE_TMA_LOAD_KV = USE_TMA_LOAD_KV_;
  static constexpr int GROUP_SIZE = GROUP_SIZE_;
  static constexpr int BLOCK_SHAPE_Q = BLOCK_SHAPE_Q_;
  static_assert(BLOCK_SHAPE_Q % 64 == 0);
  static constexpr int BLOCK_SHAPE_KV = BLOCK_SHAPE_KV_;
  static constexpr int HEAD_DIM_QK = HEAD_DIM_QK_;
  static constexpr int HEAD_DIM_VO = HEAD_DIM_VO_;
  static constexpr int NUM_PER_STAGE = BLOCK_SHAPE_KV * HEAD_DIM_QK;
  static_assert(HEAD_DIM_QK % 32 == 0);
  static_assert(HEAD_DIM_VO % 32 == 0);

  static constexpr int NUM_WARPS = 12;
  static constexpr int NUM_THREADS = 384;
  static constexpr int NUM_PRODUCER_THREADS = 128;

  using TileShape_QKD = Shape<Int<BLOCK_SHAPE_Q>, Int<BLOCK_SHAPE_KV>, Int<HEAD_DIM_QK>>;
  using TileShape_PDV = Shape<Int<BLOCK_SHAPE_Q>, Int<HEAD_DIM_VO>, Int<BLOCK_SHAPE_KV>>;

  static constexpr int NUM_STAGES = NUM_STAGES_;

  using AtomLayoutQKD = Layout<Shape<Int<BLOCK_SHAPE_Q / 64>, _1, _1>>;
  using AtomLayoutPV = Layout<Shape<Int<BLOCK_SHAPE_Q / 64>, _2, _1>>;
  using TiledMmaQK = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<DTypeQ, DTypeKV, DTypeQKAccum, TileShape_QKD>(), AtomLayoutQKD{}));
  using TiledMmaPV = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/DTypePVAccum, TileShape_PDV,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      AtomLayoutPV{}));
  using TiledMmaPVSS = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<DTypeKV, DTypeKV, /*ElementAccum=*/DTypePVAccum, TileShape_PDV,
                                 GMMA::Major::K, GMMA::Major::MN>(),
      AtomLayoutPV{}));

  static constexpr int NUM_MMA_THREADS = size(TiledMmaPV{});

  using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeQ, decltype(cute::get<0>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_QKD{})));

  using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<1>(TileShape_QKD{})),
                                   decltype(cute::get<2>(TileShape_QKD{}))>());
  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtomK{},
      make_shape(shape<1>(TileShape_QKD{}), shape<2>(TileShape_QKD{}), Int<NUM_STAGES>{})));
  using SmemLayoutVt = decltype(composition(
      SmemLayoutK{}, make_ordered_layout(make_shape(get<2>(TileShape_QKD{}),
                                                    get<1>(TileShape_QKD{}), Int<NUM_STAGES>{}),
                                         Step<_2, _1, _3>{})));
  using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeKV, decltype(cute::get<2>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutV = decltype(tile_to_shape(
      SmemLayoutAtomV{},
      make_shape(get<2>(TileShape_PDV{}), get<1>(TileShape_PDV{}), Int<1>{})));

  // Note this is the transpose in terms of the view, not in terms of memory.
  using SmemLayoutVtOneStage = decltype(composition(
      SmemLayoutV{}, make_ordered_layout(make_shape(get<1>(TileShape_PDV{}),
                                                    get<2>(TileShape_PDV{}), Int<1>{}),
                                         Step<_2, _1, _3>{})));

  using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeO, decltype(cute::get<0>(TileShape_PDV{})),
                                   decltype(cute::get<1>(TileShape_PDV{}))>());
  using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 1>(TileShape_PDV{})));

  using SmemCopyAtom = Copy_Atom<cute::SM90_U32x4_STSM_N, DTypeQ>;
  
  static constexpr bool IS_CTA_32 = (BLOCK_SHAPE_KV == 32);
  using SmemLayoutRowOneStage = Layout<Shape<_2, Int<128>>, Stride<_1, _2>>;
  using SmemLayoutRowTwoStage = Layout<Shape<_2, Int<128>, _2>, Stride<_1, _2, _256>>;
  using SmemLayoutRow = std::conditional_t<IS_CTA_32, SmemLayoutRowTwoStage, SmemLayoutRowOneStage>;

  using SmemLayoutAtomP = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                   GMMA::Major::K, DTypeQ, decltype(cute::get<0>(TileShape_QKD{})),
                                   decltype(cute::get<1>(TileShape_QKD{}))>());
  using SmemLayoutPSSOneStage = decltype(tile_to_shape(SmemLayoutAtomP{}, select<0, 1>(TileShape_QKD{})));
  using SmemLayoutPSSTwoStage = decltype(tile_to_shape(SmemLayoutAtomP{}, make_shape(Int<BLOCK_SHAPE_Q>{}, Int<BLOCK_SHAPE_KV>{}, Int<2>{})));
  using SmemLayoutP = std::conditional_t<IS_CTA_32, SmemLayoutPSSTwoStage, SmemLayoutPSSOneStage>;

  using MainloopPipelineQ = typename cutlass::PipelineAsync<1>;
  using PipelineStateQ = typename cutlass::PipelineState<1>;
  using MainloopPipeline =
      std::conditional_t<USE_TMA_LOAD_KV, typename cutlass::PipelineTmaAsync<NUM_STAGES>,
                         typename cutlass::PipelineAsync<NUM_STAGES>>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

  using SharedStorage = SharedStorageQKVO<MainloopPipeline, MainloopPipelineQ, DTypeQ, DTypeKV, DTypeQKAccum, DTypeO, IdType, BLOCK_SHAPE_KV,
                                          SmemLayoutQ, SmemLayoutK, SmemLayoutP, SmemLayoutRow, SmemLayoutO>;
};

}  // namespace mla_attn

#endif
