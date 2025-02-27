# MIT License
#
# Copyright (c) 2025 PaddlePaddle Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
from typing import Tuple

import deep_gemm
import paddle
from deep_gemm import calc_diff, cell_div, get_col_major_tma_aligned_tensor
from paddle import Tensor


def per_token_cast_to_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    assert x.dim() == 2 and x.shape[1] % 128 == 0
    m, n = x.shape
    x_view = paddle.view(x, (m, -1, 128))
    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=2)
    x_amax = paddle.view(x_amax, (m, -1))
    x_amax = paddle.clip(x_amax, min=1e-4)

    scaled_x = x_view * (448.0 / x_amax.unsqueeze(2))
    scaled_x_converted = paddle.view(scaled_x.astype(paddle.float8_e4m3fn), (m, n))

    x_amax_scaled = paddle.view((x_amax / 448.0), (m, -1))

    result = (scaled_x_converted, x_amax_scaled)
    return result


def per_block_cast_to_fp8(x: Tensor) -> Tuple[Tensor, Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = paddle.zeros((cell_div(m, 128) * 128, cell_div(n, 128) * 128), dtype=x.dtype)
    x_padded[:m, :n] = x
    x_view = paddle.view(x_padded, (-1, 128, x_padded.shape[1] // 128, 128))

    x_abs = paddle.abs(x_view).astype(paddle.float32)
    x_amax = paddle.amax(x_abs, axis=(1, 3), keepdim=True)
    x_amax = paddle.clip(x_amax, min=1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).astype(paddle.float8_e4m3fn)

    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        paddle.view(x_amax / 448.0, (x_view.shape[0], x_view.shape[2]))
    )


def construct(m: int, k: int, n: int) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Tensor]:
    x = paddle.randn((m, k), dtype=paddle.bfloat16)
    y = paddle.randn((n, k), dtype=paddle.bfloat16)
    out = paddle.empty((m, n), dtype=paddle.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def test_gemm() -> None:
    print("Testing GEMM:")
    for m in (64, 128):
        for k, n in [(7168, 2112)]:
            x_fp8, y_fp8, out, ref_out = construct(m, k, n)
            deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            diff = calc_diff(out, ref_out)
            assert diff < 0.001, f"{m=}, {k=}, {n=}, {diff:.5f}"

    print()


if __name__ == "__main__":

    paddle.seed(0)
    random.seed(0)

    print("Library path:")
    print(f" > {deep_gemm.__path__}\n")
    test_gemm()
