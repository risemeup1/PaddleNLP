# MIT License
#
# Copyright (c) 2025 DeepSeek-Ai/DeepGEMM
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

from . import jit
from .jit_kernels import (
    ceil_div,
    gemm_fp8_fp8_bf16_nt,
    get_col_major_tma_aligned_tensor,
    get_m_alignment_for_contiguous_layout,
    get_num_sms,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    set_num_sms,
)
from .utils import bench, calc_diff, get_cuda_home
