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
import copy
import ctypes
import os
from typing import Any, Dict, Iterable, Tuple

import paddle
from paddle import Tensor

# Name map for Python `eval`
typename_map: Dict[Any, str] = {
    **{t: t.__name__ for t in (bool, int, float)},
    paddle.int32: "paddle.int32",
    paddle.float32: "paddle.float32",
    paddle.bfloat16: "paddle.bfloat16",
    paddle.float8_e4m3fn: "paddle.float8_e4m3fn",
    paddle.device.cuda.Stream: "paddle.device.cuda.Stream",
}
# `ctype` map for Python casting
ctype_map: Dict[Any, Any] = {
    **{t: getattr(ctypes, f"c_{t.__name__}") for t in (bool, int, float)},
    **{
        t: ctypes.c_void_p
        for t in (paddle.int32, paddle.float32, paddle.bfloat16, paddle.float8_e4m3fn, paddle.device.cuda.Stream)
    },
}


# Type map for both Python API and source code usages
genc_map = {
    bool: ("bool", "bool"),
    int: ("int", "int"),
    float: ("float", "float"),
    paddle.int32: ("void*", "int*"),
    paddle.float32: ("void*", "float*"),
    paddle.bfloat16: ("void*", "__nv_bfloat16*"),
    paddle.float8_e4m3fn: ("void*", "__nv_fp8_e4m3*"),
    paddle.device.cuda.Stream: ("void*", "cudaStream_t"),
}


def map_ctype(value: Any) -> Any:
    ctype = ctype_map[value.dtype if isinstance(value, Tensor) else type(value)]
    if isinstance(value, Tensor):
        return ctype(value.data_ptr())
    if isinstance(value, paddle.device.cuda.Stream):
        return ctype(value.cuda_stream)
    return ctype(value)


def cpp_format(template: str, keys: Dict[str, Any]) -> str:
    # We don't use `str.format` because it's not safe for C++ {} braces
    new_template = copy.deepcopy(template)
    for key, value in keys.items():
        new_template = new_template.replace(f"{{{key}}}", f"{value}")
    return new_template


def generate(includes: Iterable[str], arg_defs: Iterable[Tuple], body: str) -> str:
    # Common prefix
    code = "// DeepGEMM auto-generated JIT CUDA source file\n\n"

    # Includes
    preload_sys_includes = ["<cuda.h>", "<cuda_fp8.h>", "<cuda_runtime.h>", "<iostream>"]
    preload_package_includes = ['"cutlass/cutlass.h"']

    assert isinstance(includes, list) or isinstance(includes, tuple)
    sys_includes = sorted(
        list(set(preload_sys_includes + [include for include in includes if include.startswith("<")]))
    )
    package_includes = sorted(
        list(set(preload_package_includes + [include for include in includes if include.startswith('"')]))
    )
    code += "\n".join(f"#include {include}" for include in sys_includes) + "\n\n"
    code += "\n".join(f"#include {include}" for include in package_includes) + "\n\n"

    # Function signature
    raw = "__raw_"
    get_def = lambda n, t: f"{genc_map[t][0]} " + (raw if genc_map[t][0] != genc_map[t][1] else "") + n
    code += f'extern "C" void launch('
    code += ", ".join(
        [get_def(*arg_def) for arg_def in arg_defs]
        + [
            "int& __return_code",
        ]
    )
    code += ") {\n"

    # Cast raw types
    code += "    // Cast raw types (if needed)\n"
    for arg_name, arg_type in arg_defs:
        if genc_map[arg_type][0] != genc_map[arg_type][1]:
            code += f"    auto {arg_name} = reinterpret_cast<{genc_map[arg_type][1]}>({raw}{arg_name});\n"

    # Function body
    code += "\n".join([(("    " if line else "") + line) for line in body.split("\n")])

    # End the function
    code += "}\n\n"

    # Debug print
    if os.getenv("DG_JIT_DEBUG", None):
        print(f"Generated code:\n{code}")

    return code
