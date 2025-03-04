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

import ctypes
import os
from typing import Optional

import paddle
from paddle import Tensor

from .template import map_ctype


class Runtime:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lib = None
        self.args = None

        assert self.is_path_valid(self.path)

    @staticmethod
    def is_path_valid(path: str) -> bool:
        # Exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            return False

        # Contains all necessary files
        files = ["kernel.cu", "kernel.args", "kernel.so"]
        return all(os.path.exists(os.path.join(path, file)) for file in files)

    def __call__(self, *args) -> int:
        # Load SO file
        if self.lib is None or self.args is None:
            self.lib = ctypes.CDLL(os.path.join(self.path, "kernel.so"))
            with open(os.path.join(self.path, "kernel.args"), "r") as f:
                self.args = eval(f.read())

        # Check args and launch
        assert len(args) == len(self.args), f"Expected {len(self.args)} arguments, got {len(args)}"
        cargs = []
        for arg, (name, dtype) in zip(args, self.args):
            if isinstance(arg, Tensor):
                assert arg.dtype == dtype, f"Expected tensor dtype `{dtype}` for `{name}`, got `{arg.dtype}`"
            else:
                assert isinstance(arg, dtype), f"Expected built-in type `{dtype}` for `{name}`, got `{type(arg)}`"
            cargs.append(map_ctype(arg))

        return_code = ctypes.c_int(0)
        self.lib.launch(*cargs, ctypes.byref(return_code))
        return return_code.value


class RuntimeCache:
    def __init__(self) -> None:
        self.cache = {}

    def __getitem__(self, path: str) -> Optional[Runtime]:
        # In Python runtime
        if path in self.cache:
            return self.cache[path]

        # Already compiled
        if os.path.exists(path) and Runtime.is_path_valid(path):
            runtime = Runtime(path)
            self.cache[path] = runtime
            return runtime
        return None

    def __setitem__(self, path, runtime) -> None:
        self.cache[path] = runtime
