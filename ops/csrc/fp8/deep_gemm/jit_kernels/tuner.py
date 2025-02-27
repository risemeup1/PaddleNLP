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

import copy
import os

# from functools import cached_property
from typing import Any, Dict

import paddle

from ..jit import Runtime, build, cpp_format, generate

# class CacheKey:
#     def __init__(self, keys: Dict[str, Any]) -> None:
#         self.keys = keys

#     @cached_property
#     def get_signature(self) -> str:
#         return repr(self.keys)

#     def


class JITTuner:
    def __init__(self) -> None:
        self.tuned = {}

    def compile_and_tune(
        self,
        name: str,
        keys: Dict[str, Any],
        space: tuple,
        includes: tuple,
        arg_defs: tuple,
        template: str,
        args: tuple,
    ) -> Runtime:
        # NOTES: we always assume the space and template will not change
        # We also assume the GPU device will not be changed
        # NOTES: the function must have no accumulated side effects
        keys = {k: keys[k] for k in sorted(keys.keys())}
        signature = (name, f"{keys}")
        if signature in self.tuned:
            return self.tuned[signature]

        if os.getenv("DG_JIT_DEBUG", None):
            print(f"Auto-tuning JIT kernel {name} with keys {keys}")
        assert args is not None
        space = (dict(),) if len(space) == 0 else space

        kernels = []
        for tuned_keys in space:
            assert isinstance(tuned_keys, dict)
            full_keys = copy.deepcopy(keys)
            full_keys.update(tuned_keys)
            code = generate(includes, arg_defs, cpp_format(template, full_keys))

            # Illegal build must raise errors
            kernels.append((build(name, arg_defs, code), tuned_keys))

        best_runtime, best_time, best_keys = None, None, None
        for runtime, tuned_keys in kernels:
            if len(space) > 1:
                # Check kernel validity
                return_code = runtime(*args)
                if return_code != 0:
                    # Pass illegal kernels, e.g. insufficient shared memory capacity
                    if os.getenv("DG_JIT_DEBUG", None):
                        print(
                            f"Illegal JIT kernel {name} with keys {keys} and tuned keys {tuned_keys}: error code {return_code}"
                        )
                    continue

                # Measure performance with L2 flush and a large GEMM kernel before to reduce overhead between kernels
                start_event = paddle.device.cuda.Event(enable_timing=True)
                end_event = paddle.device.cuda.Event(enable_timing=True)
                paddle.empty(int(256e6 // 4), dtype=paddle.int32).zero_()
                paddle.randn((8192, 8192), dtype=paddle.float32) @ paddle.randn((8192, 8192), dtype=paddle.float32)
                start_event.record()
                for i in range(20):
                    assert runtime(*args) == 0
                end_event.record()
                end_event.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            else:
                elapsed_time = 0

            # Compare if better
            if best_time is None or elapsed_time < best_time:
                best_runtime, best_time, best_keys = runtime, elapsed_time, tuned_keys
            if os.getenv("DG_JIT_DEBUG", None):
                print(f"Tuned JIT kernel {name} with keys {keys} and tuned keys {tuned_keys} has time {elapsed_time}")
        assert best_runtime is not None, f"Failed to tune JIT kernel {name} with keys {keys}"

        # Cache the best runtime and return
        if os.getenv("DG_JIT_DEBUG", None) or os.getenv("DG_PRINT_AUTOTUNE", None):
            print(f"Best JIT kernel {name} with keys {keys} has tuned keys {best_keys} and time {best_time}")
        self.tuned[signature] = best_runtime
        return best_runtime


jit_tuner = JITTuner()
