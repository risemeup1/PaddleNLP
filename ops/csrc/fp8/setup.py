# MIT License
#
# Copyright (c) 2025 DeepSeek/DeepGEMM Authors
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

import os
import shutil
import subprocess

import setuptools
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

current_dir = os.path.dirname(os.path.realpath(__file__))
jit_include_dirs = ("deep_gemm/include/deep_gemm",)
third_party_include_dirs = (
    "../../../csrc/third_party/cutlass/include/cute",
    "../../../csrc/third_party/cutlass/include/cutlass",
)


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.make_jit_include_symlinks()

    @staticmethod
    def make_jit_include_symlinks():
        # Make symbolic links of third-party include directories
        for d in third_party_include_dirs:
            dirname = d.split("/")[-1]
            src_dir = f"{current_dir}/{d}"
            dst_dir = f"{current_dir}/deep_gemm/include/{dirname}"
            assert os.path.exists(src_dir)
            if os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)
            os.symlink(src_dir, dst_dir, target_is_directory=True)


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Then run the regular build
        build_py.run(self)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, "deep_gemm/include")
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split("/")[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


if __name__ == "__main__":
    # noinspection PyBroadException
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        revision = "+" + subprocess.check_output(cmd).decode("ascii").rstrip()
    except:
        revision = ""

    setuptools.setup(
        name="deep_gemm",
        version="1.0.0" + revision,
        packages=["deep_gemm", "deep_gemm/jit", "deep_gemm/jit_kernels"],
        package_data={
            "deep_gemm": [
                "include/deep_gemm/**/*",
                "include/cute/**/*",
                "include/cutlass/**/*",
            ]
        },
        cmdclass={
            "develop": PostDevelopCommand,
            "build_py": CustomBuildPy,
        },
    )
