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

import os
import shutil
import subprocess

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

current_dir = os.path.dirname(os.path.realpath(__file__))
jit_include_dirs = ("deep_gemm/include/deep_gemm",)
third_party_include_dirs = (
    "../../../csrc/third_party/cutlass/include/cute",
    "../../../csrc/third_party/cutlass/include/cutlass",
)


def create_symlink(source, destination):
    if os.path.exists(destination):
        if os.path.islink(destination):
            os.unlink(destination)
    os.symlink(source, destination, target_is_directory=True)


def copy_directory(source, destination):
    if os.path.exists(destination):
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


class PostDevelopCommand(develop):
    def run(self):
        super().run()
        self.make_jit_include_symlinks()

    @staticmethod
    def make_jit_include_symlinks():
        for dir_path in third_party_include_dirs:
            dirname = os.path.basename(dir_path)
            src_dir = os.path.join(current_dir, dir_path)
            dst_dir = os.path.join(current_dir, "deep_gemm", "include", dirname)
            assert os.path.exists(src_dir)
            create_symlink(src_dir, dst_dir)


class PostInstallCommand(install):
    def run(self):
        super().run()
        self.copy_jit_includes()

    def copy_jit_includes(self):
        include_dir = os.path.join(self.build_lib, "deep_gemm", "include")
        os.makedirs(include_dir, exist_ok=True)
        for dir_path in jit_include_dirs + third_party_include_dirs:
            src_dir = os.path.join(current_dir, dir_path)
            dst_dir = os.path.join(include_dir, os.path.basename(dir_path))
            assert os.path.exists(src_dir)
            copy_directory(src_dir, dst_dir)


def get_git_revision():
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        return "+" + subprocess.check_output(cmd).decode("ascii").strip()
    except subprocess.CalledProcessError:
        return ""


if __name__ == "__main__":
    revision = get_git_revision()

    setuptools.setup(
        name="deep_gemm",
        version=f"1.0.0{revision}",
        packages=["deep_gemm", "deep_gemm/jit", "deep_gemm/jit_kernels"],
        cmdclass={"develop": PostDevelopCommand, "install": PostInstallCommand},
        license="MIT",
    )
