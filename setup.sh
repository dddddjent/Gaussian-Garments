#!/usr/bin/env bash
# Template command, from the workspace root: bash Gaussian-Garments/setup.sh
# Creates a fresh dedicated environment; does not clone or modify mpmavatar.
set -eo pipefail
GAUGAR_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GAUGAR_BUILD="$GAUGAR_REPO/../data/build/gaugar"
eval "$(conda shell.bash hook)"
conda create -n gaugar python=3.11 pip -y
conda activate gaugar
conda install -c conda-forge -y \
  cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 \
  libcublas-dev=13.1 libcusparse-dev=12.6 libcusolver-dev=12.0 \
  gcc_linux-64=13 gxx_linux-64=13 ninja cmake ffmpeg 'colmap=4.2.0=cuda_130*'
# Reload compiler activation hooks installed above.
conda deactivate
conda activate gaugar
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST=12.0
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib"
export CUB_HOME="$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export CC="$CXX"
export MAX_JOBS=6
export FORCE_CUDA=1
conda env config vars set -n gaugar CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST=12.0
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r "$GAUGAR_REPO/requirements.txt"
mkdir -p "$GAUGAR_BUILD"
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "$GAUGAR_BUILD/pytorch3d"
git clone --depth 1 --recursive https://github.com/lizhe00/AnimatableGaussians.git "$GAUGAR_BUILD/AnimatableGaussians"
git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git "$GAUGAR_BUILD/simple-knn"
# Upstream rasterizer uses fixed-width integer types without including their header.
sed -i '/#include <iostream>/a #include <cstdint>' \
  "$GAUGAR_BUILD/AnimatableGaussians/gaussians/diff_gaussian_rasterization_depth_alpha/cuda_rasterizer/rasterizer_impl.h"
python -m pip install --no-build-isolation \
  "$GAUGAR_BUILD/pytorch3d" \
  "$GAUGAR_BUILD/simple-knn" \
  "$GAUGAR_REPO/scene/styleunet" \
  "$GAUGAR_BUILD/AnimatableGaussians/gaussians/diff_gaussian_rasterization_depth_alpha"
