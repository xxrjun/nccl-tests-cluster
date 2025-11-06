#!/bin/bash


# YOU MAY NEED TO CHANGE THE PATH BELOW
export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda"}
export NCCL_HOME="${NCCL_HOME:-$HOME/nccl-tests-cluster/nccl/build}"
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export PATH=/usr/local/cuda/bin:$PATH

git clone https://github.com/NVIDIA/nccl.git || true
if [ ! -d nccl ]; then
    echo "Error: NCCL repository not found!"
    exit 1
fi
cd nccl || exit 1
git clone https://github.com/NVIDIA/nccl-tests.git || true


echo "====================================="
echo "Build NCCL"
echo "====================================="

# You can also check the GPU architecture via https://developer.nvidia.com/cuda-gpus
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/\.//g') # FIXME: login node might not have GPU
echo "Detected GPU Architecture: $ARCH"
make -j32 src.build NVCC_GENCODE="-gencode=arch=compute_${ARCH},code=sm_${ARCH}"

# Install tools to create debian packages (sudo permission required)
# sudo apt install build-essential devscripts debhelper fakeroot
# Build NCCL deb package
# make pkg.debian.build
# ls build/pkg/deb/

echo "====================================="
echo "Build NCCL Tests"
echo "====================================="

if [ ! -d nccl-tests ]; then
    echo "Error: NCCL Tests repository not found!"
    exit 1
fi
cd nccl-tests || exit 1

# If This assume CUDA is installed in /usr/local/cuda, otherwise, set CUDA_HOME accordingly
# make MPI=1 NCCL_HOME=$NCCL_HOME
make MPI=1 NCCL_HOME=$NCCL_HOME CUDA_HOME=$CUDA_HOME
# make CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl

cd ../.. || exit 1