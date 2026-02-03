#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -DPython3_ROOT_DIR=$(python -c "import sys; print(sys.prefix)") \
    -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME:-"$(realpath $(dirname $(which nvcc))/../)"} \

# 编译
make -j$(nproc)

# 运行
echo "Starting Preprocessing..."
./StarryGL_Processor