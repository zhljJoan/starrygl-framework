#!/bin/bash

mkdir -p build && cd build
cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -DPython3_ROOT_DIR=$(python -c "import sys; print(sys.prefix)") \
    -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME:-"$(realpath $(dirname $(which nvcc))/../)"} \
&& make -j32 \
&& rm -rf ../starrygl/lib \
&& mkdir ../starrygl/lib \
&& cp lib*.so ../starrygl/lib/ \
&& cp third_party/ldg_partition/lib*.so ../starrygl/lib/ \
&& patchelf --set-rpath '$ORIGIN:$ORIGIN/lib' --force-rpath ../starrygl/lib/*.so
