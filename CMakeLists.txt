cmake_minimum_required(VERSION 3.15)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(starrygl VERSION 0.1 LANGUAGES CXX CUDA)

option(WITH_PYTHON "Link to Python when building" ON)
option(WITH_CUDA "Link to CUDA when building" ON)

# ==================== C++ / CUDA 标准 ====================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ==================== 关键：指定 GPU 架构 ====================
# A40 是 sm_80，禁用 FP8
set(CMAKE_CUDA_ARCHITECTURES 80)  # 仅生成 sm_80 代码
# 或者更明确：
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

# 可选：显式禁用 FP8（保险）
add_compile_definitions(__CUDA_NO_FP8_CONVERSIONS__)

# ==================== 依赖 ====================
find_package(OpenMP REQUIRED)
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

# NCCL（从第三方路径）
find_library(NCCL_LIBRARIES nccl PATHS "./third-party/nccl/build/lib" REQUIRED)
include_directories("./third-party/nccl/build/include")

if(WITH_PYTHON)
    add_definitions(-DWITH_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

if(WITH_CUDA)
    add_definitions(-DWITH_CUDA)
    add_definitions(-DWITH_UVM)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_RUNTIME_LIB "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
endif()

# ==================== 包含路径 ====================
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc/include/starrygl")
include_directories(${INCLUDE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})

# ==================== 源文件 ====================
file(GLOB_RECURSE CPP_SRCS "csrc/srcs/starrygl/*.cpp")
file(GLOB_RECURSE CUDA_SRCS "csrc/srcs/starrygl/*.cu")

message(STATUS "CPP sources: ${CPP_SRCS}")
message(STATUS "CUDA sources: ${CUDA_SRCS}")

# ==================== 构建目标 ====================
add_library("${PROJECT_NAME}_sampler" SHARED ${CPP_SRCS})
add_library("${PROJECT_NAME}_comm" SHARED ${CUDA_SRCS})

# 包含目录
target_include_directories(${PROJECT_NAME}_sampler PRIVATE ${INCLUDE_DIR})
target_include_directories(${PROJECT_NAME}_comm PRIVATE ${INCLUDE_DIR})

# 编译选项
target_compile_options(${PROJECT_NAME}_sampler PRIVATE -O3 -DTORCH_EXTENSION_NAME=lib${PROJECT_NAME}_sampler)
target_compile_options(${PROJECT_NAME}_comm PRIVATE -O3 -DTORCH_EXTENSION_NAME=lib${PROJECT_NAME}_comm)

# CUDA 特定选项（关键！）
if(WITH_CUDA)
    set_target_properties(${PROJECT_NAME}_comm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_options(${PROJECT_NAME}_comm PRIVATE
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
        -lineinfo
    )
endif()

# ==================== 链接 ====================
target_link_libraries(${PROJECT_NAME}_sampler
    PRIVATE
        OpenMP::OpenMP_CXX
        Threads::Threads
        ${TORCH_LIBRARIES}
)

target_link_libraries(${PROJECT_NAME}_comm
    PRIVATE
        OpenMP::OpenMP_CXX
        Threads::Threads
        CUDA::cudart
        ${TORCH_LIBRARIES}
        ${NCCL_LIBRARIES}
)

# Python 支持
if(WITH_PYTHON)
    target_compile_definitions(${PROJECT_NAME}_sampler PRIVATE -DWITH_PYTHON)
    target_compile_definitions(${PROJECT_NAME}_comm PRIVATE -DWITH_PYTHON)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
    target_link_libraries(${PROJECT_NAME}_sampler PRIVATE ${TORCH_PYTHON_LIBRARY})
    target_link_libraries(${PROJECT_NAME}_comm PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

# CUDA 宏
if(WITH_CUDA)
    target_compile_definitions(${PROJECT_NAME}_sampler PRIVATE -DWITH_CUDA -DWITH_UVM)
    target_compile_definitions(${PROJECT_NAME}_comm PRIVATE -DWITH_CUDA -DWITH_UVM)
endif()

# ==================== 输出 ====================
message(STATUS "Building ${PROJECT_NAME}_sampler and ${PROJECT_NAME}_comm for A40 (sm_80)")