cmake_minimum_required(VERSION 3.14)
project(StarryGL_Processor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native")

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(StarryGL_Processor src/main.cpp)
target_link_libraries(StarryGL_Processor "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)