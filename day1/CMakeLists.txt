cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# 修改为你自己的nvcc路径，或者删掉这行，如果能运行的话。
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(add2 LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# 修改为你自己的python路径，或者删掉这行，如果能运行的话。
# include_directories(/usr/include/python3.7)
# include_directories(include)

set(SRCS add2.cpp add2_kernel.cu)
add_library(add2 SHARED ${SRCS})

target_link_libraries(add2 "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")