## CMAKE_INCLUDE_PATH 添加了python的头文件路径
## cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" -DCMAKE_INCLUDE_PATH=/usr/include/python3.10/ ..

cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# 修改为你自己的nvcc路径，或者删掉这行，如果能运行的话。
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(add2 LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# 修改为你自己的python路径，或者删掉这行，如果能运行的话。
if(DEFINED CMAKE_INCLUDE_PATH)
  include_directories(${CMAKE_INCLUDE_PATH})
endif()
# include_directories(include)

set(SRCS add2.cpp add2_kernel.cu)
add_library(add2 SHARED ${SRCS})

target_link_libraries(add2 "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")