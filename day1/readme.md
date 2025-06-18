# 3种调用方式
## https://zhuanlan.zhihu.com/p/645330027
## https://godweiyang.com/2021/03/21/torch-cpp-cuda-2/

## 1. 使用torch.utils.cpp_extension.load方式, JIT编译,动态生成model对象.
查看 test_jit.py中的算子调用方式

在add2.cpp中的PYBIND11_MODULE宏

## 2. 使用setuptools
查看 test_load.py中的算子调用方式

编写setup.py文件, 查看add2.cpp中的PYBIND11_MODULE.

安装:
python3 setup.py install

使用时:
```
import add2
```

## 3. 使用CMake
查看 test_cmake.py中算子调用方式

编写CMakeLists.txt文件
查看add2.cpp中的TORCH_LIBRARY

构建过程:
```
mkdir build
cd build
# CMAKE_INCLUDE_PATH设置了python的路径
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" -DCMAKE_INCLUDE_PATH=/usr/include/python3.10/ ..
make
```
