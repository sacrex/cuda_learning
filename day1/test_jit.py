import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

cuda_model = load(name="add2",
                  sources=["add2.cpp", "add2_kernel.cu"],
                  verbose=True)

n = 16
a = torch.rand(n, device="cuda:0")
b = torch.rand(n, device="cuda:0")
c = torch.rand(n, device="cuda:0")
print(f"a: {a}, b: {b}")

def run_cuda():
    cuda_model.torch_launch_add2(c, a, b, n)
    return c

run_cuda()
torch.cuda.synchronize(device="cuda:0")
print("c: ", c)

print("troch.equal(a+b, c): ", torch.equal(a+b, c))