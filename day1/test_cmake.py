import time
import numpy as np
import torch

torch.ops.load_library("build/libadd2.so")

n = 16
a = torch.rand(n, device="cuda:0")
b = torch.rand(n, device="cuda:0")
c = torch.rand(n, device="cuda:0")
print("a: ", a, "\nb:", b)

def run_cuda():
    torch.ops.add2.torch_launch_add2(c, a, b, n)
    return c

run_cuda()
torch.cuda.synchronize(device="cuda:0")
print("c: ", c)

print("troch.equal(a+b, c): ", torch.equal(a+b, c))
