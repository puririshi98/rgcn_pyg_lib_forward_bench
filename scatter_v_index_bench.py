import torch
from torch import Tensor
import time
def broadcast(src: Tensor, ref: Tensor, dim: int) -> Tensor:
    size = ((1, ) * dim) + (-1, ) + ((1, ) * (ref.dim() - dim - 1))
    return src.view(size).expand_as(ref)
dim = 0
src = torch.randn(size=(100000,128)).to('cuda')
index = torch.randint(high=src.size(0), size=(100000,)).to('cuda')
dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
size = src.size()[:dim] + (dim_size, ) + src.size()[dim + 1:]

def og_func(index, src, dim):
  index = broadcast(index, src, dim)
  return src.new_zeros(size).scatter_add_(dim, index, src)

def new_func(index, src, dim):
  return src.new_zeros(size).index_add_(dim, index, src)

for i in range(60):
  if i > 9:
    since = time.time()
  og_func(index, src, dim)
  torch.cuda.synchronize()
print("original implementation takes", (time.time()-since)/50, "s/iter")
for i in range(60):
  if i > 9:
    since = time.time()
  new_func(index, src, dim)
  torch.cuda.synchronize()
print("new implementation takes", (time.time()-since)/50, "s/iter")

  
  
