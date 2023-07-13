import torch_geometric
import torch
# scale up to see when it breaks
for i in range(2, 10):
  print("trying w/ num nodes = 10^" + str(i))
  max_index = 10**i
  node_idx = torch.zeros(max_index, dtype=torch.long,
                       device='cuda')
