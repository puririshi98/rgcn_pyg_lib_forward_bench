import torch
import torch_geometric
import time
aggr = torch_geometric.nn.aggr.MultiAggregation(["sum","mean","min","max"], "attn")
x = torch.randn(50,32)
idx = torch.randint(0,50,size=(2, 100))
for i in range(60):
  if i > 9:
    since=time.time()
  aggr(x, idx)
print("avg time per fwd pass=", (time.time()-since)/50.0)
