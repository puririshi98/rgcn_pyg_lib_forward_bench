import torch
import torch_geometric
import time
aggr = torch_geometric.nn.aggr.MultiAggregation([torch_geometric.nn.aggr.SumAggregation()]*256, mode="attn", mode_kwargs={'in_channels':512, 'out_channels':256, 'num_heads':32}).cuda()
x = torch.randn(50000,512).cuda()
idx = torch.arange(50000).cuda()
for i in range(60):
  if i > 9:
    since=time.time()
  aggr(x, idx)
print("avg time per fwd pass=", (time.time()-since)/50.0)
