import torch
import torch_geometric
import time
aggr = torch_geometric.nn.aggr.MultiAggregation([torch_geometric.nn.aggr.SumAggregation(),torch_geometric.nn.aggr.MeanAggregation(),torch_geometric.nn.aggr.MinAggregation(),torch_geometric.nn.aggr.MaxAggregation()], mode="attn", mode_kwargs={'in_channels':64, 'out_channels':32, 'num_heads':1024})
x = torch.randn(50000,64)
idx = torch.arange(50000)
for i in range(60):
  if i > 9:
    since=time.time()
  aggr(x, idx)
print("avg time per fwd pass=", (time.time()-since)/50.0)
