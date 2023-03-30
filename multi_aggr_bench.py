import torch
import torch_geometric
import time
aggr = torch_geometric.nn.aggr.MultiAggregation([torch_geometric.nn.aggr.SumAggregation(),torch_geometric.nn.aggr.MeanAggregation(),torch_geometric.nn.aggr.MinAggregation(),torch_geometric.nn.aggr.MaxAggregation()], mode="attn", mode_kwargs={'in_channels':32, 'out_channels':16})
x = torch.randn(50,32)
idx = torch.randint(0,50,size=(2, 100))
for i in range(60):
  if i > 9:
    since=time.time()
  aggr(x, idx)
print("avg time per fwd pass=", (time.time()-since)/50.0)
