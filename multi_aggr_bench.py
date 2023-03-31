import torch
import torch_geometric
import time
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

aggr = torch_geometric.nn.aggr.MultiAggregation([torch_geometric.nn.aggr.SumAggregation()]*128, mode="attn", mode_kwargs={'in_channels':128, 'out_channels':64, 'num_heads':8}).cuda()
x = torch.randn(500,128).cuda()
idx = torch.arange(500).cuda()
for i in range(60):
  if i > 9:
    since=time.time()
  aggr(x, idx)
print("avg time per fwd pass=", (time.time()-since)/50.0)
