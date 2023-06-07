from torch_geometric.nn.pool.select import topk
from torch_geometric.datasets import FakeDataset
import torch
import time
times = {}
fast_times = {}
try:
    for num_nodes in [100,1000,10000, 100000]:
        for num_feats in [2,4,8,16,32,64,128,256]:
          try:
              x = torch.randn((num_nodes, num_feats), dtype=torch.float, device='cuda')
              batch = x.new_zeros(x.size(0), dtype=torch.long)
              for i in range(60):
                if i > 9:
                  since = time.time()
                topk(x, batch)
              times[(num_nodes, num_feats)] = (time.time()-since)/50.0 
              print("For", num_nodes, "nodes and", num_feats, "feats:")
              print("average topK fwd pass time:", times[(n_per_type, n_types)])
          except:
              continue
    reprint=True
except Exception as e:
    print("failed w/:", e)
    print("times=", times)
    reprint=False
if reprint:
  print("times=", times)
