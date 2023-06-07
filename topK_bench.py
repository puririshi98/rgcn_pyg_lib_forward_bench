from torch_geometric.nn.pool.select import topk
from torch_geometric.datasets import FakeHeteroDataset
import torch
import time
times = {}
fast_times = {}
try:
    for n_per_type in [100,1000,10000, 100000]:
        for n_types in [2,4,8,16,32,64,128,256]:
          try:
              data = FakeHeteroDataset(num_node_types=n_types, num_edge_types=n_types, avg_num_nodes=n_per_type).data
              data = data.to_homogeneous().to('cuda')
              x = data.x
              batch = x.new_zeros(x.size(0), dtype=torch.long)
              for i in range(60):
                if i > 9:
                  since = time.time()
                topk(x, batch)
              times[(n_per_type, n_types)] = (time.time()-since)/50.0                               
              print("average topK fwd pass time:", times[(n_per_type, n_types)])
          except:
              continue
    reprint=True
except:
    print("times=", times)
    reprint=False
if reprint:
  print("times=", times)
