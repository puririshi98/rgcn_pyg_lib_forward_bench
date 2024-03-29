from torch_geometric.nn.conv import HGTConv
from torch_geometric.datasets import FakeHeteroDataset
import time
data = FakeHeteroDataset(num_node_types=16, num_edge_types=128).data.to('cuda')
net = HGTConv(-1, 32, data.metadata()).to('cuda')
x_dict = data.collect('x')
edge_index_dict = data.collect('edge_index')
for i in range(60):
  if i > 9:
    since = time.time()
  net(x_dict, edge_index_dict)
print("average fwd pass time:", (time.time()-since)/50.0)
  
