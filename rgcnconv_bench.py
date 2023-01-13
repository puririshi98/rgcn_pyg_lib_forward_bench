from torch_geometric.nn.conv import RGCNConv
from torch_geometric.data import HeteroData
import time
data = HeteroData
net = RGCNConv(in_channels=64, out_channels=32, num_relations=2)
x_dict = data.collect('x')
edge_index_dict = data.collect('edge_index')
for i in range(60):
  if i > 9:
    since = time.time()
  net(edge_index_dict)
print("average fwd pass time:", (time.time()-since)/50.0)
