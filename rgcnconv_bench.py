from torch_geometric.nn.conv import RGCNConv
from torch_geometric.data import HeteroData
import torch
import time
data = HeteroData()
data['n1'] = torch.randn(5000, 64)
data['n2'] = torch.randn(5000, 64)
data['n3'] = torch.randn(5000, 64)
data[('n1','e1','n2')] = torch.randint(5000, size=(2,10000))
data[('n2','e2','n3')] = torch.randint(5000, size=(2,10000))
data[('n1','e3','n3')] = torch.randint(5000, size=(2,10000))
data[('n2','e1_rev','n1')] = torch.randint(5000, size=(2,10000))
data[('n3','e2_rev','n2')] = torch.randint(5000, size=(2,10000))
data[('n3','e3_rev','n1')] = torch.randint(5000, size=(2,10000))

net = RGCNConv(in_channels=64, out_channels=32, num_relations=len(data.edge_types))
data = data.to('cuda').to_homogeneous()
for i in range(60):
  if i > 9:
    since = time.time()
  net(data.x, data.edge_index, data.edge_type)
print("average fwd pass time:", (time.time()-since)/50.0)
