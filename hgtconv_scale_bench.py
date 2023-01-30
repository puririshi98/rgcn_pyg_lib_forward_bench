from torch_geometric.nn.conv import HGTConv
from torch_geometric.datasets import FakeHeteroDataset
import time
def bench(num_node_types=16, num_edge_types=128):
  data = FakeHeteroDataset(num_node_types=num_node_types, num_edge_types=num_edge_types).data.to('cuda')
  net = HGTConv(-1, 32, data.metadata()).to('cuda')
  x_dict = data.collect('x')
  edge_index_dict = data.collect('edge_index')
  for i in range(60):
    if i > 9:
      since = time.time()
    net(x_dict, edge_index_dict)
  out_time = float((time.time()-since)/50.0)
  print('fwd pass time for HGT w/ ({:} nodetypes, {:} edgetypes) = {:f}'.format(num_node_types, num_edge_types, out_time)
  return out_time

fwd_times = {}
for num_node_types in [2,4,8,16,32,64,128]:
  for num_edge_types in [4, 8, 16, 32, 64, 256]:
    fwd_times[(num_node_types, num_edge_types)] = bench(num_node_types, num_edge_types)
    
