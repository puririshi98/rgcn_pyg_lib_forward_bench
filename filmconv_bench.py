from torch_geometric.nn.conv import FastFiLMConv, FiLMConv
from torch_geometric.datasets import FakeHeteroDataset
import torch
import time
times = {}
fast_times = {}
try:
    for n_per_type in [100,1000,10000, 100000]:
        for n_types in [2,4,8,16,32,64,128,256]:
          data = FakeHeteroDataset(num_node_types=n_types, num_edge_types=n_types, avg_num_nodes=n_per_type).data
          max_dim = -1
          for n_type in data.node_types:
            max_dim = max(max_dim, data[n_type].x.size(-1))
          for n_type in data.node_types:
            if data[n_type].x.size(-1) < max_dim:
              data[n_type].x = torch.cat((data[n_type].x, torch.zeros((data[n_type].x.size(0),max_dim - data[n_type].x.size(-1)))), dim=-1)
          data = data.to_homogeneous().to('cuda')                           
          net = FiLMConv(in_channels=data.x.size(-1), out_channels=int(data.x.size(-1)/2), num_relations=n_types).cuda()
          fastnet = FastFiLMConv(in_channels=data.x.size(-1), out_channels=int(data.x.size(-1)/2), num_relations=n_types, is_sorted=True).cuda()
          for i in range(60):
            if i > 9:
              since = time.time()
            net(data.x, data.edge_index, data.edge_type)
          times[(n_per_type, n_types)] = (time.time()-since)/50.0                               
          print("average vanilla fwd pass time:", times[(n_per_type, n_types)])
          for i in range(60):
            if i > 9:
              since = time.time()
            fastnet(data.x, data.edge_index, data.edge_type)
          fast_times[(n_per_type, n_types)] = (time.time()-since)/50.0                               
          print("average fast fwd pass time:", fast_times[(n_per_type, n_types)])
except:
    print("times=", times)
    print("fast_times=", fast_times)
print("times=", times)
print("fast_times=", fast_times)
