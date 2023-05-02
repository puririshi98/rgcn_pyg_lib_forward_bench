import torch
from torch_geometric.nn.dense import Linear, HeteroLinear
import time
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
fused_times = {}
loop_times = {}
try:
  for num_nodes_per_type in [10**2,10**3,10**4,10**5]:
    for out_feats in [2, 4,8,16,32,64,128,256]:
      for n_feats in [4,8,16,32,64,128,256,512]:
        for num_types in [4, 8, 16, 32, 64, 128, 256, 512]:
          try:
            print("benchmarking", num_types,"types w/", num_nodes_per_type, "nodes per type and", n_feats, "input features and", out_feats, "outuput feats")
            x_dict = {'v'+str(i):torch.randn((num_nodes_per_type, n_feats)).cuda() for i in range(num_types)}
            x = torch.cat(list(x_dict.values()), dim=0)
            node_type = torch.cat([(j * torch.ones(x_j.shape[0])).long()
                                   for j, x_j in enumerate(x_dict.values())]).cuda()
            lin = Linear(n_feats, out_feats).cuda()
            heterolin = HeteroLinear(n_feats, out_feats, len(list(x_dict.keys())), True).cuda()
            for i in range(60):
                if i==10:
                    since=time.time()
                heterolin(x=x, type_vec=node_type)
            key = (num_types, num_nodes_per_type, n_feats, out_feats)
            fused_times[key] = ((time.time()-since)/50.0)
            print("Avg time for fuse based=", fused_times[key])
            for i in range(60):
                if i==10:
                    since=time.time()
                o = x.new_empty(x.size(0), 64)
                for j in range(num_types):
                    mask = j==node_type
                    o[mask] = lin(x[mask])
            loop_times[key] = ((time.time()-since)/50.0)
            print("Avg time for for-loop=", loop_times[key])
          except:
            continue
except:
  print("Loop Times:", loop_times)
  print("Fused Times:", fused_times)




print("Loop Times:", loop_times)
print("Fused Times:", fused_times)
