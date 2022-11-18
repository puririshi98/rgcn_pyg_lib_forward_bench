import torch
from torch_geometric.nn import ToHeteroModule
from torch_geometric.nn.dense import Linear
import time
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
dict_times = []
fused_times = []
num_nodes_per_type = 10000
n_feats = 128
out_feats = 64


for num_types in [4, 8, 16, 32, 64, 128, 256, 512]:
    x_dict = {'v'+str(i):torch.randn((num_nodes_per_type, n_feats)).cuda() for i in range(num_types)}
    metadata= (x_dict.keys(), [])
    x = torch.cat(list(x_dict.values()), dim=0)
    node_type = torch.cat([(j * torch.ones(x_j.shape[0])).long()
                           for j, x_j in enumerate(x_dict.values())])
    heterolin = ToHeteroModule(Linear(n_feats, out_feats), metadata).cuda()
    for i in range(60):
        if i==10:
            since=time.time()
        heterolin(x=x, node_type=node_type)
    fused_times.append((time.time()-since)/50.0)
    print("Avg time for fused ", num_types, '=', fused_times[-1])
    for i in range(60):
        if i==10:
            since=time.time()
        heterolin(x_dict)
    dict_times.append((time.time()-since)/50.0)
    print("Avg time for dict based", num_types, '=', dict_times[-1])




print("Dict Times:", dict_times)
print("Fused Times:", fused_times)

