import torch
from torch_geometric.nn import to_hetero
from torch_geometric.nn.dense import Linear
import time
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
dict_times = []
fused_times = []
loop_times = []
num_nodes_per_type = 10000
n_feats = 128
out_feats = 64


for num_types in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    x_dict = {'v'+str(i):torch.randn((num_nodes_per_type, n_feats)).cuda() for i in range(num_types)}
    x = torch.cat(list(x_dict.values()), dim=0)
    node_type = torch.cat([(j * torch.ones(x_j.shape[0])).long()
                           for j, x_j in enumerate(x_dict.values())])
    lin = Linear(n_feats, out_feats).cuda()
    heterolin = to_hetero(lin,(x_dict.keys(), [('v1','e1','v1'])).cuda()
    for i in range(60):
        if i==10:
            since=time.time()
        heterolin(x=x, type_vec=node_type)
    fused_times.append((time.time()-since)/50.0)
    print("Avg time for fuse based", num_types, '=', fused_times[-1])
    for i in range(60):
        if i==10:
            since=time.time()
        heterolin(x_dict)
    dict_times.append((time.time()-since)/50.0)
    print("Avg time for dict based", num_types, '=', dict_times[-1])
    for i in range(60):
        if i==10:
            since=time.time()
        for i in range(num_types):
            lin(x_dict['v'+str(i)])
    loop_times.append((time.time()-since)/50.0)
    print("Avg time for for-loop", num_types, '=', loop_times[-1])




print("Loop Times:", loop_times)
print("Dict Times:", dict_times)
print("Fused Times:", fused_times)

