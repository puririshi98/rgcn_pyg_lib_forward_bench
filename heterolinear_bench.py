import torch
from torch_geometric.nn.dense import Linear, HeteroLinear
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
                           for j, x_j in enumerate(x_dict.values())]).cuda()
    lin = Linear(n_feats, out_feats).cuda()
    heterolin = HeteroLinear(n_feats, out_feats, len(list(x_dict.keys())), True).cuda()
    for i in range(60):
        if i==10:
            try:
                print("heterolin.use_segmm=", heterolin.use_segmm)
            except:
                pass
            since=time.time()
        heterolin(x=x, type_vec=node_type)
    fused_times.append((time.time()-since)/50.0)
    print("Avg time for fuse based", num_types, '=', fused_times[-1])
    for i in range(60):
        if i==10:
            since=time.time()
        o = x.new_empty(x.size(0), 64)
        for j in range(num_types):
            mask = j==node_type
            o[mask] = lin(x[mask])
    loop_times.append((time.time()-since)/50.0)
    print("Avg time for for-loop", num_types, '=', loop_times[-1])




print("Loop Times:", loop_times)
print("Fused Times:", fused_times)
