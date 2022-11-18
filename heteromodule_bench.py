import torch
from torch_geometric.nn import ToHeteroModule
from torch_geometric.nn.dense import Linear
import time
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
times = []
num_nodes_per_type = 10000
n_feats = 128
out_feats = 64


for num_types in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    x_dict = {'v'+str(i):torch.randn((num_nodes_per_type, n_feats)).cuda() for i in range(num_types)}
    metadata= (x_dict.keys(), [])
    heterolin = ToHeteroModule(Linear(n_feats, out_feats), metadata)
    for i in range(60):
        if i==10:
            since=time.time()
        heterolin(x_dict)
    times.append((time.time()-since)/50.0)
    print("Avg time for", num_types, '=', times[-1])





print("Segment Matmul Times:", times)
