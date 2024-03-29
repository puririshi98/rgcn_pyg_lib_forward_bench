import torch

from torch_geometric.nn.dense import Linear, HeteroDictLinear
import time
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
dict_times = {}
loop_times = {}
num_nodes_per_type = 10000
n_feats = 128
out_feats = 64


for num_types in [4, 8, 16, 20, 21, 22, 23, 24, 32, 40, 45, 50, 55, 60, 64, 128, 256, 512, 1024]:
    x_dict = {'v'+str(i):torch.randn((num_nodes_per_type, n_feats)).cuda() for i in range(num_types)}
    metadata= (x_dict.keys(), [])
    x = torch.cat(list(x_dict.values()), dim=0)
    node_type = torch.cat([(j * torch.ones(x_j.shape[0])).long()
                           for j, x_j in enumerate(x_dict.values())])
    lin = Linear(n_feats, out_feats).cuda()
    heterolin = HeteroDictLinear(n_feats, out_feats, types=list(x_dict.keys())).cuda()
    for i in range(60):
        if i==10:
            since=time.time()
        heterolin(x_dict)
    dict_times[num_types] = ((time.time()-since)/50.0)
    print("Avg time for dict based", num_types, '=', dict_times[num_types])
    os = []
    for i in range(60):
        if i==10:
            since=time.time()
        for i in range(num_types):
            k = 'v'+str(i)
            lin(x_dict[k])
    loop_times[num_types] = ((time.time()-since)/50.0)
    print("Avg time for for-loop", num_types, '=', loop_times[num_types])




print("Loop Times:", loop_times)
print("Dict Times:", dict_times)
