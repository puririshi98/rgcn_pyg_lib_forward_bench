import torch
import pyg_lib
import time
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
times = []
num_nodes_per_type = 10000
n_feats = 128
out_feats = 64


for num_types in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    inputs = torch.randn((num_nodes_per_type * num_types, n_feats)).cuda()
    ptr = torch.tensor(list(range(0, (num_types + 1) * num_nodes_per_type, num_nodes_per_type))).cuda()
    other = torch.randn((num_types, n_feats, out_feats), requires_grad=True).cuda()
    since=time.time()
    for i in range(10):
        pyg_lib.ops.segment_matmul(inputs, ptr, other)
    times.append((time.time()-since)/10.0)
    print("Avg time for", num_types, '=', times[-1])





print("Segment Matmul Times:", times)
