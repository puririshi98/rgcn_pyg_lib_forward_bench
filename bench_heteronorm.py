from torch_geometric.nn.norm.batch_norm import HeteroBatchNorm as OGNorm
from torch_geometric.nn.norm import HeteroBatchNorm
import torch
import time
def bench(num_node_types=16):
  x_dict = {'n' + str(i):torch.randn(1000, 32).cuda() for i in range(num_node_types)}
  types = list(x_dict.keys())
  ognet = OGNorm(32, len(types)).to('cuda')
  net = HeteroBatchNorm(32, types).to('cuda')
  for i in range(60):
    if i > 9:
      since = time.time()
    net(x_dict)
  my_out_time = float((time.time()-since)/50.0)
  print('fwd pass time for my heteronorm w/ ({:} nodetypes = {:f}'.format(num_node_types, my_out_time))
  for i in range(60):
    if i > 9:
      since = time.time()
    x = torch.cat(list(x_dict.values()))
    sizes = [x_dict[key].size(0) for key in types]
    type_vec = torch.arange(len(types), device=x.device)
    size = torch.tensor(sizes, device=x.device)
    type_vec = type_vec.repeat_interleave(size)
    ognet(x, type_vec)
  og_time = float((time.time()-since)/50.0)
  print('fwd pass time for original heteronorm w/ ({:} nodetypes = {:f}'.format(num_node_types, og_time))
  return my_out_time, og_time

my_fwd_times, og_fwd_times = {}, {}
for num_node_types in [2, 4, 8, 16, 32, 64, 128]:
  my_fwd_times[(num_node_types)], og_fwd_times[(num_node_types)] = bench(num_node_types)
print("my_fwd_times =", my_fwd_times)
print("og_fwd_times =", og_fwd_times)
print("my_time/their_time=", {num_node_types:my_fwd_times[num_node_types]/og_fwd_times[num_node_types] for num_node_types in og_fwd_times.keys()})
