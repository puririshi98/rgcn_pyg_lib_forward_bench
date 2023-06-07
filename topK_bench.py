from torch_geometric.nn.pool.select.topk import topk
import torch
import time
times = {}
try:
    for x_dim in [10**i for i in range(1,9)]:
          try:
              x = torch.randn((x_dim,), dtype=torch.float, device='cuda')
              batch = x.new_zeros(x.size(0), dtype=torch.long)
              for i in range(60):
                if i > 9:
                  since = time.time()
                topk(x=x, ratio=.5, batch=batch)
              times[x_dim] = (time.time()-since)/50.0 
              print("For", x_dim, "nodes:")
              print("average topK fwd pass time:", times[x_dim])
          except Exception as e:
              print("iter failed w/ exception", e)
              continue
    reprint=True
except Exception as e:
    print("failed w/:", e)
    print("times=", times)
    reprint=False
if reprint:
  print("times=", times)
