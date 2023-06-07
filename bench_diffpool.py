from torch_geometric.nn.dense import dense_diff_pool
import torch
import time
times = {}
try:
    for B in [2**i for i in range(1,8)]:
      for N in [2**i for i in range(1,10)]:
        for F in [2**i for i in range(1,8)]:
          for C in [2**i for i in range(1,4)]:
            try:
                x = torch.randn((B,N,F,), dtype=torch.float, device='cuda')
                adj = torch.randint(high=2, size=(B,N,N), dtype=torch.long).to_sparse_csr().cuda()
                s = torch.randint(high=2, size=(B,N,C), dtype=torch.long).to_sparse_csr().cuda()
                for i in range(60):
                  if i > 9:
                    since = time.time()
                  dense_diff_pool(x, adj, s)
                times[(B,N,F,C)] = (time.time()-since)/50.0 
                print("For", x_dim, "nodes:")
                print("average topK fwd pass time:", times[(B,N,F,C)])
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
