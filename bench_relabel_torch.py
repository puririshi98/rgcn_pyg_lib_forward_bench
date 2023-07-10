import torch_geometric
import torch
# scale up to see when it breaks
subet = torch.tensor([3, 4, 5], device='cuda', dtype=torch.long)
# scale up to see when it breaks
for i in range(2, 10):
  print("trying w/ num nodes = 10^" + str(i))
  num_nodes = 10**i
  data = torch_geometric.datasets.FakeDataset(avg_num_nodes=num_nodes).data
  node_idx = torch.zeros(data.x.size(0), dtype=torch.long,
                       device='cuda')
  node_idx[subset] = torch.arange(3, device=device)
  data.edge_index = node_idx[data.edge_index]
