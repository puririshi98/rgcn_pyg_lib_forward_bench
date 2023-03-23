from torch_geometric.nn.conv import HGTConv, FastHGTConv
from torch_geometric.data import HeteroData
from torch_geometric import seed_everything
import time
import math



seed_everything(420)
data = HeteroData()
data['v0'].x = torch.randn(5, 4).cuda()
data[('v0','e1','v0')].edge_index = torch.randint(high=5, size=(2,10)).cuda()
seed_everything(420)
fast_net = FastHGTConv(4, 2, data.metadata()).to('cuda')
seed_everything(420)
og_net = HGTConv(4, 2, data.metadata()).to('cuda')
x_dict = data.collect('x')
# make params match
for og_param, my_param in zip(og_net.parameters(), fast_net.parameters()):
  try:
    my_param.data = torch.ones_like(my_param.data)
  except:
    pass
  try:
    og_param.data = torch.ones_like(og_param.data)
  except:
    pass

edge_index_dict = data.collect('edge_index')
our_o = list(fast_net(x_dict, edge_index_dict).values())[0]
og_o = list(og_net(x_dict, edge_index_dict).values())[0]
assert torch.allclose(our_o, og_o), "max diff = " + str((our_o - og_o).abs().max()) + '\n diff tensor = ' + str((our_o - og_o)) + '\n my tensor = ' +str(our_o)+ '\n their tensor = ' +str(og_o)
