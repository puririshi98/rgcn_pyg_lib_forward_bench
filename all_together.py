import sys
import rmm

rmm.reinitialize(pool_allocator=True,initial_pool_size=5e+9, maximum_pool_size=20e+9)
sys.path += ['/work/pytorch_geometric', '/work/gaas/python']
import cugraph
import cudf
from ogb.nodeproppred import NodePropPredDataset

dataset = NodePropPredDataset(name = 'ogbn-mag') 

data = dataset[0]
import cudf
import dask_cudf
import cugraph
from cugraph.experimental import MGPropertyGraph
from cugraph.experimental import PropertyGraph
pG = PropertyGraph()

vertex_offsets = {}
last_offset = 0

for node_type, num_nodes in data[0]['num_nodes_dict'].items():
    vertex_offsets[node_type] = last_offset
    last_offset += num_nodes
    
    blank_df = cudf.DataFrame({'id':range(vertex_offsets[node_type], vertex_offsets[node_type] + num_nodes)})
    blank_df.id = blank_df.id.astype('int32')
    if isinstance(pG, MGPropertyGraph):
        blank_df = dask_cudf.from_cudf(blank_df, npartitions=2)
    pG.add_vertex_data(blank_df, vertex_col_name='id', type_name=node_type)

vertex_offsets
for i, (node_type, node_features) in enumerate(data[0]['node_feat_dict'].items()):
    vertex_offset = vertex_offsets[node_type]

    feature_df = cudf.DataFrame(node_features)
    feature_df.columns = [str(c) for c in range(feature_df.shape[1])]
    feature_df['id'] = range(vertex_offset, vertex_offset + node_features.shape[0])
    feature_df.id = feature_df.id.astype('int32')
    if isinstance(pG, MGPropertyGraph):
        feature_df = dask_cudf.from_cudf(feature_df, npartitions=2)

    pG.add_vertex_data(feature_df, vertex_col_name='id', type_name=node_type)
for i, (edge_key, eidx) in enumerate(data[0]['edge_index_dict'].items()):
    node_type_src, edge_type, node_type_dst = edge_key
    print(node_type_src, edge_type, node_type_dst)
    vertex_offset_src = vertex_offsets[node_type_src]
    vertex_offset_dst = vertex_offsets[node_type_dst]
    eidx = [n + vertex_offset_src for n in eidx[0]], [n + vertex_offset_dst for n in eidx[1]]

    edge_df = cudf.DataFrame({'src':eidx[0], 'dst':eidx[1]})
    edge_df.src = edge_df.src.astype('int32')
    edge_df.dst = edge_df.dst.astype('int32')
    edge_df['type'] = edge_type
    if isinstance(pG, MGPropertyGraph):
        edge_df = dask_cudf.from_cudf(edge_df, npartitions=2)

    pG.add_edge_data(edge_df, vertex_col_names=['src','dst'], type_name=edge_type)
    pG.add_edge_data(edge_df, vertex_col_names=['dst','src'], type_name=f'{edge_type}_bw')
y_df = cudf.DataFrame(data[1]['paper'], columns=['y'])
y_df['id'] = range(vertex_offsets['paper'], vertex_offsets['paper'] + len(y_df))
y_df.id = y_df.id.astype('int32')
if isinstance(pG, MGPropertyGraph):
    y_df = dask_cudf.from_cudf(y_df, npartitions=2)

pG.add_vertex_data(y_df, vertex_col_name='id', type_name='paper')
from cugraph.gnn.pyg_extensions.data.cugraph_store import to_pyg

feature_store, graph_store = to_pyg(pG)
from torch_geometric.loader import LinkNeighborLoader
from cugraph.gnn.pyg_extensions import CuGraphLinkNeighborLoader
loader = CuGraphLinkNeighborLoader(
    data=(feature_store, graph_store),
    edge_label_index='writes',
    shuffle=True,
    num_neighbors=[10,25],
    batch_size=50,
)

edge_types = [attr.edge_type for attr in graph_store.get_all_edge_attrs()]
edge_types
num_classes = pG.get_vertex_data(columns=['y'])['y'].max() + 1
if isinstance(pG, MGPropertyGraph):
    num_classes = num_classes.compute()
pG._vertex_prop_dataframe[pG._vertex_prop_dataframe._VERTEX_==1939695]
import torch
import torch.nn.functional as F

from custom_rgcnconv_2 import RGCNConv

n_classes = num_classes
class Net(torch.nn.Module):
    def __init__(self, lib):
        super().__init__()
        self.conv1 = RGCNConv(128, 16, 8, lib=lib)
        self.l2 = RGCNConv(16, num_classes, 8, lib=lib)

    def forward(self, x, edge_index, edge_type):
        x = (self.conv1(x, edge_index, edge_type))
        x = F.relu(x)
        x = self.l2(x, edge_index, edge_type)
        return x
lib = bool(int(sys.argv[2]))
model = Net(lib).to(sys.argv[1])

import time
sumtime = 0

def fuse_batch(batch):
    x_dict = batch.collect('x')
    x = torch.cat(list(x_dict.values()), dim=0)
    num_node_dict = batch.collect('num_nodes')
    increment_dict = {}
    ctr = 0
    
    for node_type in num_node_dict:
        increment_dict[node_type] = ctr
        ctr += num_node_dict[node_type]
    e_idx_dict = batch.collect('edge_index')
    etypes_list = []
    for i, e_type in enumerate(e_idx_dict.keys()):
        src_type, dst_type = e_type[0], e_type[-1]
        if torch.numel(e_idx_dict[e_type]) != 0:
            e_idx_dict[e_type][0, :] = e_idx_dict[e_type][0, :] + increment_dict[src_type]
            e_idx_dict[e_type][1, :] = e_idx_dict[e_type][1, :] + increment_dict[dst_type]
            etypes_list.append(torch.ones(e_idx_dict[e_type].shape[-1]) * i)
    edge_types = torch.cat(etypes_list).to(torch.long).to(sys.argv[1])
    eidx = torch.cat(list(e_idx_dict.values()), dim=1)
    return x, eidx, edge_types
criterion = torch.nn.CrossEntropyLoss()
forward_sumtime = 0
for i, batch in enumerate(data_object.train_dataloader):
    x, edge_index, edge_type = fuse_batch(batch)
    out = model(x, edge_index, edge_type)
    if i>=4:
        forward_sumtime += time.time() - since
    target = batch['paper'].y[:1024]
    loss = criterion(out[:1024], target)
    loss.backward()
    
    if i>=5:
        sumtime += time.time() - since
    if i>=99:
        break
    if i>=4:
        since=time.time()
print('Average Full Iter Time:', (sumtime)/95.0)
torch.cuda.empty_cache()

