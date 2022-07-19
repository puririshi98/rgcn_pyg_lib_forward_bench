from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn import Parameter as Param
from torch_scatter import scatter
from torch_sparse import SparseTensor, masked_select_nnz, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
import math

def masked_edge_index(edge_index, edge_mask):
    return edge_index[:, edge_mask]

def glorot(value):
    stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
    value.data.uniform_(-stdv, stdv)

class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,
    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.
    .. note::
        This implementation is as memory-efficient as possible by iterating
        over each individual relation type.
        Therefore, it may result in low GPU utilization in case the graph has a
        large number of relations.
        As an alternative approach, :class:`FastRGCNConv` does not iterate over
        each individual type, but may consume a large amount of memory to
        compensate.
        We advise to check out both implementations to see which one fits your
        needs.
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
            In case no input features are given, this argument should
            correspond to the number of nodes in your graph.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int, optional): If set, this layer will use the
            basis-decomposition regularization scheme where :obj:`num_bases`
            denotes the number of bases to use. (default: :obj:`None`)
        num_blocks (int, optional): If set, this layer will use the
            block-diagonal-decomposition regularization scheme where
            :obj:`num_blocks` denotes the number of blocks to use.
            (default: :obj:`None`)
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        num_relations: int,
        aggr: str = 'mean',
        root_weight: bool = True,
        bias: bool = True,
        lib: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', aggr)
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.in_channels_l = in_channels[0]

        self.weight = Parameter(
            torch.Tensor(num_relations, in_channels[0], out_channels))
        self.register_parameter('comp', None)
        self.lib = lib
        if bias:
            self.bias = Param(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

    def forward(self, x, edge_index, edge_type):
        r"""
        Args:
            x: The input node features. Can be either a :obj:`[num_nodes,
                in_channels]` node feature matrix, or an optional
                one-dimensional node index tensor (in which case input features
                are treated as trainable node embeddings).
                Furthermore, :obj:`x` can be of type :obj:`tuple` denoting
                source and destination node features.
            edge_index (LongTensor or SparseTensor): The edge indices.
            edge_type: The one-dimensional relation type/index for each edge in
                :obj:`edge_index`.
                Should be only :obj:`None` in case :obj:`edge_index` is of type
                :class:`torch_sparse.tensor.SparseTensor`.
                (default: :obj:`None`)
        """
        # Convert input features to a pair of node features or node indices.
        x_l: OptTensor = None
        if isinstance(x, tuple):
            x_l = x[0]
        else:
            x_l = x
        if x_l is None:
            x_l = torch.arange(self.in_channels_l, device=self.weight.device)

        x_r: Tensor = x_l
        if isinstance(x, tuple):
            x_r = x[1]

        size = (x_l.size(0), x_r.size(0))



        weight = self.weight
        # use pyg-lib segment_matmul
        try:
            import pyg_lib
            pyg_lib_avail = True
        except:
            pyg_lib_avail = False
        out = torch.zeros(x_r.size(0), self.out_channels, device=x_r.device)
        if not self.lib:
            # propagate_type: (x: Tensor)
            for i in range(self.num_relations):
                # print("Relation number:", i)
                tmp = masked_edge_index(edge_index, edge_type == i)
                # print('edge_index.shape=',edge_index.shape)
                # print('edge_type.shape=',edge_type.shape)
                # print('tmp.shape=', tmp.shape)
                # print('x_l.shape=', x_l.shape)
                h = self.propagate(tmp, x=x_l, size=size)
                # print('h.shape=',h.shape)
                # print('weight[i].shape=',weight[i].shape)
                out = out + (h @ weight[i])
        else:
            # not as fast(stil 2.7x speedup over vanilla gpu) but correct
            # h = []
            # for i in range(self.num_relations):
            #     h.append(self.propagate(masked_edge_index(edge_index, edge_type == i), x=x_l, size=size) )     
            # ptr = torch.tensor([i for i in range(0, h[0].shape[0] * (self.num_relations + 1), h[0].shape[0])])
            # h = torch.cat(h)
            # # print('inputs.shape=', h.shape)
            # # print('ptr=',ptr)
            # # print('weight.shape=', weight.shape)
            # # assert not torch.isnan(h).any() and not torch.isinf(h).any()
            # # assert not torch.isnan(weight).any() and not torch.isinf(weight).any()
            # o_tmp = torch.ops.pyg.segment_matmul(h, ptr, weight)
            # # assert not torch.isnan(o_tmp).any() and not torch.isinf(o_tmp).any()
            # out += sum(torch.tensor_split(o_tmp, self.num_relations))
            # assert not torch.isnan(out).any() and not torch.isinf(out).any()

            # not numerically correct but super fast(4.8x)
            # h = self.propagate(edge_index, x=x_l, size=size)      
            # ptr = torch.tensor([i for i in range(0, h.shape[0] * (self.num_relations + 1), h.shape[0])])
            # h = h.repeat(self.num_relations, 1)
            # # print('inputs.shape=', h.shape)
            # # print('ptr=',ptr)
            # # print('weight.shape=', weight.shape)
            # out = sum(torch.tensor_split(torch.ops.pyg.segment_matmul(h, ptr, weight), self.num_relations))

            #attempt at reconciling the two (numerically correct and no for loops)(8.6x)            
            ptr = torch.tensor([i for i in range(0, x_l.shape[0] * (self.num_relations + 1), x_l.shape[0])])
            x_l = x_l.repeat(self.num_relations, 1)
            # print('inputs.shape=', x_l.shape)
            # print('ptr=',ptr)
            # print('weight.shape=', weight.shape)
            x_in = sum(torch.tensor_split(torch.ops.pyg.segment_matmul(x_l, ptr, weight), self.num_relations))
            h = self.propagate(edge_index, x=x_in, size=size)



        if self.bias is not None:
            out += self.bias
        # assert not torch.isnan(out).any() and not torch.isinf(out).any()
        # print('out.shape=',out.shape)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_relations={self.num_relations})')