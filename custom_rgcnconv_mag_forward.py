import enum
import json
import os
import pathlib
import re
import shutil
import time
from collections import defaultdict
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import NodePropPredDataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader

#define rgcn
class StrEnum(str, enum.Enum):
    def __new__(cls, *args):
        for arg in args:
            if not isinstance(arg, (str, enum.auto)):
                raise TypeError(
                    "Values of StrEnums must be strings: {} is a {}".format(
                        repr(arg), type(arg)
                    )
                )
        return super().__new__(cls, *args)

    def __str__(self):
        return self.value

    # def __repr__(self):
    #    return self.value

    def _generate_next_value_(name, *_):
        return name
class Meta(StrEnum):
    NODES = "nodes"
    NODE_TYPES = "node_types"
    EDGES = "edges"
    EDGE_TYPES = "edge_types"
    SPLIT = "split"
    SPLIT_NAME = "split_column_name"
    SRC_TYPE = "src_node_type"
    DST_TYPE = "dst_node_type"
    SRC_ID = "src_node_id"
    DST_ID = "dst_node_id"
    REVERSE = "generate_reverse_name"
    FEAT = "features"
    NAME = "name"
    LABEL = "label"
    FILES = "file_paths"
    FEAT_NAME = "name"
    EDGE_ID = "id"
    DTYPE = "dtype"
    SHAPE = "shape"
    NUM_NODES_DICT = "num_nodes_dict"
    # EDGE_SUBMATRIX_SHAPES_DICT = "edge_submatrix_shapes_dict"
    EDGE_FEATURE_DICT = "edge_feature_dict"
    NUM_EDGES_DICT = "num_edges_dict"
    EDGE_SPLIT_AT_DICT = "edge_split_at_dict"
    NODE_SPLIT_AT_DICT = "node_split_at_dict"
    # NODE_SUBMATRIX_SHAPES_DICT = "node_submatrix_shapes_dict"
    CATEGORICAL_WIDTH = "categorical_width"


MetadataKeys = Meta


def load_metadata(root_path):
    try:
        with open(os.path.join(root_path, "metadata.json")) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def make_pyg_loader(graph, train_ids, metadata, device):
    src_node_types = set(
        [
            i[MetadataKeys.SRC_TYPE]
            for i in metadata[MetadataKeys.EDGES][MetadataKeys.EDGE_TYPES]
        ]
    )
    dst_node_types = set(
        [
            i[MetadataKeys.DST_TYPE]
            for i in metadata[MetadataKeys.EDGES][MetadataKeys.EDGE_TYPES]
        ]
    )
    unupdated_nodes = list(set(src_node_types - dst_node_types))
    T_list = [T.ToDevice(device)]
    if len(unupdated_nodes) > 0:
        # If a node type does not get filled with message passing this will cause errors
        # Solve this by adding reverse edges:
        T_list += [T.ToUndirected(merge=False)]
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / 2
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / 2
    num_work = int(num_work)
    return NeighborLoader(
        graph,
        num_neighbors=[50, 50],
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        input_nodes=(graph.labeled_node_type, train_ids[graph.labeled_node_type]),
        num_workers=num_work,
        replace=True,
        transform=T.Compose(T_list),
    )


def get_split_nids(graph, metadata, split):
    split_values = {}
    for node in metadata[Meta.NODES][Meta.NODE_TYPES]:
        compare = None
        if node[Meta.NAME] in graph[Meta.NODE_SPLIT_AT_DICT].keys():
            compare = graph[Meta.NODE_SPLIT_AT_DICT][node[Meta.NAME]]
        if compare is not None:
            split_values[node[Meta.NAME]] = (
                (compare == split).nonzero()[:, 0].type(torch.int64)
            )
    return split_values


def extract_block_one_hot_labels(
    batch, label_name, label_size, node_type, device, logits
):
    batch_size = batch[batch.labeled_node_type].batch_size
    return (
        logits[:batch_size],
        torch.nn.functional.one_hot(
            batch[node_type].y[:batch_size], num_classes=label_size
        )
        .float()
        .to(device),
    )


def get_logit_labels_types_block(batch, logits, node_label_map, device):
    all_logits = []
    all_labels = []
    for node_type, logits in logits.items():
        label_names = node_label_map[node_type]
        for label_name, label_size in label_names:
            # - XXX: assumes labels are present for all nodes in batch
            #        will require a mask for filtering
            logits, labels = extract_block_one_hot_labels(
                batch, label_name, label_size, node_type, device, logits
            )
            all_logits.append(logits)
            all_labels.append(labels)
    labels = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)
    return logits, labels


def read_pandas_feats(root_path, dicty):
    return pd.concat(
        [pd.read_parquet(os.path.join(root_path, path)) for path in dicty[Meta.FILES]]
    )


def make_split(num, g, key):
    train, val = int(0.8 * num), int(0.1 * num)
    test = num - train - val
    split = torch.cat((torch.zeros(train), torch.ones(val), 2 * torch.ones(test)))[
        torch.randperm(num)
    ].to("cpu")
    dictkey = (
        Meta.NODE_SPLIT_AT_DICT
        if not isinstance(key, tuple)
        else Meta.EDGE_SPLIT_AT_DICT
    )
    g[dictkey][key] = split
    return g


def load_graph(root_path, metadata, use_reverse_edges_features=False):
    # Initialize PyG Graph object
    g = HeteroData()
    # print("Raw Metadata:", metadata)
    relation_types = []
    g[Meta.NUM_EDGES_DICT] = {}
    biggest_node_id_dict = {}
    # Parse Edge Indexes
    for edge in metadata[Meta.EDGES][Meta.EDGE_TYPES]:
        feats = read_pandas_feats(root_path, edge)
        relation = (
            edge[Meta.SRC_TYPE],
            edge[Meta.NAME],
            edge[Meta.DST_TYPE],
        )
        relation_types.append(relation)
        src = (
            torch.tensor(feats[Meta.SRC_ID].astype("int64").values)
            .to("cpu")
            .reshape(1, -1)
        )
        dst = (
            torch.tensor(feats[Meta.DST_ID].astype("int64").values)
            .to("cpu")
            .reshape(1, -1)
        )
        if edge[Meta.SRC_TYPE] not in biggest_node_id_dict.keys():
            biggest_node_id_dict[edge[Meta.SRC_TYPE]] = int(torch.max(src))
        else:
            biggest_node_id_dict[edge[Meta.SRC_TYPE]] = max(
                biggest_node_id_dict[edge[Meta.SRC_TYPE]], int(torch.max(src))
            )
        if edge[Meta.DST_TYPE] not in biggest_node_id_dict.keys():
            biggest_node_id_dict[edge[Meta.DST_TYPE]] = int(torch.max(dst))
        else:
            biggest_node_id_dict[edge[Meta.DST_TYPE]] = max(
                biggest_node_id_dict[edge[Meta.DST_TYPE]], int(torch.max(dst))
            )
        g[relation].edge_index = torch.cat((src, dst), axis=0)
        g[Meta.NUM_EDGES_DICT][relation] = g[relation].edge_index.size()[-1]
        if edge.get(Meta.REVERSE, None):
            relation = (
                edge[Meta.DST_TYPE],
                edge[Meta.REVERSE],
                edge[Meta.SRC_TYPE],
            )
            g[relation].edge_index = torch.cat((dst, src), axis=0)
            g[Meta.NUM_EDGES_DICT][relation] = g[relation].edge_index.size()[-1]
    node_types = []
    filled_node_types = []
    # PyG Hetero Loaders don't like additional information in the node stores
    # Make seperate dicts to hold info
    g[Meta.NODE_SPLIT_AT_DICT] = {}
    # Parse Node Features
    g.labeled_node_type = ""
    for node in metadata[Meta.NODES][Meta.NODE_TYPES]:
        node_types.append(node[Meta.NAME])
        if node.get(Meta.FILES):
            filled_node_types.append(node)
            feats = read_pandas_feats(root_path, node)
            list_of_submtrx_to_cat = []
            # submtrx_shapes = []
            for feature in node[Meta.FEAT]:
                # Store Node Labels
                if feature[Meta.NAME] == Meta.LABEL or Meta.LABEL in feature.keys():
                    g[node[Meta.NAME]].y = (
                        torch.tensor(feats[feature[Meta.NAME]])
                        .reshape(-1)
                        .to(torch.int64)
                    )
                    g.labeled_node_type = node[Meta.NAME]
                elif isinstance(feature[Meta.NAME], list):
                    for name in feature[Meta.NAME]:
                        submatrix = torch.Tensor(feats[name]).to("cpu")
                        list_of_submtrx_to_cat.append(
                            submatrix
                            if len(submatrix.size()) == 2
                            else submatrix.reshape(1, -1)
                        )
                        # For look up table
                        # submtrx_shapes.append(
                        #     (name, list_of_submtrx_to_cat[-1].size()[0])
                        # )
                else:
                    submatrix = torch.Tensor(feats[feature[Meta.NAME]]).to("cpu")
                    list_of_submtrx_to_cat.append(
                        submatrix
                        if len(submatrix.size()) == 2
                        else submatrix.reshape(1, -1)
                    )
                    # submtrx_shapes.append(
                    #     (
                    #         feature[Meta.NAME],
                    #         list_of_submtrx_to_cat[-1].size()[0],
                    #     )
                    # )
            # Store node features
            g[node[Meta.NAME]].x = torch.cat(list_of_submtrx_to_cat).T
            # Store individual feature shapes for future retrieval
            # g[node[Meta.NAME]].submtrx_shapes = dict(submtrx_shapes)
            g[node[Meta.NAME]].num_nodes = g[node[Meta.NAME]].x.size()[0]
            # Store given split or choose at random
            g = make_split(g[node[Meta.NAME]].num_nodes, g, node[Meta.NAME])
        else:
            # Need to atleast store num nodes. DGL does this automatically
            # PyG does not
            g[node[Meta.NAME]].num_nodes = biggest_node_id_dict[node[Meta.NAME]] + 1
        g[node[Meta.NAME]].n_id = torch.arange(g[node[Meta.NAME]].num_nodes)

    if Meta.NUM_NODES_DICT in metadata.keys():
        g.num_nodes = sum(list(metadata[Meta.NUM_NODES_DICT].values()))
    else:
        g.num_nodes = sum([g[node[Meta.NAME]].num_nodes for node in filled_node_types])

    # PyG Hetero Loaders don't like additional information in the edge stores
    # Make seperate dicts to hold info
    # g[Meta.EDGE_SUBMATRIX_SHAPES_DICT] = {}
    g[Meta.EDGE_FEATURE_DICT] = {}
    g[Meta.EDGE_SPLIT_AT_DICT] = {}
    g.predictable_edge_type = ""
    for edge in metadata[Meta.EDGES][Meta.EDGE_TYPES]:
        relation = (
            edge[Meta.SRC_TYPE],
            edge[Meta.NAME],
            edge[Meta.DST_TYPE],
        )
        if edge.get(Meta.FILES):
            feats = read_pandas_feats(root_path, edge)
            list_of_submtrx_to_cat = []
            # submtrx_shapes = []
            for feature in edge[Meta.FEAT]:
                if feature[Meta.NAME] == Meta.LABEL or Meta.LABEL in feature.keys():
                    g[relation].y = (
                        torch.tensor(feats[feature[Meta.NAME]].values)
                        .reshape(-1)
                        .to(torch.int64)
                    )
                    g.predictable_edge_type = tuple(relation)
                elif isinstance(feature[Meta.NAME], list):
                    for name in feature[Meta.NAME]:
                        submatrix = torch.Tensor(feats[name]).to("cpu")
                        list_of_submtrx_to_cat.append(
                            submatrix
                            if len(submatrix.size()) == 2
                            else submatrix.reshape(1, -1)
                        )
                        # submtrx_shapes.append(
                        #     (name, list_of_submtrx_to_cat[-1].size()[0])
                        # )
                else:
                    submatrix = torch.Tensor(feats[feature[Meta.NAME]].values).to("cpu")
                    list_of_submtrx_to_cat.append(
                        submatrix
                        if len(submatrix.size()) == 2
                        else submatrix.reshape(1, -1)
                    )
                    # submtrx_shapes.append(
                    #     (
                    #         str(feature[Meta.NAME]),
                    #         list_of_submtrx_to_cat[-1].size()[0],
                    #     )
                    # )
            if list_of_submtrx_to_cat:
                g[Meta.EDGE_FEATURE_DICT][relation] = torch.cat(
                    list_of_submtrx_to_cat
                ).T
                if edge.get(Meta.REVERSE, None) and use_reverse_edges_features:
                    rev_relation = (
                        edge[Meta.DST_TYPE],
                        edge[Meta.REVERSE],
                        edge[Meta.SRC_TYPE],
                    )
                    g[Meta.EDGE_FEATURE_DICT][rev_relation] = g[Meta.EDGE_FEATURE_DICT][
                        relation
                    ].clone()
            # g[Meta.EDGE_SUBMATRIX_SHAPES_DICT][relation] = dict(submtrx_shapes)
            g = make_split(g[Meta.NUM_EDGES_DICT][relation], g, relation)
    g.edge_types = relation_types
    g.node_types = node_types
    return g

class OGBN_MAG:
    """
    The OGBN_MAG class includes the transformation
    operation for a subset of the Microsoft Academic Graph (MAG).
    It's a heterogeneous network that contains four types of entities—papers
    (736,389 nodes), authors (1,134,649 nodes), institutions (8,740 nodes),
    and fields of study (59,965 nodes)—as well as four types of directed relations
    connecting two types of entities—an author is “affiliated with” an institution,
    an author “writes” a paper, a paper “cites” a paper, and a paper “has a topic
    of” a field of study. For more information, please check
    https://ogb.stanford.edu/docs/nodeprop/

    Example usage::
    # Create an instance and call transform

    from gp.preprocessing.trans_dataset import OGBN_Products
    o = OGBN_MAG(source_path, destination_path)
    o.transform()

    Parameters
    ----------
    source_path: str
        source path for downloading the original data.
    dest_path: str
        destination path for putting the transformed data.
    """

    def __init__(self, source_path: str, dest_path: str):
        self.source_path = source_path
        self.dest_path = dest_path
        self.labels = None
        self.node_data = None
        self.edge_data = None
        self.metadata = None
        self.node_types = []
        self.edge_types = []

    def transform(self):
        """
        Transforms the OGBN Mag dataset to GP's data format.
        :return: None
        """
        import cudf

        # Download and prepare the OGBN MAG dataset using the
        # OGBN's  NodePropPredDatasetfunction.
        dataset = NodePropPredDataset(name="ogbn-mag", root=self.source_path)[0]
        feat_key = "paper"
        data = dataset[0]
        labels = torch.tensor(dataset[1][feat_key])
        # All the edge types have features. So, we get each edge type one by one.
        self.edge_data = {}
        for e, edges in data["edge_index_dict"].items():
            # Get the given edge type in the order of source to destination.
            # So, first column will have the source and the second one will have dest.
            # Third column has the ids of the edges.
            edata = torch.tensor(data["edge_reltype"][e])
            src_nodes = np.array(edges[0, :])
            dest_nodes = np.array(edges[1, :])
            edge_ids = np.array(torch.arange(edges.shape[1]))
            self.edge_data[e[1]] = cudf.DataFrame(
                {
                    MetadataKeys.EDGE_ID: edge_ids,
                    MetadataKeys.SRC_ID: src_nodes,
                    MetadataKeys.DST_ID: dest_nodes,
                }
            )
            self.edge_data[e[1]]["feat"] = cudf.DataFrame(np.array(edata))

            feature = {
                MetadataKeys.NAME: "feat",
                MetadataKeys.DTYPE: str(edata.type()),
                MetadataKeys.SHAPE: edata.size(),
            }
            # e[0] = source node type, e[1] = edge type, e[2] = destination node type
            edge_type = {
                MetadataKeys.NAME: e[1],
                MetadataKeys.FILES: ["./edge_data/edge_type_" + e[1] + ".parquet"],
                MetadataKeys.SRC_TYPE: e[0],
                MetadataKeys.DST_TYPE: e[2],
                MetadataKeys.FEAT: [feature],
            }

            self.edge_types.append(edge_type)

        # Only the node type 'paper' has features in this dataset.
        feat_val = torch.tensor(data["node_feat_dict"][feat_key])

        self.node_data = dict()
        # Get the 'paper' node type data and convert it to cudf
        self.node_data["paper"] = cudf.DataFrame(np.array(feat_val)).astype("float32")

        # Set a string column name so that parquet doesn't complain about it
        new_col_names = {i: "feat_" + str(i) for i in self.node_data["paper"].columns}
        feat_col_names = [val for val in new_col_names.values()]
        self.node_data["paper"] = self.node_data["paper"].rename(columns=new_col_names)

        # Another node level data in the graph is the 'year' info for the samples.
        # Putting the 'year info into the graph data as a feature.
        year_data = torch.tensor(data["node_year"][feat_key])
        self.node_data["paper"]["year"] = cudf.DataFrame(np.array(year_data)).astype(
            "int32"
        )
        # venue is the node labels for this dataset.
        self.node_data["paper"]["venue"] = cudf.DataFrame(np.array(labels)).astype(
            "int32"
        )

        # Split the data based on the year like what's described in OGBN's website.
        self.node_data["paper"]["split"] = cudf.Series(
            np.zeros(self.node_data["paper"]["venue"].size), dtype=np.int8
        )
        self.node_data["paper"].loc[
            self.node_data["paper"]["year"] == 2018, "split"
        ] = 1
        self.node_data["paper"].loc[self.node_data["paper"]["year"] > 2018, "split"] = 2
        self.node_data["paper"].drop(columns=["year"], inplace=True)

        # Calculate author, institution features.
        self.node_data["paper"]["paper_id"] = list(
            range(0, self.node_data["paper"].shape[0])
        )

        author_feat = (
            self.edge_data["writes"]
            .merge(
                self.node_data["paper"],
                left_on="dst_node_id",
                right_on="paper_id",
                how="left",
            )
            .groupby("src_node_id", sort=True)
            .mean()
        )

        new_feat_col_names = [val for val in new_col_names.values()]
        self.node_data["author"] = author_feat[new_feat_col_names].astype("float32")
        self.node_data["author"]["author_id"] = list(
            range(0, self.node_data["author"].shape[0])
        )
        self.node_data["author"]["split"] = cudf.Series(
            np.zeros(self.node_data["author"].shape[0]), dtype=np.int8
        )

        institution_feat = (
            self.edge_data["affiliated_with"]
            .merge(
                self.node_data["author"], left_on="src_node_id", right_on="author_id"
            )
            .groupby("dst_node_id", sort=True)
            .mean()
        )
        self.node_data["institution"] = institution_feat[new_feat_col_names].astype(
            "float32"
        )
        self.node_data["institution"]["split"] = cudf.Series(
            np.zeros(self.node_data["institution"].shape[0]), dtype=np.int8
        )

        field_of_study = (
            self.edge_data["has_topic"]
            .merge(self.node_data["paper"], left_on="src_node_id", right_on="paper_id")
            .groupby("dst_node_id", sort=True)
            .mean()
        )
        self.node_data["field_of_study"] = field_of_study[new_feat_col_names].astype(
            "float32"
        )
        self.node_data["field_of_study"]["split"] = cudf.Series(
            np.zeros(self.node_data["field_of_study"].shape[0]), dtype=np.int8
        )

        # Get the required info for the metadata file
        self.node_data["paper"].drop(columns=["paper_id"], inplace=True)
        features = dict()
        features["paper"] = list()
        features["paper"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: feat_val.size(),
            }
        )
        features["paper"].append(
            {
                MetadataKeys.NAME: "venue",
                MetadataKeys.DTYPE: str(labels.type()),
                MetadataKeys.SHAPE: labels.size(),
                MetadataKeys.LABEL: True,
            }
        )

        self.node_data["author"].drop(columns=["author_id"], inplace=True)
        features["author"] = list()
        features["author"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: [
                    self.node_data["author"].shape[0],
                    self.node_data["author"].shape[1] - 1,
                ],
            }
        )

        features["institution"] = list()
        features["institution"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: [
                    self.node_data["institution"].shape[0],
                    self.node_data["institution"].shape[1] - 1,
                ],
            }
        )

        features["field_of_study"] = list()
        features["field_of_study"].append(
            {
                MetadataKeys.NAME: feat_col_names,
                MetadataKeys.DTYPE: str(feat_val.type()),
                MetadataKeys.SHAPE: [
                    self.node_data["field_of_study"].shape[0],
                    self.node_data["field_of_study"].shape[1] - 1,
                ],
            }
        )
        ntypes = data["num_nodes_dict"].keys()
        # There are multiple node types in this dataset.
        for i, ntype in enumerate(ntypes):
            # Only one node type has features. Once that is found, features can be added
            # into the metadata.
            if ntype in features:
                node_type = {
                    MetadataKeys.NAME: ntype,
                    MetadataKeys.FILES: ["./node_data/node_type_" + ntype + ".parquet"],
                    MetadataKeys.FEAT: features[ntype],
                }
            else:
                # Since these node types don't have any feature, nothing is added.
                node_type = {MetadataKeys.NAME: ntype, MetadataKeys.FEAT: []}

            self.node_types.append(node_type)

        # We create the metadata in the end.
        self.metadata = {
            MetadataKeys.NODES: {
                MetadataKeys.NODE_TYPES: self.node_types,
                MetadataKeys.SPLIT_NAME: "split",
            },
            MetadataKeys.EDGES: {MetadataKeys.EDGE_TYPES: self.edge_types},
        }

        self._write_to_files()

    def _write_to_files(self, exist_ok=True):
        os.makedirs(os.path.join(self.dest_path, "node_data/"), exist_ok=exist_ok)
        os.makedirs(os.path.join(self.dest_path, "edge_data/"), exist_ok=exist_ok)

        for i in range(len(self.node_types)):
            if self.node_types[i][MetadataKeys.NAME] in self.node_data.keys():
                self.node_data[self.node_types[i][MetadataKeys.NAME]].to_parquet(
                    os.path.join(
                        self.dest_path, self.node_types[i][MetadataKeys.FILES][0]
                    )
                )
        for i in range(len(self.edge_types)):
            if self.edge_types[i][MetadataKeys.NAME] in self.edge_data.keys():
                self.edge_data[self.edge_types[i][MetadataKeys.NAME]].to_parquet(
                    os.path.join(
                        self.dest_path, self.edge_types[i][MetadataKeys.FILES][0]
                    )
                )

        with open(os.path.join(self.dest_path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
class DataObject:
    def __init__(
        self,
        train_dataloader=None,
        valid_dataloader=None,
        test_dataloader=None,
        data_path: Optional[str] = None,
        target_extraction_policy=None,
        use_reverse_edges_features=False,
    ):
        self.data_path = data_path
        self.use_reverse_edges_features = use_reverse_edges_features
        self.construct_cache = {}
        self.metadata = load_metadata(data_path)
        # import pdb; pdb.set_trace()
        self.load_data(data_path)
        self.construct_cache["train_dataloader"] = train_dataloader
        self.construct_cache["valid_dataloader"] = valid_dataloader
        self.construct_cache["test_dataloader"] = test_dataloader
        self.build_train_dataloader_pre_dist(train_dataloader)

    def load_data(self, data_path):
        self.construct_cache["graph"] = load_graph(
            data_path,
            self.metadata,
            use_reverse_edges_features=self.use_reverse_edges_features,
        )

    def init_post_dist(self, device="cuda", use_ddp=False):
        self.device = device
        self.use_ddp = use_ddp
        self.build_train_dataloader_post_dist(device, use_ddp)

    # For task specific dataloading, users should implement an inheritor class of DataObject
    def build_train_dataloader_pre_dist(self, dataloader_spec):
        self.train_dataloader = (dataloader_spec, self.construct_cache["graph"])

    def build_train_dataloader_post_dist(self, device, use_ddp):
        return


    @property
    def train(self):
        return self.train_dataloader

    def get_labels(self, batch):
        raise NotImplementedError

    @property
    def get_metadata(self):
        return self.metadata

    @property
    def graph(self):
        return self.construct_cache.get("graph")


class NodeDataObject(DataObject):
    def build_train_dataloader_pre_dist(self, dataloader_spec):
        train_ids = get_split_nids(
            self.construct_cache["graph"], self.metadata, split=0
        )
        self.construct_cache["train_ids"] = train_ids

    def build_train_dataloader_post_dist(self, device):
        train_ids = self.construct_cache["train_ids"]
        self.train_dataloader = make_pyg_loader(
            self.construct_cache["graph"],
            train_ids,
            self.metadata,
            device,
        )
DATA_DIR = "/workspace/data/"

from custom_rgcnconv import RGCNConv

torch.cuda.empty_cache()
source_path = os.path.join(DATA_DIR, "ogbn/mag/")
destination_path = os.path.join(DATA_DIR, "ogbn/mag/GP_Transformed/")

if not os.path.exists(destination_path):
    prep = OGBN_MAG(source_path, destination_path)
    prep.transform()

data_object = NodeDataObject(
    data_path=destination_path,
)
data_object.build_train_dataloader_post_dist('cpu')
data = data_object.graph
n_classes = torch.numel(torch.unique(data['paper'].y))
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = RGCNConv(128, 64, 4)
        self.conv2 = RGCNConv(64, n_classes, 4)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return F.log_softmax(x, dim=1)
model = Net()
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
    print({etype:val.shape for etype,val in e_idx_dict.items()})
    for i, e_type in enumerate(e_idx_dict.keys()):
        if torch.numel(e_idx_dict[e_type]) != 0:
            src_type, dst_type = e_type[0], e_type[-1]
            new_idxs = e_idx_dict[e_type][0]+ increment_dict[src_type]
            e_idx_dict[e_type] = new_idxs[1] + increment_dict[dst_type]
            etypes_list.append(torch.ones_like(e_idx_dict[e_type]) * i)
    print(len(etypes_list))
    print([i.shape for i in etypes_list])
    print({etype:val.shape for etype,val in e_idx_dict.items()})
    edge_types = torch.cat(etypes_list)
    return x, torch.cat(list(e_idx_dict.values()), dim=1), edge_types
for i, batch in enumerate(data_object.train_dataloader):
    x, edge_index, edge_type = fuse_batch(batch)
    since=time.time()
    out = model(x, edge_index, edge_type)
    sumtime += time.time() - since
    if i==49:
        break
print('Average forward pass time:', sumtime/50.0)
torch.cuda.empty_cache()
