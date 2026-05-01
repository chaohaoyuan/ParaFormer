import os
import pickle as pkl

import networkx as nx
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid

from .data_utils import normalize_feat, rand_train_test_idx


class NCDataset(object):
    """Node classification dataset container.

    Attributes:
        graph: dict with keys 'edge_index', 'node_feat', 'num_nodes'
        label: tensor of labels
    """

    def __init__(self, name, root=None):
        self.name = name
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop,
                ignore_negative=ignore_negative)
            split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(data_dir, dataname, no_feat_norm=False):
    """Load a node classification dataset by name.

    Supported datasets:
        - cora, citeseer, pubmed (Planetoid)
        - film (Actor/Film from geom-gcn)
        - chameleon, squirrel (WebKB with new splits)
        - deezer-europe
    """
    if dataname == 'deezer-europe':
        dataset = load_deezer_dataset(data_dir)
    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname, no_feat_norm)
    elif dataname == 'film':
        dataset = load_geom_gcn_dataset(data_dir, dataname)
    elif dataname in ('chameleon', 'squirrel'):
        dataset = load_wiki_new(data_dir, dataname, no_feat_norm)
    else:
        raise ValueError(f'Invalid dataset name: {dataname}')
    return dataset


def load_deezer_dataset(data_dir):
    dataset = NCDataset('deezer-europe')
    deezer = scipy.io.loadmat(os.path.join(data_dir, 'deezer/deezer-europe.mat'))
    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_planetoid_dataset(data_dir, name, no_feat_norm=False):
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=os.path.join(data_dir, 'Planetoid'),
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=os.path.join(data_dir, 'Planetoid'), name=name)
    data = torch_dataset[0]
    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_geom_gcn_dataset(data_dir, name):
    graph_adjacency_list_file_path = os.path.join(
        data_dir, 'geom-gcn/{}/out1_graph_edges.txt'.format(name))
    graph_node_features_and_labels_file_path = os.path.join(
        data_dir, 'geom-gcn/{}/out1_node_feature_label.txt'.format(name))

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as f:
            f.readline()
            for line in f:
                line = line.rstrip().split('\t')
                assert len(line) == 3
                assert int(line[0]) not in graph_node_features_dict
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as f:
            f.readline()
            for line in f:
                line = line.rstrip().split('\t')
                assert len(line) == 3
                assert int(line[0]) not in graph_node_features_dict
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as f:
        f.readline()
        for line in f:
            line = line.rstrip().split('\t')
            assert len(line) == 2
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [feat for _, feat in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [lbl for _, lbl in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    def preprocess_features(feat):
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels
    return dataset


def load_wiki_new(data_dir, name, no_feat_norm=False):
    path = os.path.join(data_dir, f'wiki_new/{name}/{name}_filtered.npz')
    data = np.load(path)
    node_feat = data['node_features']
    labels = data['node_labels']
    edges = data['edges']
    edge_index = edges.T

    if not no_feat_norm:
        node_feat = normalize_feat(node_feat)

    dataset = NCDataset(name)
    edge_index = torch.as_tensor(edge_index).long()
    node_feat = torch.as_tensor(node_feat).float()
    labels = torch.as_tensor(labels).long()

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': node_feat.shape[0]}
    dataset.label = labels
    return dataset
