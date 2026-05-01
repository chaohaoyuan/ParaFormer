import json
import csv
import os
import pickle as pkl
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import subgraph
from ogb.nodeproppred import NodePropPredDataset, PygNodePropPredDataset

from .data_utils import rand_train_test_idx, even_quantile_labels, to_sparse_tensor, class_rand_splits


class NCDataset(object):
    """Node classification dataset container."""

    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25,
                      label_num_per_class=20):
        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop,
                ignore_negative=ignore_negative)
            split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(
                self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname=''):
    """Load a large graph dataset by name.

    Supported datasets:
        - ogbn-arxiv, ogbn-products, ogbn-proteins (OGB)
        - arxiv-year, pokec, snap-patents, yelp-chi
        - twitch-e, fb100, deezer-europe
        - amazon-photo, amazon-computer
        - coauthor-cs, coauthor-physics
        - cora, citeseer, pubmed (Planetoid)
        - chameleon, squirrel, film, cornell, texas, wisconsin (geom-gcn)
    """
    if dataname == 'twitch-e':
        if sub_dataname not in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW'):
            print('Invalid sub_dataname, deferring to DE graph')
            sub_dataname = 'DE'
        dataset = load_twitch_dataset(data_dir, sub_dataname)
    elif dataname == 'fb100':
        if sub_dataname not in ('Penn94', 'Amherst41', 'Cornell5',
                                 'Johns Hopkins55', 'Reed98'):
            print('Invalid sub_dataname, deferring to Penn94 graph')
            sub_dataname = 'Penn94'
        dataset = load_fb100_dataset(data_dir, sub_dataname)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset(data_dir)
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset(data_dir)
    elif dataname == 'arxiv-year':
        dataset = load_arxiv_year_dataset(data_dir)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    elif dataname == 'snap-patents':
        dataset = load_snap_patents_mat(data_dir)
    elif dataname == 'yelp-chi':
        dataset = load_yelpchi_dataset(data_dir)
    elif dataname == 'amazon2m':
        dataset = load_amazon2m_dataset(data_dir)
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'ogbn-papers100M':
        dataset = load_papers100M(data_dir)
    elif dataname == 'ogbn-papers100M-sub':
        dataset = papers100M_sub(data_dir)
    elif dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname)
    elif dataname in ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, dataname)
    elif dataname in ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(data_dir, dataname)
    else:
        raise ValueError(f'Invalid dataname: {dataname}')
    return dataset


def load_twitch_dataset(data_dir, lang):
    assert lang in ('DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU', 'TW')
    filepath = os.path.join(data_dir, f'twitch/{lang}')
    label, node_ids, src, targ = [], [], [], []
    uniq_ids = set()
    with open(f'{filepath}/musae_{lang}_target.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == 'True'))
                node_ids.append(int(row[5]))
    node_ids = np.array(node_ids, dtype=np.int64)
    with open(f'{filepath}/musae_{lang}_edges.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))
    with open(f'{filepath}/musae_{lang}_features.json', 'r') as f:
        j = json.load(f)
    src, targ = np.array(src), np.array(targ)
    label = np.array(label)
    inv_node_ids = {nid: idx for idx, nid in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]
    n = label.shape[0]
    A = scipy.sparse.csr_matrix(
        (np.ones(len(src)), (np.array(src), np.array(targ))), shape=(n, n))
    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    label = label[reorder_node_ids]
    dataset = NCDataset(lang)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = node_feat.shape[0]
    dataset.graph = {'edge_index': edge_index, 'edge_feat': None,
                     'node_feat': node_feat, 'num_nodes': num_nodes}
    dataset.label = torch.tensor(label)
    return dataset


def load_fb100_dataset(data_dir, filename):
    feature_vals_all = np.empty((0, 6))
    for f in ['Penn94', 'Amherst41', 'Cornell5', 'Johns Hopkins55', 'Reed98']:
        mat = scipy.io.loadmat(os.path.join(data_dir, f'facebook100/{f}.mat'))
        metadata = mat['local_info'].astype(np.int64)
        feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
        feature_vals_all = np.vstack((feature_vals_all, feature_vals))
    mat = scipy.io.loadmat(os.path.join(data_dir, f'facebook100/{filename}.mat'))
    A, metadata = mat['A'], mat['local_info'].astype(np.int64)
    dataset = NCDataset(filename)
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    label = metadata[:, 1] - 1
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feature_vals_all[:, col]))
        features = np.hstack((features, feat_onehot))
    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    dataset.graph = {'edge_index': edge_index, 'edge_feat': None,
                     'node_feat': node_feat, 'num_nodes': num_nodes}
    dataset.label = torch.where(torch.tensor(label) > 0, 1, 0)
    return dataset


def load_deezer_dataset(data_dir):
    dataset = NCDataset('deezer-europe')
    deezer = scipy.io.loadmat(os.path.join(data_dir, 'deezer/deezer-europe.mat'))
    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]
    dataset.graph = {'edge_index': edge_index, 'edge_feat': None,
                     'node_feat': node_feat, 'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_arxiv_year_dataset(data_dir, nclass=5):
    dataset = NCDataset('arxiv-year')
    ogb_dataset = NodePropPredDataset(name='ogbn-arxiv', root=os.path.join(data_dir, 'ogb'))
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])
    label = even_quantile_labels(dataset.graph['node_year'].flatten(), nclass, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    return dataset


def load_amazon2m_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-products', root=os.path.join(data_dir, 'ogb'))
    dataset = NCDataset('amazon2m')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    dataset.graph['num_node'] = torch.as_tensor(dataset.label.shape[0])

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        dir_path = os.path.join(data_dir, 'ogb/ogbn_products/split/random_0.5_0.25')
        tensor_split_idx = {}
        if os.path.exists(dir_path):
            for key in ['train', 'valid', 'test']:
                tensor_split_idx[key] = torch.as_tensor(
                    np.loadtxt(os.path.join(dir_path, f'amazon2m_{key}.txt')), dtype=torch.long)
        else:
            os.makedirs(dir_path)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] = \
                rand_train_test_idx(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            for key in tensor_split_idx:
                np.savetxt(os.path.join(dir_path, f'amazon2m_{key}.txt'),
                          tensor_split_idx[key], fmt='%d')
        return tensor_split_idx

    dataset.load_fixed_splits = load_fixed_splits
    return dataset


def load_papers100M(data_dir):
    ogb_dataset = PygNodePropPredDataset('ogbn-papers100M', root=data_dir)
    ogb_data = ogb_dataset[0]
    dataset = NCDataset('ogbn-papers100M')
    dataset.graph = {
        'edge_index': torch.as_tensor(ogb_data.edge_index),
        'node_feat': torch.as_tensor(ogb_data.x),
        'num_nodes': ogb_data.num_nodes,
    }
    split_idx = ogb_dataset.get_idx_split()
    dataset.label = torch.as_tensor(ogb_data.y.data, dtype=int).reshape(-1, 1)

    def get_idx_split():
        return {k: split_idx[k] for k in ['train', 'valid', 'test']}

    dataset.load_fixed_splits = get_idx_split
    return dataset


def load_proteins_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=os.path.join(data_dir, 'ogb'))
    dataset = NCDataset('ogbn-proteins')

    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}
    dataset.load_fixed_splits = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)
    edge_index_ = to_sparse_tensor(dataset.graph['edge_index'],
                                    dataset.graph['edge_feat'], dataset.graph['num_nodes'])
    dataset.graph['node_feat'] = edge_index_.mean(dim=1)
    dataset.graph['edge_feat'] = None
    return dataset


def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=os.path.join(data_dir, 'ogb'))
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        return {key: torch.as_tensor(split_idx[key]) for key in split_idx}
    dataset.load_fixed_splits = ogb_idx_to_tensor
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset


def load_pokec_mat(data_dir):
    # Requires pokec.mat or edge_index.npy/node_feat.npy/label.npy
    pokec_dir = os.path.join(data_dir, 'pokec')
    if os.path.exists(os.path.join(pokec_dir, 'pokec.mat')):
        fulldata = scipy.io.loadmat(os.path.join(pokec_dir, 'pokec.mat'))
        edge_index = fulldata['edge_index']
        node_feat = fulldata['node_feat']
        label = fulldata['label']
    else:
        edge_index = np.load(os.path.join(pokec_dir, 'edge_index.npy'))
        node_feat = np.load(os.path.join(pokec_dir, 'node_feat.npy'))
        label = np.load(os.path.join(pokec_dir, 'label.npy'))

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat).float()
    num_nodes = int(node_feat.shape[0])
    dataset.graph = {'edge_index': edge_index, 'edge_feat': None,
                     'node_feat': node_feat, 'num_nodes': num_nodes}
    label = torch.tensor(label).flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        split_dir = os.path.join(pokec_dir, 'split_0.5_0.25')
        tensor_split_idx = {}
        if os.path.exists(split_dir):
            for key in ['train', 'valid', 'test']:
                tensor_split_idx[key] = torch.as_tensor(
                    np.loadtxt(os.path.join(split_dir, f'pokec_{key}.txt')), dtype=torch.long)
        else:
            os.makedirs(split_dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] = \
                rand_train_test_idx(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            for key in tensor_split_idx:
                np.savetxt(os.path.join(split_dir, f'pokec_{key}.txt'),
                          tensor_split_idx[key], fmt='%d')
        return tensor_split_idx

    dataset.load_fixed_splits = load_fixed_splits
    return dataset


def load_snap_patents_mat(data_dir, nclass=5):
    fulldata = scipy.io.loadmat(os.path.join(data_dir, 'snap_patents.mat'))
    dataset = NCDataset('snap_patents')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat'].todense(), dtype=torch.float)
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index, 'edge_feat': None,
                     'node_feat': node_feat, 'num_nodes': num_nodes}
    years = fulldata['years'].flatten()
    label = even_quantile_labels(years, nclass, verbose=False)
    dataset.label = torch.tensor(label, dtype=torch.long)
    return dataset


def load_yelpchi_dataset(data_dir):
    fulldata = scipy.io.loadmat(os.path.join(data_dir, 'YelpChi.mat'))
    A = fulldata['homo']
    edge_index = np.array(A.nonzero())
    node_feat = fulldata['features']
    label = np.array(fulldata['label'], dtype=np.int64).flatten()
    num_nodes = node_feat.shape[0]
    dataset = NCDataset('YelpChi')
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    node_feat = torch.tensor(node_feat.todense(), dtype=torch.float)
    dataset.graph = {'edge_index': edge_index, 'node_feat': node_feat,
                     'edge_feat': None, 'num_nodes': num_nodes}
    dataset.label = torch.tensor(label, dtype=torch.long)
    return dataset


def load_planetoid_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=os.path.join(data_dir, 'Planetoid'),
                              name=name, transform=transform)
    data = torch_dataset[0]
    dataset = NCDataset(name)
    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]
    dataset.graph = {'edge_index': data.edge_index, 'node_feat': data.x,
                     'edge_feat': None, 'num_nodes': data.num_nodes}
    dataset.label = data.y
    return dataset


def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        torch_dataset = Amazon(root=os.path.join(data_dir, 'Amazon'),
                               name='Photo', transform=transform)
    else:
        torch_dataset = Amazon(root=os.path.join(data_dir, 'Amazon'),
                               name='Computers', transform=transform)
    data = torch_dataset[0]
    dataset = NCDataset(name)
    dataset.graph = {'edge_index': data.edge_index, 'node_feat': data.x,
                     'edge_feat': None, 'num_nodes': data.num_nodes}
    dataset.label = data.y
    return dataset


def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        torch_dataset = Coauthor(root=os.path.join(data_dir, 'Coauthor'),
                                 name='CS', transform=transform)
    else:
        torch_dataset = Coauthor(root=os.path.join(data_dir, 'Coauthor'),
                                 name='Physics', transform=transform)
    data = torch_dataset[0]
    dataset = NCDataset(name)
    dataset.graph = {'edge_index': data.edge_index, 'node_feat': data.x,
                     'edge_feat': None, 'num_nodes': data.num_nodes}
    dataset.label = data.y
    return dataset


def load_geom_gcn_dataset(data_dir, name):
    graph_adj_path = os.path.join(data_dir, f'geom-gcn/{name}/out1_graph_edges.txt')
    graph_node_path = os.path.join(data_dir, f'geom-gcn/{name}/out1_node_feature_label.txt')

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_path) as f:
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
        with open(graph_node_path) as f:
            f.readline()
            for line in f:
                line = line.rstrip().split('\t')
                assert len(line) == 3
                assert int(line[0]) not in graph_node_features_dict
                graph_node_features_dict[int(line[0])] = np.array(
                    line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adj_path) as f:
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
    features = np.array([feat for _, feat in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array([lbl for _, lbl in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    def preprocess_features(feat):
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(feat)

    features = preprocess_features(features)
    edge_index = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)
    dataset.graph = {'edge_index': edge_index, 'node_feat': node_feat,
                     'edge_feat': None, 'num_nodes': num_nodes}
    dataset.label = labels
    return dataset


def papers100M_sub(data_dir):
    data_path = os.path.join(data_dir, 'ogbn_papers100M', 'subgraph.pt')
    num_nodes = 1000000
    ogb_dataset = PygNodePropPredDataset('ogbn-papers100M', root=data_dir)
    ogb_data = ogb_dataset[0]
    edge_index = torch.as_tensor(ogb_data.edge_index)
    node_feat = torch.as_tensor(ogb_data.x)
    total_nodes = ogb_data.num_nodes
    node_labels = torch.as_tensor(ogb_data.y.data, dtype=int).reshape(-1, 1)
    split_idx = ogb_dataset.get_idx_split()

    train_idx_i = split_idx['train'][split_idx['train'] < num_nodes]
    valid_idx_i = split_idx['valid'][split_idx['valid'] < num_nodes]
    test_idx_i = split_idx['test'][split_idx['test'] < num_nodes]
    split_all = torch.cat([train_idx_i, valid_idx_i, test_idx_i])
    split_len = split_all.shape[0]
    train_num, valid_num = int(split_len * 0.7), int(split_len * 0.1)
    train_idx_i, valid_idx_i = split_all[:train_num], split_all[train_num:train_num + valid_num]
    test_idx_i = split_all[train_num + valid_num:]

    idx_i = torch.arange(num_nodes)
    if os.path.exists(data_path):
        edge_index_i = torch.load(data_path)
    else:
        edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=total_nodes, relabel_nodes=False)

    x_i, y_i = node_feat[:num_nodes], node_labels[:num_nodes]
    dataset = NCDataset('ogbn-papers100M')
    dataset.graph = {'edge_index': edge_index_i, 'node_feat': x_i,
                     'num_nodes': num_nodes}
    dataset.label = y_i

    def get_idx_split():
        return {'train': train_idx_i, 'valid': valid_idx_i, 'test': test_idx_i}

    dataset.load_fixed_splits = get_idx_split

    folder = os.path.join(data_dir, 'ogbn_papers100M')
    os.makedirs(folder, exist_ok=True)
    if not os.path.exists(data_path):
        torch.save(edge_index_i, data_path)
    return dataset
