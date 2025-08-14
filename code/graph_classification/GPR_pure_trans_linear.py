#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from scipy.sparse import coo_matrix, csr_matrix




class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = 0.5 * x + 0.5 * layer_[-1]
        return x


class TransConvLayer(torch.nn.Module):
    def __init__(self, K_transformer, alpha, input_dim, hidden_channels, dropout):
        super(TransConvLayer, self).__init__()
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        self.value2 = nn.Linear(hidden_channels, hidden_channels)
        self.fc = nn.Linear(input_dim, hidden_channels)
        self.K_transformer = K_transformer
        self.alpha = alpha
        TEMP = alpha * (1 - alpha) ** np.arange(self.K_transformer + 1)
        TEMP[-1] = (1 - alpha) ** self.K_transformer
        self.temp = Parameter(torch.Tensor(TEMP))
        # self.temp = torch.Tensor(TEMP)
        print(self.temp)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = dropout
        

    def attention(self, query, key, value, K=10):
        # q, k, v [N, d]
        # K, int, page
        query = F.softmax(query, dim=1)
        key = F.softmax(key, dim=0).transpose(-1, -2)
        N = query.shape[0]
        kv = torch.einsum('ik,kj->ij', key, value)
        kq = torch.einsum('ik,kj->ij', key, query)
        kq_vs = [kv]
        zs = []
        for i in range(1, K+1):
            zs.append(torch.einsum('ik,kj->ij',query, kq_vs[-1]))
            kq_vs.append(torch.einsum('ik,kj->ij', kq_vs[-1], kq))
        return zs


    def forward(self, x, edge_index=None, edge_weight=None, output_attn=False):
        q = self.query(x)
        k = self.key(x)
        # v = self.value(x)
        v = x
        zs = self.attention(q, k, v, self.K_transformer)
        # v = F.relu(v)
        # v = F.dropout(v, p=self.dropout, training=self.training)
        hidden = v * self.temp[0]
        for k in range(self.K_transformer):
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * F.dropout(zs[k], p=self.dropout, training=self.training)
        return hidden

    def reset_parameters(self):
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.value2.reset_parameters()
        self.fc.reset_parameters()
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K_transformer+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K_transformer


class TransConv(nn.Module):
    def __init__(self, K_transformer, alpha, input_dim, hidden_channels, num_layers, dropout, trans_use_bn, use_ffn):
        super(TransConv, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_channels)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = dropout
        self.use_bn = trans_use_bn
        self.num_layers = num_layers
        # self.bns = nn.ModuleList()
        # self.bns.append(nn.LayerNorm(hidden_channels))
        self.use_ffn = use_ffn
        # K_transformer, alpha, input_dim, hidden_channels, dropout
        self.alpha = 0.5
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(K_transformer=K_transformer, alpha=alpha, input_dim=hidden_channels, hidden_channels=hidden_channels, dropout=dropout))
            # self.bns.append(nn.LayerNorm(hidden_channels))



    def forward(self, data, edge_index, edge_weight=None):
        layer_ = []
        x = self.fc(data)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, x, edge_index, edge_weight)
            x = self.alpha * x + (1 - self.alpha) * layer_[i]# + 0.5*layer_[0]
            
            # if self.use_bn:
                # x = self.bns[i+1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
        return x
        # return layer_


    def reset_parameters(self):
        self.fc.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        # for bn in self.bns:
            # bn.reset_parameters()



class GPR_Transformer(nn.Module):
    def __init__(self, K_transformer, init_alpha, in_channels, out_channels, hidden_channels, num_layers, dropout, use_ffn=False, use_bn=False,
                 graph_weight=0.8, gnn=None, use_graph=True, aggregate='add'):
        super(GPR_Transformer, self).__init__()
        # K_transformer, alpha, input_dim, hidden_channels, num_layers, dropout, trans_use_bn, use_ffn
        self.spectral_attention = TransConv(K_transformer=K_transformer, alpha=init_alpha, input_dim=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, trans_use_bn=use_bn, use_ffn=use_ffn)
        # self.spectral_attention2 = TransConv(K_transformer=K_transformer, alpha=init_alpha, input_dim=hidden_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, trans_use_bn=use_bn, use_ffn=use_ffn)

        self.use_graph = use_graph
        # self.fc = nn.Linear(hidden_channels, out_channels)
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.fc=nn.Linear(hidden_channels,out_channels)
        self.graph_conv = gnn
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.params1=list(self.spectral_attention.parameters())
        self.params2=list(self.graph_conv.parameters()) if gnn is not None else []
        self.params2.extend(list(self.fc.parameters()) )


    def forward(self, x, edge_index):
        # x, edge_index = data.graph['node_feat'], data.graph['edge_index']
        node_n = x.shape[0]
        out = self.spectral_attention(x, edge_index)
        # out = self.spectral_attention2(out, edge_index)
        # print(edge_index[:0])
        # print(x.shape)
        # print(out.shape)
        # exit()

        if self.use_graph:
            gcn = self.graph_conv(x, edge_index)
            # gcn = self.graph_conv(data)
            # print(gcn.shape)
            # print(out.shape)
            # exit()
            out = self.fc(out)
            out = self.graph_weight * gcn + (1 - self.graph_weight) * out
        else:
            
            out = self.fc(out)
        return out


    def reset_parameters(self):
        self.spectral_attention.reset_parameters()
        self.fc.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()