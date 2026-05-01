import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class TransConvLayer(nn.Module):
    """Single layer of GPR polynomial global attention."""

    def __init__(self, K_transformer, alpha, hidden_channels, dropout):
        super().__init__()
        self.query = nn.Linear(hidden_channels, hidden_channels)
        self.key = nn.Linear(hidden_channels, hidden_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        self.K_transformer = K_transformer
        self.alpha = alpha
        TEMP = alpha * (1 - alpha) ** np.arange(self.K_transformer + 1)
        TEMP[-1] = (1 - alpha) ** self.K_transformer
        self.temp = Parameter(torch.tensor(TEMP))
        self.dropout = dropout

    def attention(self, query, key, value):
        """Compute K-step polynomial propagation via kernel trick.

        Args:
            query, key, value: [N, d] tensors
        Returns:
            zs: list of [N, d] propagated features at each step
        """
        query = F.softmax(query, dim=1)
        key = F.softmax(key, dim=0).transpose(-1, -2)
        kv = torch.einsum('ik,kj->ij', key, value)
        kq = torch.einsum('ik,kj->ij', key, query)
        kq_vs = [kv]
        zs = []
        for i in range(1, self.K_transformer + 1):
            zs.append(torch.einsum('ik,kj->ij', query, kq_vs[-1]))
            kq_vs.append(torch.einsum('ik,kj->ij', kq_vs[-1], kq))
        return zs

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        zs = self.attention(q, k, v)
        v = F.relu(v)
        v = F.dropout(v, p=self.dropout, training=self.training)
        hidden = v * self.temp[0]
        for k in range(self.K_transformer):
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * F.dropout(zs[k], p=self.dropout, training=self.training)
        return hidden

    def reset_parameters(self):
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K_transformer + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K_transformer


class TransConv(nn.Module):
    """Stacked TransConvLayers forming the global attention branch."""

    def __init__(self, K_transformer, alpha, in_channels, hidden_channels,
                 num_layers, dropout, use_bn=True):
        super().__init__()
        self.fc = nn.Linear(in_channels, hidden_channels)
        self.dropout = dropout
        self.use_bn = use_bn
        self.alpha_res = 0.5

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))

        for _ in range(num_layers):
            self.convs.append(
                TransConvLayer(K_transformer=K_transformer, alpha=alpha,
                               hidden_channels=hidden_channels, dropout=dropout))
            self.bns.append(nn.LayerNorm(hidden_channels))

    def forward(self, x):
        x = self.fc(x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_prev = x
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = self.alpha_res * x + (1 - self.alpha_res) * layer_prev
            if self.use_bn:
                x = self.bns[i + 1](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_prev = x

        return x

    def reset_parameters(self):
        self.fc.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
