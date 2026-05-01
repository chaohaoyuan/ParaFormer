import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpr_attention import TransConv
from .gnn_encoder import GraphConv


class ParaFormer(nn.Module):
    """ParaFormer: Generalized PageRank Polynomial Graph Transformer.

    Combines:
    - GPR polynomial global attention (all-pair interactions, O(N) complexity)
    - Local GNN encoder (graph convolution on input topology)

    The two branches are combined via weighted sum:
        output = graph_weight * GNN(x, edge_index) + (1 - graph_weight) * Trans(x)
    """

    def __init__(self, in_channels, out_channels,
                 # Global attention (Transformer) branch
                 hidden_channels=256,
                 K_transformer=10,
                 init_alpha=0.3,
                 trans_num_layers=1,
                 trans_dropout=0.5,
                 trans_use_bn=True,
                 trans_use_weight=True,
                 trans_use_act=True,
                 # Local GNN branch
                 gnn_num_layers=3,
                 gnn_dropout=0.5,
                 gnn_use_weight=True,
                 gnn_use_init=False,
                 gnn_use_bn=True,
                 gnn_use_residual=True,
                 gnn_use_act=True,
                 # Aggregation
                 use_graph=True,
                 graph_weight=0.8,
                 aggregate='add'):
        super().__init__()

        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate

        # Global attention branch
        self.trans_conv = TransConv(
            K_transformer=K_transformer,
            alpha=init_alpha,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=trans_num_layers,
            dropout=trans_dropout,
            use_bn=trans_use_bn,
        )

        # Local GNN branch
        self.graph_conv = GraphConv(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            use_bn=gnn_use_bn,
            use_residual=gnn_use_residual,
            use_weight=gnn_use_weight,
            use_init=gnn_use_init,
            use_act=gnn_use_act,
        )

        # Output projection
        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type: {aggregate}')

        # Parameter groups for separate weight decay
        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters())
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        """Forward pass.

        Args:
            x: node features [N, in_channels]
            edge_index: graph connectivity [2, E]
        Returns:
            logits [N, out_channels]
        """
        x_trans = self.trans_conv(x)

        if self.use_graph:
            x_gnn = self.graph_conv(x, edge_index)
            if self.aggregate == 'add':
                x = self.graph_weight * x_gnn + (1 - self.graph_weight) * x_trans
            else:
                x = torch.cat((x_trans, x_gnn), dim=1)
        else:
            x = x_trans

        x = self.fc(x)
        return x

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()
        self.fc.reset_parameters()
