import argparse
import sys
import os.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.paraformer import ParaFormer


def parse_method(args, num_classes, num_features, device):
    if args.method == 'paraformer':
        model = ParaFormer(
            in_channels=num_features,
            out_channels=num_classes,
            hidden_channels=args.hidden_channels,
            K_transformer=args.K_transformer,
            init_alpha=args.init_alpha,
            trans_num_layers=args.trans_num_layers,
            trans_dropout=args.trans_dropout,
            trans_use_bn=args.trans_use_bn,
            trans_use_weight=args.trans_use_weight,
            trans_use_act=args.trans_use_act,
            gnn_num_layers=args.gnn_num_layers,
            gnn_dropout=args.gnn_dropout,
            gnn_use_weight=args.gnn_use_weight,
            gnn_use_init=args.gnn_use_init,
            gnn_use_bn=args.gnn_use_bn,
            gnn_use_residual=args.gnn_use_residual,
            gnn_use_act=args.gnn_use_act,
            use_graph=args.use_graph,
            graph_weight=args.graph_weight,
            aggregate=args.aggregate,
        ).to(device)
    else:
        raise ValueError(f'Invalid method: {args.method}')
    return model


def parser_add_main_args(parser):
    # Dataset and evaluation
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--train_prop', type=float, default=.5)
    parser.add_argument('--valid_prop', type=float, default=.25)
    parser.add_argument('--protocol', type=str, default='semi',
                        help='protocol for cora datasets: semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true',
                        help='use random splits with fixed labels per class')
    parser.add_argument('--label_num_per_class', type=int, default=20)
    parser.add_argument('--valid_num', type=int, default=500)
    parser.add_argument('--test_num', type=int, default=500)
    parser.add_argument('--metric', type=str, default='acc',
                        choices=['acc', 'rocauc', 'f1_micro', 'f1_macro'])

    # Method
    parser.add_argument('--method', type=str, default='paraformer')

    # ParaFormer: common
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--use_graph', action='store_true', help='use input graph topology')
    parser.add_argument('--aggregate', type=str, default='add', help='add or cat')
    parser.add_argument('--graph_weight', type=float, default=0.8)

    # ParaFormer: GPR global attention
    parser.add_argument('--K_transformer', type=int, default=10)
    parser.add_argument('--init_alpha', type=float, default=0.3)
    parser.add_argument('--trans_num_layers', type=int, default=1)
    parser.add_argument('--trans_dropout', type=float, default=0.5)
    parser.add_argument('--trans_use_weight', action='store_true')
    parser.add_argument('--trans_use_bn', action='store_true')
    parser.add_argument('--trans_use_residual', action='store_true')
    parser.add_argument('--trans_use_act', action='store_true')

    # ParaFormer: local GNN
    parser.add_argument('--gnn_num_layers', type=int, default=3)
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--gnn_use_weight', action='store_true')
    parser.add_argument('--gnn_use_init', action='store_true')
    parser.add_argument('--gnn_use_bn', action='store_true')
    parser.add_argument('--gnn_use_residual', action='store_true')
    parser.add_argument('--gnn_use_act', action='store_true')

    # Training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-3)
    parser.add_argument('--trans_weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=200)

    # Display
    parser.add_argument('--display_step', type=int, default=50)
    parser.add_argument('--no_feat_norm', action='store_true')
