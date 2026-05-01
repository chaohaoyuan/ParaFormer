import argparse
import os
import random
import sys
import time
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from utils.data_utils import (eval_acc, eval_rocauc, eval_f1_micro, eval_f1_macro,
                               load_fixed_splits, count_parameters)
from utils.dataset_large import load_dataset
from utils.logger import Logger
from parse import parse_method, parser_add_main_args

warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args):
    model.eval()
    out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    train_acc = eval_func(dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(dataset.label[split_idx['test']], out[split_idx['test']])

    multilabel_datasets = ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins')
    if args.dataset in multilabel_datasets:
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']],
                               true_label.squeeze(1)[split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(out[split_idx['valid']],
                               dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss


def print_run_statistics(run, results):
    result = 100 * torch.tensor(results)
    argmax = result[:, 1].argmax().item()
    print(f'Run {run + 1:02d}: '
          f'Highest Train: {result[:, 0].max():.2f} '
          f'Highest Valid: {result[:, 1].max():.2f} '
          f'Highest Test: {result[:, 2].max():.2f} '
          f'| Final Train: {result[argmax, 0]:.2f} '
          f'Final Test: {result[argmax, 2]:.2f}')


def main():
    parser = argparse.ArgumentParser(description='ParaFormer Large Graph Training')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    torch.cuda.set_device(args.device)

    # Load data
    dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)

    # Set up splits
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(args.runs)]
    elif args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-proteins',
                          'pokec', 'amazon2m']:
        split_idx_lst = [dataset.load_fixed_splits() for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, args.dataset, args.protocol)

    # Move data to device
    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)
    dataset.label = dataset.label.to(device)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    print(f'num nodes {n} | num classes {c} | num node feats {d}')

    # Eval metric
    metric_map = {'acc': eval_acc, 'rocauc': eval_rocauc,
                  'f1_micro': eval_f1_micro, 'f1_macro': eval_f1_macro}
    eval_func = metric_map[args.metric]

    # Loss
    multilabel_datasets = ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins')
    if args.dataset in multilabel_datasets:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    # Model
    model = parse_method(args, c, d, device)
    print(f'# params: {count_parameters(model)}')

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.trans_weight_decay},
        {'params': model.params2, 'weight_decay': args.gnn_weight_decay},
    ], lr=args.lr)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)

        model.reset_parameters()
        best_val = float('-inf')
        patience = 0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

            multilabel_datasets = ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins')
            if args.dataset in multilabel_datasets:
                true_label = (F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                              if dataset.label.shape[1] == 1 else dataset.label)
                loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
            else:
                out_log = F.log_softmax(out, dim=1)
                loss = criterion(out_log[train_idx], dataset.label.squeeze(1)[train_idx])

            loss.backward()
            optimizer.step()

            if epoch % args.eval_step == 0:
                train_acc, valid_acc, test_acc, valid_loss = evaluate(
                    model, dataset, split_idx, eval_func, criterion, args)
                logger.add_result(run, [train_acc, valid_acc, test_acc, valid_loss])

                if valid_acc > best_val:
                    best_val = valid_acc
                    patience = 0
                else:
                    patience += 1

                if epoch % args.display_step == 0:
                    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, '
                          f'Valid: {100 * valid_acc:.2f}%, '
                          f'Test: {100 * test_acc:.2f}%')

            if patience >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

        print_run_statistics(run, logger.results[run])

    results = logger.print_statistics()
    print(results)


if __name__ == '__main__':
    main()
