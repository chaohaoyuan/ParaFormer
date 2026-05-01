import argparse
import os
import random
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from utils.data_utils import eval_acc, eval_f1_macro, class_rand_splits, load_fixed_splits
from utils.dataset_medium import load_nc_dataset
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

    if args.dataset == 'deezer-europe':
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
    parser = argparse.ArgumentParser(description='ParaFormer Medium Graph Training')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    # Load dataset
    dataset = load_nc_dataset(args.data_dir, args.dataset, args.no_feat_norm)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    # Set up splits
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                         for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [class_rand_splits(
            dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
    else:
        split_idx_lst = load_fixed_splits(args.data_dir, dataset, args.dataset, args.protocol)

    # Move data to device
    if args.dataset != 'deezer-europe':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device)
    dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    print(f'num nodes {n} | num classes {c} | num node feats {d}')

    # Build model
    model = parse_method(args, c, d, device)

    if args.dataset == 'deezer-europe':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    eval_func = eval_acc if args.metric == 'acc' else eval_f1_macro

    # Optimizer with separate weight decay for transformer and GNN
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.trans_weight_decay},
        {'params': model.params2, 'weight_decay': args.weight_decay},
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

            if args.dataset == 'deezer-europe':
                true_label = (F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                              if dataset.label.shape[1] == 1 else dataset.label)
                loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
            else:
                out_log = F.log_softmax(out, dim=1)
                loss = criterion(out_log[train_idx], dataset.label.squeeze(1)[train_idx])

            loss.backward()
            optimizer.step()

            train_acc, valid_acc, test_acc, valid_loss = evaluate(
                model, dataset, split_idx, eval_func, criterion, args)
            logger.add_result(run, [train_acc, valid_acc, test_acc, valid_loss])

            if valid_acc > best_val:
                best_val = valid_acc
                patience = 0
            else:
                patience += 1
                if patience >= args.patience:
                    break

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%')

        print_run_statistics(run, logger.results[run])

    results = logger.print_statistics()
    print(results)


if __name__ == '__main__':
    main()
