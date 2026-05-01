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
from torch_geometric.utils import to_undirected, subgraph

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


def eval_batch_acc(true, pred):
    pred = torch.max(pred, dim=1, keepdim=True)[1]
    true_cnt = (true == pred).sum()
    return true.shape[0], true_cnt.item()


@torch.no_grad()
def evaluate_batch(model, dataset, split_idx, args, device, n, true_label):
    num_batch = n // args.batch_size + 1
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    model.to(device)
    model.eval()

    idx = torch.randperm(n)
    train_total, train_correct = 0, 0
    valid_total, valid_correct = 0, 0
    test_total, test_correct = 0, 0

    for i in range(num_batch):
        idx_i = idx[i * args.batch_size:(i + 1) * args.batch_size]
        x_i = x[idx_i].to(device)
        edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
        edge_index_i = edge_index_i.to(device)
        y_i = true_label[idx_i].to(device)

        out_i = model(x_i, edge_index_i)

        t, c = eval_batch_acc(y_i[train_mask[idx_i]], out_i[train_mask[idx_i]])
        train_total += t
        train_correct += c
        t, c = eval_batch_acc(y_i[valid_mask[idx_i]], out_i[valid_mask[idx_i]])
        valid_total += t
        valid_correct += c
        t, c = eval_batch_acc(y_i[test_mask[idx_i]], out_i[test_mask[idx_i]])
        test_total += t
        test_correct += c

    train_acc = train_correct / train_total if train_total > 0 else 0
    valid_acc = valid_correct / valid_total if valid_total > 0 else 0
    test_acc = test_correct / test_total if test_total > 0 else 0
    return train_acc, valid_acc, test_acc, 0


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
    parser = argparse.ArgumentParser(description='ParaFormer Large Graph Batch Training')
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

    # Preprocess
    if not args.directed and args.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.label = dataset.label.to(device)
    n = dataset.graph['num_nodes']
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]
    print(f'num nodes {n} | num classes {c} | num node feats {d}')

    # Prepare label tensor
    multilabel_datasets = ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins')
    if args.dataset in multilabel_datasets:
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
    else:
        true_label = dataset.label

    # Model
    model = parse_method(args, c, d, device)
    print(f'# params: {count_parameters(model)}')

    criterion = nn.NLLLoss()

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.params1, 'weight_decay': args.trans_weight_decay},
        {'params': model.params2, 'weight_decay': args.gnn_weight_decay},
    ], lr=args.lr)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train']

        model.reset_parameters()
        best_val = float('-inf')
        patience = 0

        for epoch in range(args.epochs):
            model.train()

            # Mini-batch training
            num_batch = n // args.batch_size + 1
            idx = torch.randperm(n)

            for i in range(num_batch):
                optimizer.zero_grad()

                idx_i = idx[i * args.batch_size:(i + 1) * args.batch_size]
                x_i = dataset.graph['node_feat'][idx_i].to(device)
                edge_index_i, _ = subgraph(idx_i, dataset.graph['edge_index'],
                                           num_nodes=n, relabel_nodes=True)
                edge_index_i = edge_index_i.to(device)

                out_i = model(x_i, edge_index_i)

                train_mask_i = torch.zeros(idx_i.shape[0], dtype=torch.bool)
                for j, tid in enumerate(idx_i):
                    if tid in train_idx:
                        train_mask_i[j] = True

                if train_mask_i.sum() > 0:
                    out_log = F.log_softmax(out_i, dim=1)
                    loss = criterion(out_log[train_mask_i],
                                     true_label.squeeze(1)[idx_i][train_mask_i].to(device))
                    loss.backward()
                    optimizer.step()

            if epoch % args.eval_step == 0:
                train_acc, valid_acc, test_acc, _ = evaluate_batch(
                    model, dataset, split_idx, args, device, n, true_label)
                logger.add_result(run, [train_acc, valid_acc, test_acc, 0])

                if valid_acc > best_val:
                    best_val = valid_acc
                    patience = 0
                else:
                    patience += 1

                if epoch % args.display_step == 0:
                    print(f'Epoch: {epoch:02d}, '
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
