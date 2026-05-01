# ParaFormer: A Generalized PageRank Graph Transformer for Graph Representation Learning

This repository provides the official implementation of **ParaFormer**, a scalable graph Transformer that leverages Generalized PageRank (GPR) attention to efficiently model all-pair node interactions with linear O(N) complexity.

## Overview

ParaFormer combines two complementary components:

1. **GPR Polynomial Global Attention** — Computes all-pair node interactions via a polynomial expansion of the kernelized attention. Using learnable propagation coefficients (initialized as Personalized PageRank weights), it captures long-range dependencies with O(N) time and memory.

2. **Local GNN Encoder** — A standard GCN-based message-passing branch that captures the local graph topology.

The two branches are combined via a weighted sum: `output = graph_weight * GNN(x) + (1-graph_weight) * Trans(x)`.

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- PyG >= 2.0.0
- torch_sparse, torch_scatter

Install dependencies:

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, follow the official [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

## Package Structure

```
ParaFormer/
├── models/
│   ├── paraformer.py         # ParaFormer model
│   ├── gpr_attention.py      # GPR polynomial global attention
│   └── gnn_encoder.py        # Local GNN encoder
├── utils/
│   ├── data_utils.py         # Data splitting and evaluation utilities
│   ├── dataset_medium.py     # Medium graph dataset loading
│   ├── dataset_large.py      # Large graph dataset loading
│   └── logger.py             # Training logger
├── experiments/
│   ├── medium/
│   │   ├── main.py           # Training script for medium graphs
│   │   ├── parse.py          # Argument parsing
│   │   └── run.sh            # Reproduce results
│   └── large/
│       ├── main.py           # Training script for large graphs
│       ├── main-batch.py     # Mini-batch variant
│       ├── parse.py          # Argument parsing
│       └── run.sh            # Reproduce results
├── data/                     # Place datasets here
├── requirements.txt
└── README.md
```

## Quick Start

### Prepare Datasets

See [data/README.md](data/README.md) for dataset download links.

> **Note:** The exact hyperparameters for reproducing paper results are being finalized and will be released soon. Please refer to the paper for details on experimental settings.

### Run Medium Graph Experiments

```bash
cd experiments/medium
bash run.sh
```

### Run Large Graph Experiments

```bash
cd experiments/large
bash run.sh
```

### Mini-batch Training for Very Large Graphs

```bash
cd experiments/large
python main-batch.py --data_dir ../../../data \
    --method paraformer --dataset <dataset> --metric acc \
    --hidden_channels 256 --use_graph --graph_weight 0.5 \
    --gnn_num_layers 3 --gnn_use_bn --gnn_use_residual --gnn_use_weight --gnn_use_act \
    --trans_num_layers 1 --trans_use_bn --trans_use_residual --trans_use_weight \
    --seed 123 --runs 5 --device 0 --batch_size 10000
```

## Model Usage

```python
from models.paraformer import ParaFormer

model = ParaFormer(
    in_channels=1433,        # Input feature dimension
    out_channels=7,          # Number of classes
    hidden_channels=256,     # Hidden dimension
    K_transformer=10,        # Polynomial order (GPR steps)
    init_alpha=0.3,          # Initial PPR teleport probability
    trans_num_layers=1,      # Number of global attention layers
    gnn_num_layers=3,        # Number of GNN layers
    use_graph=True,          # Use local graph topology
    graph_weight=0.8,        # GNN vs Transformer weight (0~1)
)

# Forward pass
logits = model(x, edge_index)
```

## Key Arguments

### GPR Attention
- `--K_transformer` (int, default=10): Number of polynomial propagation steps. Larger = more global receptive field.
- `--init_alpha` (float, default=0.3): Initial PPR teleport probability. Coefficients are learnable and will be tuned during training.
- `--trans_num_layers` (int, default=1): Stack depth of global attention layers.

### Local GNN
- `--gnn_num_layers` (int, default=3): Number of GCN layers for the local branch.
- `--gnn_use_residual`: Enable residual connections.
- `--gnn_use_bn`: Enable batch normalization.

### Aggregation
- `--graph_weight` (float, default=0.8): Blend ratio. `graph_weight * GNN + (1-graph_weight) * Transformer`.
- `--aggregate` (str, default='add'): 'add' for weighted sum or 'cat' for concatenation.

## Citation

If you find this code useful, please consider citing our work:

```bibtex
@inproceedings{yuan2026paraformer,
  title={ParaFormer: A Generalized PageRank Graph Transformer for Graph Representation Learning},
  author={Yuan, Chaohao and Song, Zhenjie and Kuruoglu, Ercan Engin and Zhao, Kangfei and Liu, Yang and Zhao, Deli and Cheng, Hong and Rong, Yu},
  booktitle={Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  pages={881--891},
  year={2026}
}
```

## Acknowledgments

This codebase builds upon [SGFormer](https://github.com/qitianwu/SGFormer) and [DIFFormer](https://github.com/qitianwu/DIFFormer).
