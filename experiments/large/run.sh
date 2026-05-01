#!/bin/bash
# Reproduce ParaFormer results on large graph benchmarks.
# Hyperparameters are being finalized and will be released soon.
# Adjust --device and --data_dir as needed.

DATA_DIR=../../../data

echo "Hyperparameters for reproducing paper results are being finalized."
echo "Please check back soon or refer to the paper for details."
echo ""
echo "Example command structure:"
echo "  python main.py --data_dir \$DATA_DIR --method paraformer --dataset <dataset> \\"
echo "      --hidden_channels <H> --K_transformer <K> --init_alpha <A> \\"
echo "      --trans_num_layers <L> --gnn_num_layers <L> \\"
echo "      --use_graph --graph_weight <W> \\"
echo "      --seed 123 --runs 5 --device 0"
