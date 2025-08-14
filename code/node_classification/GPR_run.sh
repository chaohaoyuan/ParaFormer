


# for ab study
python main.py --backbone gcn --dataset cora --lr 0.01 --num_layers 2 \
    --hidden_channels 32 --weight_decay 5e-4 --dropout 0.5 \
    --method GPR --ours_layers 1 --graph_weight 0.7 \
    --ours_dropout 0.5 --use_residual --alpha 0.5 --ours_weight_decay 0.001 \
    --no_feat_norm --rand_split --train_prop 0.6 --valid_prop 0.2\
    --seed 123 --device 1 --runs 5 --use_graph --init_alpha 0.1

python main.py --backbone gcn --dataset citeseer --lr 0.005 --num_layers 2 \
    --hidden_channels 256 --weight_decay 0.01 --dropout 0.5 \
    --method GPR --ours_layers 1 --graph_weight 0.5 --use_graph\
    --ours_dropout 0.7 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
    --rand_split --train_prop 0.6 --valid_prop 0.2 --no_feat_norm \
    --seed 123 --device 4 --runs 5 --init_alpha 0.1

# for ab study
python main.py --backbone gcn --dataset citeseer --lr 0.005 --num_layers 2 \
    --hidden_channels 128 --weight_decay 0.01 --dropout 0.5 \
    --method GPR --ours_layers 1 --graph_weight 0.5 --use_graph\
    --ours_dropout 0.7 --use_residual --alpha 0.5 --ours_weight_decay 0.01 \
    --rand_split --train_prop 0.6 --valid_prop 0.2 --no_feat_norm \
    --seed 123 --device 4 --runs 5 --init_alpha 0.1


python main.py --backbone gcn --dataset pubmed --lr 0.005 --num_layers 2 \
    --hidden_channels 256 --weight_decay 5e-4 --dropout 0.5 \
    --method GPR --ours_layers 1 --use_graph --graph_weight 0.8 \
    --ours_dropout 0.6 --use_residual --alpha 0.5 --ours_weight_decay 0.001  \
    --rand_split --train_prop 0.6 --valid_prop 0.2 --no_feat_norm \
    --seed 123 --device 0 --runs 5 --init_alpha 0.1

python main.py --backbone gcn --dataset film --lr 0.1 --num_layers 2 \
    --hidden_channels 96 --weight_decay 0.001 --dropout 0.6   \
    --method GPR --ours_layers 1 --use_graph --graph_weight 0.5 --num_heads 1 --ours_use_residual --ours_use_act \
    --alpha 0.5 --ours_dropout 0.3 --ours_weight_decay 0.001 --device 3 --runs 10 --epochs 200 --init_alpha 0.9

python main.py --backbone gcn  --dataset squirrel --lr 0.005 --num_layers 5 \
    --hidden_channels 64 --weight_decay 5e-4 --ours_weight_decay 5e-4 --dropout 0.6   \
    --method GPR --ours_layers 1 --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
    --alpha 0.5  --device 7 --runs 10 --epoch 1000 --init_alpha 0.1 --patience 200


python main.py --backbone gcn --dataset chameleon --lr 0.005 --num_layers 2 \
    --hidden_channels 64 --ours_layers 1 --weight_decay 0.001 --dropout 0.5 \
    --method GPR --num_heads 1 --ours_use_residual --use_graph\
    --alpha 0.5  --device 6 --runs 10 --epochs 200 --ours_weight_decay 0.001

python main.py --backbone gcn --rand_split --dataset deezer-europe \
    --lr 0.05 --num_layers 2 \
    --hidden_channels 128 --weight_decay 5e-05 \
    --dropout 0.7 --ours_weight_decay 0.\
    --method GPR --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
    --alpha 0.5  --device 3 --runs 5 --init_alpha 0.7
