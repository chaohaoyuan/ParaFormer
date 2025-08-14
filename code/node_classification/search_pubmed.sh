export OMP_NUM_THREADS=8


for DATASET in pubmed-gcn-search
do
    echo "DATASET=${DATASET}"
    for LR in 0.005 0.01 0.05 0.1
    do
        echo "LR=${LR}"
        echo "LR=${LR}" >> results/${DATASET}.txt
        for dim in 64 96 128 256
        do
            echo "dim=${dim}"
            echo "dim=${dim}" >> results/${DATASET}.txt
            for dropout in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
            do
                echo "DP=${dropout}"
                echo "DP=${dropout}" >> results/${DATASET}.txt
                python main.py --backbone gcn --dataset pubmed --lr ${LR} --num_layers 2 \
                    --hidden_channels ${dim} --weight_decay 5e-4 --dropout 0.5 \
                    --method GPR --ours_layers 1 --use_graph --graph_weight 0.8 \
                    --ours_dropout ${dropout} --use_residual --alpha 0.5 --ours_weight_decay 0.001  \
                    --rand_split --train_prop 0.6 --valid_prop 0.2 --no_feat_norm \
                    --seed 123 --device 0 --runs 5 --init_alpha 0.0 >> results/${DATASET}.txt
            done
        done
    done
done