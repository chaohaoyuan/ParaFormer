export OMP_NUM_THREADS=8


for DATASET in film-gcn-search
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
            for dropout in 0.3 0.4 0.5 0.6 0.7
            do
                echo "DP=${dropout}"
                echo "DP=${dropout}" >> results/${DATASET}.txt
                python main.py --backbone gcn --dataset film --lr ${LR} --num_layers 2 \
                --hidden_channels ${dim} --weight_decay 0.001 --dropout 0.6   \
                --method GPR --ours_layers 1 --use_graph --graph_weight 0.5 --num_heads 1 --ours_use_residual --ours_use_act \
                --alpha 0.5 --ours_dropout ${dropout} --ours_weight_decay 0.001 --device 3 --runs 10 --epochs 200 --init_alpha 0.9  >> results/${DATASET}.txt
            done
        done
    done
done