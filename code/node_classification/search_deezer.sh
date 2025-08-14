export OMP_NUM_THREADS=8


for DATASET in deezer-search
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
                python main.py --backbone gcn --rand_split --dataset deezer-europe \
                --lr ${LR} --num_layers 2 \
                --hidden_channels ${dim} --weight_decay 5e-05 \
                --dropout ${dropout} --ours_weight_decay 0.\
                --method GPR --ours_layers 1 --use_graph --num_heads 1 --ours_use_residual \
                --alpha 0.5  --device 3 --runs 5 --init_alpha 0.7 >> results/${DATASET}.txt
            done
        done
    done
done