export OMP_NUM_THREADS=8


for DATASET in squirrel-gcn-search
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
            for dropout in 0.2 0.3 0.4 0.5 0.6 0.7
            do
                echo "DP=${dropout}"
                echo "DP=${dropout}" >> results/${DATASET}.txt
                python main.py --backbone gcn  --dataset squirrel --lr ${LR} --num_layers 5 \
                --hidden_channels ${dim} --weight_decay 5e-4 --ours_weight_decay 5e-4 --dropout ${dropout}   \
                --method GPR --ours_layers 1 --use_graph --ours_use_act --ours_use_residual --num_heads 1 \
                --alpha 0.5  --device 7 --runs 10 --epoch 1000 --init_alpha 0 --patience 200 >> results/${DATASET}.txt
            done
        done
    done
done    