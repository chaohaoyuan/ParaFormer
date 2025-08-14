export OMP_NUM_THREADS=8


for DATASET in chameleon-gcn-search
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
                python main.py --backbone gcn --dataset chameleon --lr ${LR} --num_layers 2 \
                --hidden_channels ${dim} --ours_layers 1 --weight_decay 0.001 --dropout ${dropout} \
                --method GPR --num_heads 1 --ours_use_residual --use_graph\
                --alpha 0.5  --device 6 --runs 10 --epochs 200 --ours_weight_decay 0.001 >> results/${DATASET}.txt
            done
        done
    done
done