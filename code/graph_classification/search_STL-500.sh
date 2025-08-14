for DATASET in STL-500
do
    echo "DATASET=${DATASET}"
    for wd in 0 0.0001 0.001 0.1
    do
        echo "WD=${wd}"
        echo "WD=${wd}" >> results/${DATASET}.txt
        for LR in 0.0005 0.005 0.01 0.05
        do
            echo "LR=${LR}"
            echo "LR=${LR}" >> results/${DATASET}.txt
            for dim in 32 64 128 256
            do
                echo "dim=${dim}"
                echo "dim=${dim}" >> results/${DATASET}.txt
                for dropout in 0 0.1 0.3 0.5 0.7
                do
                    echo "DP=${dropout}"
                    echo "DP=${dropout}" >> results/${DATASET}.txt
                    # python main.py --backbone gcn --dataset chameleon --lr ${LR} --num_layers 2 \
                    # --hidden_channels ${dim} --ours_layers 1 --weight_decay 0.001 --dropout ${dropout} \
                    # --method GPR --num_heads 1 --ours_use_residual --use_graph\
                    # --alpha 0.5  --device 6 --runs 10 --epochs 200 --ours_weight_decay 0.001
                    CUDA_VISIBLE_DEVICES=1 python main.py --dataset stl10 --method GPR --rand_split_class --label_num_per_class 50 --valid_num 1000 \
                    --lr ${LR} --weight_decay ${wd} --dropout ${dropout} --num_layers 2 --hidden_channels ${dim} --use_residual --use_bn  --alpha 0.5 --kernel simple \
                    --epochs 600 --seed 42 --device 0 --runs 3 --display_step 1000 >> results/${DATASET}.txt
                done
            done
        done
    done
done