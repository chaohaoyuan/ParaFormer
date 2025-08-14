#### image/text classification datasets, random splits with different training label nums per class ####

## stl10
label_list=(10 50 100)
# DIFFormer-s
for label in ${label_list[@]}
do
python main.py --dataset stl10 --method difformer --rand_split_class --label_num_per_class $label --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 400 --use_residual --use_bn --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 5
done

python main.py --dataset stl10 --method gcn --rand_split_class --label_num_per_class 100 --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 400 --use_residual --use_bn --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 5

python main.py --dataset stl10 --method mlp --rand_split_class --label_num_per_class 100 --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 400 --use_residual --use_bn --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 5

python main.py --dataset stl10 --method GPR --rand_split_class --label_num_per_class 100 --valid_num 1000 \
        --lr 0.005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 64 --use_residual --use_bn --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 5

python main.py --dataset stl10 --method GPR --rand_split_class --label_num_per_class 100 --valid_num 1000 \
        --lr 0.005 --weight_decay 0.0005 --dropout 0.3 --num_layers 16 --hidden_channels 32 --use_residual --use_bn --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 3



python main.py --dataset stl10 --method GPR --rand_split_class --label_num_per_class 10 --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 400 --use_residual --use_bn --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 5

python main.py --dataset stl10 --method gprgnn --rand_split_class --label_num_per_class 50 --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 400 --use_residual --use_bn --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 5


# DIFFormer-a
for label in ${label_list[@]}
do
  python main.py --dataset stl10 --method difformer --rand_split_class --label_num_per_class $label --valid_num 1000 \
         --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 400 --use_residual --use_bn  --alpha 0.5 --kernel sigmoid \
         --epochs 600 --seed 123 --device 2 --runs 5
done

## cifar10
label_list=(10 50 100)
# DIFFormer-s
for label in ${label_list[@]}
do
  python main.py --dataset cifar10 --method difformer --rand_split_class --label_num_per_class $label --valid_num 1000 \
        --lr 0.0001 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 300 --use_residual --use_bn  --alpha 0.5 --kernel simple \
        --epochs 600 --seed 123 --device 2 --runs 5
done

# DIFFormer-a
for label in ${label_list[@]}
do
  python main.py --dataset cifar10 --method difformer --rand_split_class --label_num_per_class $label --valid_num 1000 \
      --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 300 --use_residual --use_bn  --alpha 0.5 --kernel sigmoid \
      --epochs 600 --seed 123 --device 2 --runs 5
done


## 20news-group
label_list=(100 200 400)
# DIFFormer-s
for label in ${label_list[@]}
do
  python main.py --dataset 20news --method difformer --rand_split_class --label_num_per_class $label --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 300 --use_residual --use_bn  --alpha 0.5 --kernel simple \
        --epochs 600 --seed 42 --device 2 --runs 5
done

# DIFFormer-a
for label in ${label_list[@]}
do
  python main.py --dataset 20news --method difformer --rand_split_class --label_num_per_class $label --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 300 --use_residual --use_bn  --alpha 0.5 --kernel sigmoid \
        --epochs 600 --seed 123 --device 2 --runs 5
done

python main.py --dataset 20news --method difformer --rand_split_class --label_num_per_class 100 --valid_num 1000 \
        --lr 0.0005 --weight_decay 0.1 --dropout 0.0 --num_layers 2 --hidden_channels 300 --use_residual --use_bn  --alpha 0.5 --kernel simple \
        --epochs 600 --seed 42 --device 2 --runs 5


CUDA_VISIBLE_DEVICES=3 python main.py --dataset 20news --method GPR --rand_split_class --label_num_per_class 400 --valid_num 1000 \
        --lr 0.005 --weight_decay 0.001 --dropout 0.3 --num_layers 2 --hidden_channels 512 --use_residual --use_bn  --alpha 0.5 --kernel simple \
        --epochs 600 --seed 42 --device 0 --runs 3