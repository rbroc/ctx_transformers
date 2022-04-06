#!/bin/sh

#python3 train_mlm.py --log-path single_1l --dataset-name 10context_large --context-type single --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --n-layers 1 --reset-head

python3 train_mlm.py --log-path single_2l --dataset-name 10context_large --context-type single --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --n-layers 2 --reset-head

python3 train_mlm.py --log-path single_3l --dataset-name 10context_large --context-type single --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --n-layers 3 --reset-head

