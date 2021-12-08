#!/bin/sh

python3 train_mlm.py --log-path biencoder_slowupdate --context-type random --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type biencoder --n-layers 3 --n-layers-context-encoder 1 --aggregate add --reset-head

python3 train_mlm.py --log-path biencoder_slowupdate --context-type author --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type biencoder --n-layers 3 --n-layers-context-encoder 1 --aggregate add --reset-head

python3 train_mlm_combined.py --log-path standard_innermask_slowupdate --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 2 --aggregate add --reset-head

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2
