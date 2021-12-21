#!/bin/sh

# Need to implement functionality for no head masking at test / re-eval
# Need to implement functionality for testing
# Make list of models to test


python3 train_mlm.py --log-path hierarchical_1layers --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 1

python3 train_mlm.py --log-path hierarchical_1layers --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 1

python3 train_mlm.py --log-path standard_3layers --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate add --reset-head

python3 train_mlm.py --log-path standard_3layers --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate add --reset-head
