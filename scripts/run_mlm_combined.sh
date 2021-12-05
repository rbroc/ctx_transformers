#!/bin/sh

# Run from new branch - if same, check if something's wrong
# Try with longer learning rate
# also try biencoder sharing
# Remembed to fix the training / test attention update issue
# Remember to set trainable=False when testing (for attention update)
# How does it work for attention masking? Keep random? Remove?

#python3 train_mlm_combined.py --log-path standard_innermask_slowupdate --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 2 --aggregate add --reset-head

python3 train_mlm_combined.py --log-path biencoder_innermask_slowupdate --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type biencoder --n-layers 3 --n-layers-context-encoder 1 --aggregate add --reset-head

#python3 train_mlm_combined.py --log-path biencoder_test_slowupdate --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 2 --aggregate attention --reset-head

#python3 train_mlm_combined.py --log-path hierarchical_test_slowupdate --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2 --reset-head