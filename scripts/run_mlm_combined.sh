#!/bin/sh

# *** NEXT *** 
# PROJECT 1:
# Train standard for longer (head + add?) -> 3 epochs, to start with
# Train hierarchical for longer -> 3 epochs, to start with
# Train biencoder -> 3 epochs, to start with

# PROJECT 2:
# Try one separable - PRIORITY

# DOUBTS
# Remember to shuffle
# Ensure stability with random contexts when masking the head? Or remove random?
# Divide by 2 when sum?

# REMINDER
# Hierarchical: masks the hierarchical head
# Biencoder: masks attention in the context
# Standard: masks the hierarchical head, if present


# Biencoder
python3 train_mlm_combined.py --log-path biencoder_test --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 100 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type biencoder --n-layers 3 --n-layers-context-encoder 1 --aggregate add

#python3 train_mlm_combined.py --log-path standard_test --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 100 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 2 --aggregate attention

#python3 train_mlm_combined.py --log-path hierarchical_test --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 100 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2