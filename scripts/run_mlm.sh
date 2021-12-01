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


# Hierarchical
python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2

python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2