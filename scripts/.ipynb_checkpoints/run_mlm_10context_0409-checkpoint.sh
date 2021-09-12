#!/bin/sh

for grouping in author single
do
python3 train_mlm.py --log-path 10context_adam2neg5 --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --context-pooling cls --add-dense 0 --dims 768 --freeze-encoder-false --reset-head
done

# NEXT
# Biasing prediction matrix (train fsom scratch)
# Train random longer (non pretrained, pretrained)
# From triplet loss trained
# Small encoder (e.g., triplet-loss model?)
# With concat

# OTHER 
# Add attention layer
# Do context only to start with!
# Redo prior?

# Compare to single
# Do pretrained, train all
# Do pretrained with encoder freeze
    # W/ head training
    # No head training (one-shot)
# should also check what happens if context is not used?