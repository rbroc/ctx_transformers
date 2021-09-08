#!/bin/sh

for grouping in author
do
python3 train_mlm.py --log-path 10context_6layers_skipconn --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --context-pooling cls --add-dense 0 --dims 768 --freeze-encoder-false --reset-head
done

# NEXT
# Train random longer
# From pretrained
# From triplet loss trained

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