#!/bin/sh

for grouping in subreddit
do
python3 train_mlm.py --log-path 10context_biencoder_3_1_attention --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 10 --context-pooling cls --add-dense 0 --dims 768 --update-every 8 --freeze-encoder-false
done

# NEXT (Weds-Fri)
# Make classification model for triplet loss
# Make triplet loss script
# Make datasets for triplet loss
# Support params for triplet loss

# ENH
# Test a bunch
# Make sure add_dense supports list always
# Add dropout
# Add model loading (later)
# Send to Tal

# COMMENTS
# How to compare to no-context MLM?
# Could consider running pretrained
# Consider re-running hierarchical
# Which combos to run?
# Make sure to make notes of latest versions run
