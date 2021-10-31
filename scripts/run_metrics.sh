#!/bin/sh

# Train with one intermediate layer
python3 train_metrics.py --dataset-name 2000000_posts --log-path test_trained --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-type huber --weights ../logs/triplet/1anchor/standard/huggingface --metric-type single --targets score --add-dense 1 --dims 768 --activations relu

python3 train_metrics.py --dataset-name 3_random --log-path test_trained --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-type huber --weights ../logs/triplet/1anchor/standard/huggingface --metric-type aggregate --targets avg_score --add-dense 1 --dims 768 --activations relu

python3 train_metrics.py --dataset-name 2000000_posts --log-path test_pretrained --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-type huber --weights distilbert-base-uncased --metric-type single --targets score --add-dense 1 --dims 768 --activations relu

python3 train_metrics.py --dataset-name 3_random --log-path test_pretrained --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-type huber --weights distilbert-base-uncased --metric-type aggregate --targets avg_score --add-dense 1 --dims 768 --activations relu


# TO DO
# Run for longer
# Fix labels?
# Test with no intermediate layer or more intermediate layers
# Run aggregates on multiple metrics?
# Larger batches

# Notes:
# the loss averages across objectives (but we're logging both)
# datasets - 3_random, [could also do 10] - 2000000_posts
# targets: avg_score, avg_comm, n_posts, - score, comments