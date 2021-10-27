#!/bin/sh

python3 train_metrics.py --dataset-name 2000000_posts --log-path test --per-replica-batch-size 1 --dataset-size 10 --n-epochs 1 --update-every 16 --loss-type mae --weights distilbert-base-uncased --metric-type single --targets score comments --add-dense 2 --dims 768 768 --activations relu relu

# Notes
# --add-dense --dims --activations
# targets: avg_score, avg_comm, n_posts, - score, comments
# datasets - 3_random, [could also do 10] - 2000000_posts
# weights is path

# Notes:
# The loss averages across objectives (but we're logging both)