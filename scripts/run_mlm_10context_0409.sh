#!/bin/sh

# Sum, train all
for grouping in author
do
python3 train_mlm.py --log-path 10context_6layers_skipconn --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 5 --context-pooling cls --add-dense 0 --dims 768 --freeze-encoder-false --reset-head
done