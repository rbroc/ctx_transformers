#!/bin/sh

#for grouping in author subreddit random
#do
#python3 train_mlm.py --log-path 10context_random_100k --dataset-name 10context_random --context-type $grouping --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --freeze-encoder-false --add-dense 0
#done

#python3 train_mlm.py --log-path 10context_random_100k --dataset-name 10context_random --context-type single --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --freeze-encoder-false

python3 train_mlm.py --log-path 10context_random_100k --dataset-name 10context_random --context-type author --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --freeze-encoder-false --add-dense 3

python3 train_mlm.py --log-path 10context_random_100k --dataset-name 10context_random --context-type author --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --freeze-encoder-false --add-dense 0 --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

python3 train_mlm.py --log-path 10context_random_100k --dataset-name 10context_random --context-type author --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --freeze-encoder-false --add-dense 0 --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface --reset-head

python3 train_mlm.py --log-path 10context_random_100k --dataset-name 10context_random --context-type single --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --freeze-encoder-false --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface --reset-head


# Could reset head - done
# Could add more dense? - done
# Could try triplet-loss trained - done
# Could play with activations


