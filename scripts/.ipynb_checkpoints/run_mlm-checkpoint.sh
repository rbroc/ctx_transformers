#!/bin/sh

# Run simple mlm
#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type single --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type single --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5

# Run with author as context
#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type author --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 0

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type author --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense before_norm

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type author --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense after_norm

# With author as context and trained model
#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type author --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 0 --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type author --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense before_norm --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type author --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense after_norm --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

# Same for subreddit
#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type subreddit --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 0

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type subreddit --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense before_norm

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type subreddit --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense after_norm

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type subreddit --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 0 --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type subreddit --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense before_norm --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type subreddit --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense after_norm --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

# Same for random
#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type random --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 0

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type random --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense before_norm

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type random --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense after_norm

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type random --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 0 --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

# NOT RUN
#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type random --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense before_norm --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface

#python3 train_mlm.py --log-path 3context_random_200k --dataset-name 3context_random --context-type random --per-replica-batch-size 4 --dataset-size 200000 --n-epochs 5 --add-dense 3 --where-dense after_norm --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface