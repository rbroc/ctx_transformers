#!/bin/sh

############### NO DENSE LAYERS ###############
# Freeze all
#for grouping in author subreddit random
#do
#python3 train_mlm.py --log-path 3context_random_200k_unfreeze_reset_head --dataset-name 3context_random --context-type $grouping --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --add-dense 0 --reset-head
#done

# Run simple mlm
#python3 train_mlm.py --log-path 3context_random_200k_unfreeze_reset_head --dataset-name 3context_random --context-type single --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --reset-head

# Also train one with pretrained
#for grouping in author subreddit random
#do
#python3 train_mlm.py --log-path 3context_random_200k_unfreeze_reset_head --dataset-name 3context_random --context-type #$grouping --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --add-dense 0 --reset-#head --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface
#done

python3 train_mlm.py --log-path 3context_random_200k_unfreeze_reset_head --dataset-name 3context_random --context-type single --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --reset-head --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface



# Freeze most layers
#for grouping in author subreddit random
#do
#    python3 train_mlm.py --log-path 3context_random_200k_unfreeze --dataset-name 3context_random --context-type $grouping --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --add-dense 0 --freeze-encoder-layers 0 1 2 3 4
#done

#python3 train_mlm.py --log-path 3context_random_200k_unfreeze --dataset-name 3context_random --context-type single --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --freeze-encoder-layers 0 1 2 3 4


############### ADD 3 DENSE LAYERS ###############
# Freeze all
#for grouping in author subreddit random
#do
#    python3 train_mlm.py --log-path 3context_random_200k_unfreeze --dataset-name 3context_random --context-type $grouping --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --add-dense 3
#done

# Run simple mlm
#python3 train_mlm.py --log-path 3context_random_200k_unfreeze --dataset-name 3context_random --context-type single --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false

# Freeze most layers
#for grouping in author subreddit random
#do
#    python3 train_mlm.py --log-path 3context_random_200k_unfreeze --dataset-name 3context_random --context-type $grouping --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --add-dense 3 --freeze-encoder-layers 0 1 2 3 4
#done

#python3 train_mlm.py --log-path 3context_random_200k_unfreeze --dataset-name 3context_random --context-type single --per-replica-batch-size 3 --dataset-size 200000 --n-epochs 5 --freeze-encoder-false --freeze-encoder-layers 0 1 2 3 4


# Log:
# 1. Train all or freeze first four layers with one or 3 dense after context
# 2. Freeze all, add only one dense, retrain head
# 3. Trained encoder + context

# Next:
# Use attention to learn context?
# Trained 
# More layers?
