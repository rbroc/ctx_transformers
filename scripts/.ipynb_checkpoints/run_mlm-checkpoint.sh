#!/bin/sh

# *** NEXT *** 
# NOT SEPARABLE
# Try pretrained standard with frozen and hierarchical head - THU/FRI
# Try non-pretrained (smaller) standard with hierarchical head - SAT/SUN
# Keep doing standard - also subreddit, and longer epochs - WED + SUN/MON
# Longer hierarchical - TUE/WED
# Try biencoder, longer running (with add? or with attention?) - THU/SAT
# Evaluate

# SEPARABLE
# Try standard with head, not separable - 6-13 DEC
# Try hierarchical separable - 6-13 DEC
# Try biencoder with separable sentence encoder - 11-13 DEC
    # (ctx is 0 if random, defined if other)
    
# THEN
# 13-22 DEC: Make abstract
# 22 DEC - 9 JAN: Some long-running stuff (most promising)

# DOUBTS
# Ensure stability with random contexts when masking the head? Or remove random?
# Divide by 2 when sum?

# TESTS
# Biencoder: concat gives the best result (1 epoch)
# Standard encoder: add is better

# DO
#python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

# DO (AND DO RANDOM LONGER?)
#python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

# DO LONGER (2 layers)
#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

# DO LONGER (2 layers)
#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

# DO (2 layers)
#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

# DO LONGER (AND ADD RANDOM AND SUBREDDIT)
#python3 train_mlm.py --log-path standard --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 2 --aggregate add --reset-head

# REDO
#python3 train_mlm.py --log-path standard_hierhead --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 2 --aggregate attention --reset-head

# REDO
# python3 train_mlm.py --log-path standard_hierhead --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 2 --aggregate attention --reset-head

python3 train_mlm.py --log-path standard_hierhead_pretrained --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type standard --aggregate attention --reset-head --pretrained-weights distilbert-base-uncased --freeze-encoder 0 1 2 3 4 5

python3 train_mlm.py --log-path standard_hierhead_pretrained --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type standard --aggregate attention --reset-head --pretrained-weights distilbert-base-uncased --freeze-encoder 0 1 2 3 4 5
