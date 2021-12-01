#!/bin/sh

# 1l is actually 2l
# Do standard for longer - TRY!
# Doing hierarchical - NOT WORKING
# Do biencoder (eval) - further training + eval - TRY!
# Try mixed 

# *** NEXT *** 
# NOT SEPARABLE
# All freeze one epoch longer
# Try non-pretrained (smaller) standard with hierarchical head - SOME RESULT - continue
# Try non-pretrained (smaller) standard with add head - TINY RESULT
# Longer hierarchical - TO DO
# Keep doing standard - also subreddit, and longer epochs - TO DO
# Try biencoder, longer running  - TO DO
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

# Hierarchical
python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2

python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 1 --mlm-type hier --n-layers 2