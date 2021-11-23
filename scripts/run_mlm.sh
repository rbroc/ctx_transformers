#!/bin/sh

# To dos:
# NOT SEPARABLE
# Keep doing standard - also subreddit, and longer epochs
# Try pretrained standard with frozen and hierarchical head
# Try non-pretraiend (smaller) standard with hierarchical head
# Try hierarchical, other combos
# Try biencoder, longer running
# Evaluate

# SEPARABLE
# Try standard with head, not separable
# Try hierarchical separable
# Try biencoder with separable sentence encoder 

# Add divide by two

# TESTS
# Biencoder, compare dense, add, and attention aggreagtion
    # Concat gives the best result (1 epoch)
# Standard encoder, compare dense and add
    # add is slightly better
# Standard encoder does not display visibly good behavior


#python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

#python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

#python3 train_mlm.py --log-path standard --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 2 --aggregate add --reset-head

python3 train_mlm.py --log-path standard --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 2 --aggregate add --reset-head

# Relevant params
# - Architectures
    # standard:
        # pretrained vs no pretrained - tested 
            # if pretrained, freeze?
        # aggregation via add or concat - tested
        # add dense and dims for after aggregation, before layernorm - tested
    # hier: 
        # n_layers
    # biencoder
        # pretrained vs. no-pretrained
            # if pretrained, freeze
        # n_layers and n_layers context encoder
        # aggregate (concat, add, attention)
        # add dense and dims after aggregation
        # context pooler pools and normalizes
