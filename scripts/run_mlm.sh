#!/bin/sh

# Running some tests for aggregation
# NEXT:
# Define how to run standard, hier and bienc
# Define mixed architecture!

python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 10 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder  --n-layers 3 --n-layers-context-encoder 1 --aggregate add --reset-head

python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder  --n-layers 3 --n-layers-context-encoder 1 --aggregate concat --reset-head

python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder  --n-layers 3 --n-layers-context-encoder 1 --aggregate attention --reset-head

# --add-dense 2 --dims 768 768 --activations relu relu --freeze-encoder 0 1 2 3 4 5
# --pretrained-weights distilbert-base-uncased

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
