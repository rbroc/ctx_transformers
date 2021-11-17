#!/bin/sh

# TESTS
# Biencoder, compare dense, add, and attention aggreagtion
    # Concat gives the best result (1 epoch)
# Standard encoder, compare dense and add
    # add is slightly better

# First:
# Standard: 3 layers, 1 epoch, continue if it works - FAILS
# Hierarchial: 2 layers, 3 epochs - UP NEXT
# Biencoder: 3x1 layers, 3 epochs - UP NEXT

# Then, priorities:
# Set up evaluation
# Train separable architecture

# Next:
# Biencoder: 3x2 layers, 3 epochs
# Hierarchical: 3 layers, try 1 epoch then maybe 3 epochs
# Standard: 6 layers, 3 epoch
# Biencoder: use pretrained as token encoder [OPTIONAL]
# Standard: pretrained [OPTIONAL]

python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

#python3 train_mlm.py --log-path hierarchical --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3


#python3 train_mlm.py --log-path standard --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 3 --aggregate add --reset-head

#python3 train_mlm.py --log-path standard --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 3 --aggregate add --reset-head

#python3 train_mlm.py --log-path standard --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type standard  --n-layers 3 --aggregate add --reset-head


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
