#!/bin/sh

# HIERARCHICAL:
# Try old hierarchical
# Try old standard (w/ add)
# Run hierarchical for several epochs
# Run standard for several epochs
# Run biencoder for several epochs
# Meanwhile:
# Implement new
# Try hierarchical head on pretrained (single context)
# Try hierarchical multi-context
# Try biencoder multi-context


#python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

#python3 train_mlm.py --log-path biencoder --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --aggregate concat --reset-head --n-layers-context-encoder 1

python3 train_mlm.py --log-path hierarchical_checkpoint --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

python3 train_mlm.py --log-path hierarchical_checkpoint --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

python3 train_mlm.py --log-path hierarchical_checkpoint --dataset-name 10context_large --context-type subreddit --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 3

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
