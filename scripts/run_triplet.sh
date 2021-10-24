#!/bin/sh

for p in cls mean random
    do for ct in dense
        do for cs in 100 20 10
            do
            python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path test --per-replica-batch-size 1 --dataset-size 10 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --compress-to $cs --compress-mode $ct --pooling $p
            done
        done 
    done


# Parameters
# n posts
    # 10 1 1 - dataset
    # 3 3 3 - dataset 
    # 1 1 1 - subset
# margins - tuning
    # 1, .1, .001
# compress
    # dense
        # no intermediate, 100, 50, 20, 10
    # vae
        # no intermediate, 100, 50, 20, 10
# pooling:
    # cls
    # mean
    # random

# Start from:
    # 1 - 1 - 1, different compressions, different pooling
    
# Fix optimizer weights
