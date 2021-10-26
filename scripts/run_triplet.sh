#!/bin/sh

#for p in cls mean random
#    do for ct in dense
#        do for cs in 100 20 10
#            do
#            python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path test --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-#every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --compress-to $cs --compress-mode $ct --#pooling $p
#            done
#        done 
#    done


# NEXT
# Next train with more posts
# Next add compressions
# Only the end, play with margins (and maybe reset architecture)
# Finally, try biencoder loading


#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls

# Other margin
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin #1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --loss-margin 0.01

# Other aggregations
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin #1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling mean

python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 10 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling random

# With compression head
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 50 --compress-mode dense

#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 50 --compress-mode vae

# More anchors
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls

#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 50 --compress-mode dense

# Optional (w/ VAE and more anchors)
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 50 --compress-mode vae

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
