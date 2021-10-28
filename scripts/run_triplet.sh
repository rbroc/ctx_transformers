#!/bin/sh

# 1 anchor, plain - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls

# 1 anchor with dense 100 - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 100 --compress-mode dense

# 1 anchor with vae 100 - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 10 --compress-mode vae

# 1 anchor with dense 10
python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --intermediate-size 100 --compress-to 10 --compress-mode dense

# 1 anchor with vae 10
python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --intermediate-size 100 --compress-to 10 --compress-mode vae

# 1 anchor with random pooling
python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-pos 1 --n-neg 1 --n-anchor 1 --pretrained-weights distilbert-base-uncased --pooling random

# NEXT
# Try compression with intermediate, otherwise try with fewer
# Try 10 and 3/3/3 on the best models
# If random works, try other combinations with random
# Maybe try non_pretrained
# Rerun MLMs