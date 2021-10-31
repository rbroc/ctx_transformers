#!/bin/sh

# Plan:
# Classification 10x?
# Check for length (e.g., simple model?)
# Do 3 with compression
# Do 10 with compression

# Give classification a shot - DONE
#python3 train_classifier.py --dataset-name 1post --log-path 1post --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --nposts 1 --pretrained-weights distilbert-base-uncased --pooling cls --use-embeddings all

# 1 anchor, plain - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls

# 1 anchor with dense 100 - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 100 --compress-mode dense

# 1 anchor with vae 100 - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 10 --compress-mode vae

# 1 anchor with dense 10 - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --intermediate-size 100 --compress-to 20 --compress-mode dense

# 1 anchor with vae 10
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --intermediate-size 100 --compress-to 20 --compress-mode vae

# 1 anchor with random pooling - DONE
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 1anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-pos 1 --n-neg 1 --n-anchor 1 --pretrained-weights distilbert-base-uncased --pooling random

# Try 10 anchor, standard - PARTLY RUN
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 10anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls

# 3pos / 3neg standard - DONE
#python3 train_triplet.py --dataset-name 3pos_3neg_random --log-path 3anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 3 --n-pos 3 --n-neg 3 --pretrained-weights distilbert-base-uncased --pooling cls

# Classification with more posts
python3 train_classifier.py --dataset-name 3post --log-path 3post --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --nposts 3 --pretrained-weights distilbert-base-uncased --pooling cls --use-embeddings all

# 10 compress to 20
python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 10anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --intermediate-size 100 --compress-to 20 --compress-mode dense

# Classification with distance only
python3 train_classifier.py --dataset-name 1post --log-path 1post --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --nposts 1 --pretrained-weights distilbert-base-uncased --pooling cls --use-embeddings distance

# 3 compress to 20
python3 train_triplet.py --dataset-name 3pos_3neg_random --log-path 3anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 3 --n-pos 3 --n-neg 3 --pretrained-weights distilbert-base-uncased --pooling cls --intermediate-size 100 --compress-to 20 --compress-mode dense




# Other context sizes and combinations
# 10 compress to 100 - LATER
#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path 10anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-pos 1 #--n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 100 --compress-mode dense

# 3 compress to 100 - LATER
#python3 train_triplet.py --dataset-name 3pos_3neg_random --log-path 3anchor --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor #3 --n-pos 3 --n-neg 3 --pretrained-weights distilbert-base-uncased --pooling cls --compress-to 100 --compress-mode dense


# NEXT
# Try some of these with FFN
# Maybe try non_pretrained
# Rerun MLMs