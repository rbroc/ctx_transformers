#!/bin/sh

python3 train_classifier.py --dataset-name 1post --log-path testbnorm --per-replica-batch-size 4 --dataset-size 1000000 --n-epochs 1 --update-every 8 --nposts 1 --pretrained-weights distilbert-base-uncased --pooling cls --use-embeddings all