#!/bin/sh

python3 train_classifier.py --dataset-name 1post --log-path test --per-replica-batch-size 1 --dataset-size 10 --n-epochs 1 --update-every 8 --nposts 1 --pretrained-weights distilbert-base-uncased --pooling cls