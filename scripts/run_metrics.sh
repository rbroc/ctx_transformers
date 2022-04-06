#!/bin/sh

#python3 train_metrics.py --dataset-name 3_random_norm --log-path test_trained --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 1 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets avg_score --add-dense 1 --dims 768 --activations relu

#python3 train_metrics.py --dataset-name 3_random_norm --log-path test_pretrained --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 1 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets avg_score --add-dense 1 --dims 768 --activations relu

python3 train_metrics.py --dataset-name 3_random --log-path test_trained --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 1 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets avg_score --add-dense 3 --dims 768 768 768 --activations relu relu relu --encoder-trainable

python3 train_metrics.py --dataset-name 3_random --log-path test_pretrained --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 1 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets avg_score --add-dense 3 --dims 768 768 768 --activations relu relu relu --encoder-trainable


# TO DO
# Test with no intermediate layer or more intermediate layers
# Run aggregates on multiple metrics?
