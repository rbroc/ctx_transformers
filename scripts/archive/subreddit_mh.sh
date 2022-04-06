#!/bin/sh

for sb in 2
do
    for posts in 1
    do
        python3 train_subreddits.py --dataset-name mh --log-path "${sb}_1_1_1_standard_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../../triplet-eval/models/1_1_1_standard --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb
    done
done


for sb in 3 4 5 6 7 8 9 10 11 12
do
    for posts in 1
    do
        python3 train_subreddits.py --dataset-name mh --log-path "${sb}_distilbert_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights distilbert-base-uncased --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb

        python3 train_subreddits.py --dataset-name mh --log-path "${sb}_1_1_1_standard_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../../triplet-eval/models/1_1_1_standard --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb

        python3 train_subreddits.py --dataset-name mh --log-path "${sb}_10_1_1_standard_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../../triplet-eval/models/10_1_1_standard --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb
    done
done

for sb in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
    for posts in 1
    do
        python3 train_subreddits.py --dataset-name mh --log-path "trainable_${sb}_distilbert_$posts" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights distilbert-base-uncased --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb --encoder-trainable

        python3 train_subreddits.py --dataset-name mh --log-path "trainable_${sb}_1_1_1_standard_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../../triplet-eval/models/1_1_1_standard --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb --encoder-trainable

        python3 train_subreddits.py --dataset-name mh --log-path "trainable_${sb}_10_1_1_standard_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../../triplet-eval/models/10_1_1_standard --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb --encoder-trainable
    done
done

for sb in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
    for posts in 10
    do
        python3 train_subreddits.py --dataset-name mh --log-path "${sb}_distilbert_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights distilbert-base-uncased --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb

        python3 train_subreddits.py --dataset-name mh --log-path "${sb}_1_1_1_standard_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../../triplet-eval/models/1_1_1_standard --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb

        python3 train_subreddits.py --dataset-name mh --log-path "${sb}_10_1_1_standard_${posts}" --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../../triplet-eval/models/10_1_1_standard --target-dims 1 --nr $posts --pad-to $posts --which-task mh --subset $sb
    done
done
