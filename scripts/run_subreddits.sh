#!/bin/sh

for s in 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
    for posts in 1 10
    do
        python3 train_subreddits.py --dataset-name single_$s --log-path final_1anchor_$posts --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../trained/final_1anchor/epoch-2 --target-dims 1 --nr $posts --pad-to $posts
    done
done

#for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 26 27 28 29 30
#do
#    for posts in 1
#    do
#        python3 train_subreddits.py --dataset-name single_$s --log-path distilbert_$posts --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights distilbert-base-uncased --target-dims 1 --nr $posts --pad-to $posts
#    done
#done



