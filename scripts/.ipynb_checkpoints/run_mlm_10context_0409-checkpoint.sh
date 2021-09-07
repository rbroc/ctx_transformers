#!/bin/sh

# Attention, train attention layer only - DONE
#for grouping in author random
#do
#python3 train_mlm.py --log-path 10context_1607 --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 500000 --n-epochs 1 --context-pooling cls --aggregate attention --add-dense 0 --dims 768 --freeze-head
#done

# Sum, train encoder but not head - DONE
#for grouping in author random
#do
#python3 train_mlm.py --log-path 10context_1607 --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 500000 --n-epochs 1 --context-pooling cls --aggregate concatenate --add-dense 0 --dims 768 --freeze-head --freeze-encoder-false
#done

# Sum, train head but not encoder - TO RUN
#for grouping in author random
#do
#python3 train_mlm.py --log-path 10context_1607 --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 500000 --n-epochs 1 --context-pooling cls --aggregate concatenate --add-dense 0 --dims 768
#done

# Attention, train encoder only - CRASHES
#for grouping in author random
#do
#python3 train_mlm.py --log-path 10context_1607 --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 500000 --n-epochs 1 --context-pooling cls --aggregate attention --add-dense 0 --dims 768 --freeze-head --freeze-encoder-false
#done

# Attention, train all - CRASHES
#for grouping in author random
#do
#python3 train_mlm.py --log-path 10context_1607 --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 500000 --n-epochs 1 --context-pooling cls --aggregate attention --add-dense 0 --dims 768--freeze-encoder-false
#done

# Sum, train all
for grouping in single
do
python3 train_mlm.py --log-path 10context_3layers --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 5 --context-pooling cls --aggregate concatenate --add-dense 0 --dims 768 --freeze-encoder-false --reset-head
done

#for grouping in author random single
#do
#python3 train_mlm.py --log-path 10context_1607 --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --context-pooling cls --aggregate attention --add-dense 0 --dims 768 --freeze-encoder-false
#done
