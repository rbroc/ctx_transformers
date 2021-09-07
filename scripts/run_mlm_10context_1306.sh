#!/bin/sh

### 21/06 and 23/06
# Context without freeze encoder 
#for grouping in author random
#do
python3 train_mlm.py --log-path 10context_random_2406 --dataset-name 10context_random --context-type random --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --add-dense 0 --freeze-encoder-false --context-pooling cls --context-as-vocab-prior # COULD DO THE SAME AND RESET HEAD / ALSO WITHOUT INTERACTION WEIGHTS
# This is with a dense layer
#done

# Attention with 6 heads + cls context pooling + freeze encoder
#for grouping in author #random
#do
python3 train_mlm.py --log-path 10context_random_2406 --dataset-name 10context_random --context-type random --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --add-dense 0 --context-pooling cls --aggregate attention 
# this is with no reset
#done

# Attention with 6 heads + cls context pooling + freeze encoder + load trained
#for grouping in author #random
#do
python3 train_mlm.py --log-path 10context_random_2406 --dataset-name 10context_random --context-type random --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --add-dense 0 --context-pooling cls --aggregate attention --load-encoder-weights ../logs/triplet/10anchor_1pos_1neg_random/huggingface --reset-head # this is trained with reset
#done 

# Attention with 6 heads + cls context pooling + freeze encoder + load trained + nofreeze
#for grouping in author #random
#do
python3 train_mlm.py --log-path 10context_random_2406 --dataset-name 10context_random --context-type random --per-replica-batch-size 1 --dataset-size 100000 --n-epochs 5 --add-dense 0 --context-pooling cls --aggregate attention --freeze-encoder-false # this is with no freezing of encoder 
#done


# Run random as a test (script above)
# Run prior with layer - include interactions
# Run the following

# Redo dataset and run:
# Standard, three layers of 1536, 1536, 768; no reset - 5 epochs - tune head only - 2
# Standard, three layers of 1536, 1536, 768; reset - epochs - tune head only - 2
# Standard, three layers of 1536, 1536, 768; no reset - tune all - 1
# Standard, three layers of 1536, 1536, 768; reset - tune all - 1
# With prior, cls, no reset; - 5 epochs - tune head only - 2 
# With prior, cls, reset; - 5 epochs - tune head only - 2
# With prior, cls, no reset; - 5 epochs - tune all - 1
# With prior, cls, reset; - 5 epochs - tune all - 1
# Attention, cls, no reset; - 5 epochs - tune head only - 2 
# Attention, cls, reset; - 5 epochs - tune head only - 2
# Attention, cls, no reset; - 5 epochs - tune all - 1 
# Attention, cls, reset; - 5 epochs - tune all - 1