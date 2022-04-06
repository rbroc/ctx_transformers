#!/bin/sh

# 6l performance
#python3 train_mlm.py --log-path single_6l --dataset-name 10context_large --context-type single --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --n-layers 6 --reset-head

# With static distilbert
python3 train_mlm.py --log-path biencoder_10_pretrained --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/10_1_1_standard --pretrained-weights distilbert-base-uncased --freeze-encoder 0 1 2 3 4 5

python3 train_mlm.py --log-path biencoder_1_pretrained --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/1_1_1_standard --pretrained-weights distilbert-base-uncased --freeze-encoder 0 1 2 3 4 5

python3 train_mlm.py --log-path biencoder_10_scratch_2_pretrained --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/10_1_1_scratch_2l --pretrained-weights distilbert-base-uncased --freeze-encoder 0 1 2 3 4 5

python3 train_mlm.py --log-path biencoder_1_scratch_2_pretrained --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/1_1_1_scratch_2l --pretrained-weights distilbert-base-uncased --freeze-encoder 0 1 2 3 4 5

# With trainable 2 layers token encoder
python3 train_mlm.py --log-path biencoder_10 --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/10_1_1_standard --n-layers 2

python3 train_mlm.py --log-path biencoder_1 --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/1_1_1_standard ---n-layers 2

python3 train_mlm.py --log-path biencoder_10_scratch_2 --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/10_1_1_scratch_2l --n-layers 2

python3 train_mlm.py --log-path biencoder_1_scratch_2 --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 1 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers-context-encoder 1 --reset-head --aggregate add --ctxpath ../../triplet-eval/models/1_1_1_scratch_2l --n-layers 2

# Should also look into pretrained distlbert?
# Should do concat?
