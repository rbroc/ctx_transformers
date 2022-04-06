#!/bin/sh

# Models
# STANDARD: Biencoder 1/2 - minor gains
# STANDARD: 3 layers standard with sum - no
# SEPARABLE: 3 layers body separable (v2)
# SEPARABLE: 3 layers head separable (v1)
# SEPARABLE: 1 layers head separable (v1)
# STANDARD: 3 layers standard with attention - no 
# STANDARD: Biencoder 1/3
# SEPARABLE: Separable body biencoder with add, v2

# Goals
# STANDARD
    # Set up triplet loss benchmarking
    # Benchmark the biencoder and standard non-separable on triplet
# SEPARABLE
    # Set up test model
    # Benchmark on triplet
    # Implement biencoder separable head and run

# Hierarchical 1 layer
#python3 train_mlm.py --log-path hierarchical_1layers --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 8 --mlm-type hier --n-layers 1

# STANDARD: Biencoder 1
python3 train_mlm.py --log-path biencoder_2_1 --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 3 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 2 --n-layers-context-encoder 1 --reset-head  --aggregate attention

python3 train_mlm.py --log-path biencoder_2_1 --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 3 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 2 --n-layers-context-encoder 1 --reset-head  --aggregate attention

# STANDARD: Standard 3 layers add
#python3 train_mlm.py --log-path standard_3layers --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate add --reset-head

#python3 train_mlm.py --log-path standard_3layers --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate add --reset-head

# SEPARABLE: Standard head mask combined
#python3 train_mlm_combined.py --log-path standard_3layers --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate attention --reset-head

#python3 train_mlm_combined.py --log-path standard_3layers --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 1 --aggregate attention --reset-head

# SEPARABLE: Standard body mask combined
#python3 train_mlm_combined_v2.py --log-path standard_3layers --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate add --reset-head

# STANDARD: Standard 3 layers hierarchical head
#python3 train_mlm.py --log-path standard_3layers --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate attention --reset-head

#python3 train_mlm.py --log-path standard_3layers --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate attention --reset-head

# STANDARD: Biencoder 2
python3 train_mlm.py --log-path biencoder_3_1 --dataset-name 10context_large --context-type author --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --n-layers-context-encoder 1 --reset-head  --aggregate attention

python3 train_mlm.py --log-path biencoder_3_1 --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 8 --mlm-type biencoder --n-layers 3 --n-layers-context-encoder 1 --reset-head --aggregate attention

# SEPARABLE: Biencoder combined (try other head)
python3 train_mlm_combined.py --log-path biencoder_2_1 --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type biencoder --n-layers 2 --n-layers-context-encoder 1 --aggregate add --reset-head

python3 train_mlm_combined_v2.py --log-path biencoder_2_1 --dataset-name 10context_large --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type biencoder --n-layers 2 --n-layers-context-encoder 1 --aggregate add --reset-head

python3 train_mlm.py --log-path standard_3layers --dataset-name 10context_large --context-type random --per-replica-batch-size 1 --dataset-size 2000000 --n-epochs 2 --start-epoch 0 --update-every 1 --mlm-type standard --n-layers 3 --aggregate attention --reset-head
