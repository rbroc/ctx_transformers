#!/bin/sh

python3 train_personality.py --dataset-name personality --log-path 3d_personality_db_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_db_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_db_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_db_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu 

python3 train_personality.py --dataset-name personality --log-path 3d_personality_db_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_db_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_db_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights distilbert-base-uncased --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_standard --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_standard --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_standard --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_standard --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_standard --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_standard --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_standard --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_standard --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_1l_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_1l --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_1l_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_1l --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_1l_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_1l --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_1l_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_1l --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_1l_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_1l --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_1l_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_1l --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_1l_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_1l --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_2l_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_2l --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_2l_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_2l --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_2l_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_2l --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_2l_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_2l --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_2l_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_2l --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_2l_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_2l --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_2l_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_2l --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_6l_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_6l --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_6l_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_6l --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_6l_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_6l --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_6l_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_6l --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_6l_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_6l --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_6l_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_6l --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_1_scratch_6l_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/1_1_1_scratch_6l --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_6l_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_6l --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_6l_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_6l --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_6l_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_6l --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_6l_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_6l --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_6l_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_6l --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_6l_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_6l --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_6l_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_6l --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_2l_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_2l --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_2l_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_2l --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_2l_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_2l --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_2l_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_2l --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_2l_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_2l --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_2l_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_2l --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_2l_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_2l --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_1l_net --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_1l --metric-type aggregate --targets btw nbtw nsize trst dns brk nbr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_1l_score --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_1l --metric-type aggregate --targets sext sneu sagr scon sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_1l_sext --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_1l --metric-type aggregate --targets sext --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_1l_sneu --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_1l --metric-type aggregate --targets sneu --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_1l_sagr --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_1l --metric-type aggregate --targets sagr --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_1l_scon --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_1l --metric-type aggregate --targets scon --add-dense 3 --dims 768 768 768 --activations relu relu relu

python3 train_personality.py --dataset-name personality --log-path 3d_personality_10_scratch_1l_sopn --per-replica-batch-size 1 --n-epochs 10 --update-every 8 --loss-type mse --weights ../../triplet-eval/models/10_1_1_scratch_1l --metric-type aggregate --targets sopn --add-dense 3 --dims 768 768 768 --activations relu relu relu


