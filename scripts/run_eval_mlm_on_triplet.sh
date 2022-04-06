#!/bin/sh

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_standard_3_add --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/standard_3_add --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_standard_3_add_random --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/standard_3_add_random --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_distibert-base-uncased --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights distilbert-base-uncased --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_combined_standard_3_attention --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/combined_standard_3_attention --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_combinedv2_standard_3_add --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/combinedv2_standard_3_add --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_biencoder_2_1_context --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/biencoder_2_1_context --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_biencoder_2_1_context_random --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 ---n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/biencoder_2_1_context_random --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_biencoder_2_1_token --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/biencoder_2_1_token --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_biencoder_2_1_token_random --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/biencoder_2_1_token_random --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_standard_3_attention --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/standard_3_attention --pooling cls --test-only

python3 eval_mlm_on_triplet.py --dataset-name 3pos_3neg_random --log-path 10_1_1_standard_3_attention_random --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --update-every 8 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pretrained-weights ../pretrained/standard_3_attention_random --test-only
