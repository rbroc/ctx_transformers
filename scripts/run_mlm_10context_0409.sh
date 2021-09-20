#!/bin/sh

# Unfreeze the encoder
for grouping in author random
do
python3 train_mlm.py --log-path 10context_dropoutTarget_scratch --dataset-name 10context_large --context-type $grouping --per-replica-batch-size 1 --dataset-size 1000000 --n-epochs 1 --context-pooling cls --aggregate concatenate --add-dense 3 --dims 1536 1536 768 --freeze-encoder-false --reset-head
done

# Try doing dropout on target only - TRY SEPARATELY
# Increase dropout - TRY SEPARATELY
# Think about gradients w/ FFN?
# Check 