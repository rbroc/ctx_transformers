#!/bin/sh

# List of models
# Single, 3 layers
# With contexts, 3 layers, standard -> exists?
# Biencoder, 2 + 1 layers -> exists?
# With contexts and masking -> esists?
# Biencoder with contexts and masking -> exists?

# Evaluation
# All on MLM
# Triplet loss:
    # - Single
    # - Standard with contexts and masking
    # - Standard after attention
    # - Biencoder, token and context encoder
    # - Biencoder after attention