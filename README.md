## Training transformers with structured context
Includes code for two projects:
- Transformers as person/user encoders;
- Training MLM with structured context


## Triplet loss
Using triplet loss with 10 anchor posts, 1 negative and 1 positive example, the model learns to perform at .90
It would be useful to test a few manipulations:
- **Number of anchor/pos/neg posts**:
    - 1/1/1
    - 3/3/3
    - 10/1/1
- Which kind of **pooling**:
    - CLS token
    - Mean of tokens (very unstable if not normalized, same as CLS if normalized)
    - Picking one random token (from a few tests, seems to increase performance significantly)
- Different **dimensionalities** of the output, e.g.:
    - Standard (768)
    - 100 (with a dense layer or a VAE)
    - 20 (with a dense layer or a VAE)

Resulting models should then be evaluates on:
- Predicting single-post metrics
- Predicting user aggregate metrics

#### Next steps:
1. Train one epoch of the 1/1/1 model
    - [ ] CLS standard, dense 100, vae 100
    - [ ] random standard
    - 1b. - [ ] Potentially proceed to testing other combinations on random
    - 1c. - [ ] Potentially proceed to lower dimensionalities (50? 20?)
    - 1d. - [ ] Potentially try out a different margin on one of the models?
2. - [ ] Take the best models, and see if 10/1/1 and 3/3/3 improves things
3. - [ ] Evaluate best models on aggregate user metrics prediction (compared to pretrained)
4. - [ ] Evaluate best models on single-post metric prediction (compared to pretrained)
5. - [ ] Evaluate on masked language modeling (standard model)
6. - [ ] Tidy up baselines set of baselines <br>
Optional: 
- test 1/1/1, standard, on classification task
- try not pretrained


## Context-informed masked language modeling
The refined architectures seem to produce a difference between context types.
Next steps is to systematically understand which ones do best.
There are three main architectures: 
- A **standard transformer** with some normalization layers on top
- A **hierarchical transformer**, which, at each layer, weighs CLS layers against each other to incorporate context information
- A **biencoder**, where context and targets are encoded separately.
Each of these models can be tuned over different parameters.

#### Next steps:
Testing architectures:
1. Standard transformer
    - Aggregation strategy
        - Concatenation
        - Add
    - Pretrained vs. trained vs. from scratch
        - If from scratch, nr layers (3 vs 6)
2. Hierarchical transformer
    - Number of layers (try with 2)
3. Biencoder
    - Number of layers for each architecture
        - 3 x 1 
        - 2 x 2
        - 3 x 3
        - 6 x 1 (with pretrained model)
    - Aggregation
        - Concatenation
        - Add
        - Attention
Each of these should be tested for *author*, *subreddit*, and *random* context, and somehow compared to no-context scenarios.
Next step is to define which architectures and combinations to prioritize.
    

