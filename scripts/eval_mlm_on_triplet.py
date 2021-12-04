from reddit.utils import (load_tfrecord, 
                          split_dataset,
                          mlm_transform,
                          remove_short_targets)
from reddit.models import (BatchTransformerForMLM, 
                           BatchTransformerForContextMLM, 
                           BiencoderForContextMLM, 
                           HierarchicalTransformerForContextMLM)
from reddit.losses import MLMLoss
from reddit.training import Trainer
from transformers import (TFDistilBertModel, 
                          TFDistilBertForMaskedLM)
import glob
from pathlib import Path
import argparse
import tensorflow as tf
import itertools
from official.nlp.optimization import create_optimizer

DATA_PATH = Path('..') /'reddit'/ 'data' / 'datasets'/ 'triplet'


# Make model plotting utils! 
# Set separable
# Log model input args!

# if biencoder, choose whether to use ctx or target encoder - handle aggregation
# if hierarchical, use full thing - aggregation
# if standard, use full thing - aggregation?

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default=None,
                    help='Path to pretrained model')
parser.add_argument('--eval-dataset-name', type=str, default=None,
                    help='Name of triplet dataset to use')
parser.add_argument('--log-path', type=str, default=None,
                    help='Path for metrics and checkpoints within ../logs')
parser.add_argument('--per-replica-batch-size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--dataset-size', type=int, default=100000,
                    help='Number of examples in dataset (train + val)')
parser.add_argument('--n-epochs', type=int, default=3,
                    help='Number of epochs')
parser.add_argument('--start-epoch', type=int, default=0,
                    help='Epoch to start from')
parser.add_argument('--update-every', type=int, default=16,
                    help='Update every n steps')
parser.add_argument('--model-type', type=str,
                    help='Type of model (standard, hier, biencoder, combined)')
parser.add_argument('--grouping', type=str, default='author',
                    help='Whether evaluating triplet on group or subreddit')
parser.add_argument('--test-only', dest='test_only', action='store_true',
                    help='Whether to only run one test epoch')
parser.set_defaults(test_only=False)


def _run_training(model_path,
                  dataset_name,
                  log_path, 
                  per_replica_batch_size, 
                  dataset_size, # note that we only need to test 
                  n_epochs,
                  start_epoch,
                  update_every,
                  model_type,
                  grouping,
                  test_only):

    # Define type of training
    model = keras.load_model(model_path)
   
    # Config
    METRICS_PATH = Path('..') / 'logs' / 'mlm_eval' / log_path / grouping
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))

    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

    strategy = tf.distribute.MirroredStrategy(devices=logical_gpus)
    
    # Set up dataset 
    pattern = str(DATA_PATH/ dataset_name / 'train'/ 'batch*')
    fs = glob.glob(pattern)
    ntr = int(dataset_size * .8 / 10000)
    nval = int(dataset_size * .2 / 10000)
    fs_train = list(itertools.chain(*[(fs[i], 
                                       fs[i+200], 
                                       fs[i+400]) 
                                      for i in range(int(ntr/3))]))
    fs_val = list(itertools.chain(*[(fs[int(ntr/3)+1], 
                                     fs[int(ntr/3)+200], 
                                     fs[int(ntr/3)+1+400]) 
                                      for i in range(int(nval/3))]))
    print(len(fs_train))
    print(len(fs_val))
    
    
    ds_train = load_tfrecord(fs_train, deterministic=False, ds_type='triplet')
    ds_val = load_tfrecord(fs_val, deterministic=False, ds_type='triplet')
    
    # Compute number of batches
    global_batch_size = len(logical_gpus) * per_replica_batch_size
    print(f'{n_train_examples}, n test examples: {n_test_examples}')
    n_train_steps = int(n_train_examples / global_batch_size)
    n_test_steps = int(n_test_examples / global_batch_size)
    
    # initialize optimizer, model and loss object
    with strategy.scope():
        optimizer = create_optimizer(2e-5, # allow edit
                                     num_train_steps=n_train_steps * 5, #* n_epochs,1
                                     num_warmup_steps=n_train_steps * 5 / 10) # n_epochs / 10) 
        
        # May need to load the model again here
        # load_model
        
        # pass the model to class which only keeps triplet-relevant evaluation
        if model_type == 'biencoder':
            pass
        elif model_type == 'standard':
            pass
        elif model_type == 'hierarchical':
            pass
        elif model_type == 'single':
            pass
        else:
            pass
        
        # Set margin loss (note that we could do classification too!)
        loss = TripletLoss(margin=1.0)
        
    
    # Initialize trainer
    trainer = Trainer(model=model,
                      loss_object=loss,
                      optimizer=optimizer,
                      strategy=strategy, 
                      n_epochs=n_epochs, 
                      start_epoch=start_epoch,
                      steps_per_epoch=n_train_steps, 
                      log_every=1000,
                      ds_type='triplet',
                      log_path=str(METRICS_PATH),
                      checkpoint_device=None,
                      distributed=True,
                      eval_before_training=False,
                      test_steps=n_test_steps,
                      update_every=update_every)

    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val,
                shuffle=False,#True, # saving time
                transform=mlm_transform,
                transform_test=True,
                test_only=test_only,
                labels=True, 
                is_context=True,
                mask_proportion=.15,
                batch_size=global_batch_size,
                is_combined=True)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(mlm_type=args.mlm_type,
                  log_path=args.log_path, 
                  dataset_name=args.dataset_name,
                  per_replica_batch_size=args.per_replica_batch_size, 
                  dataset_size=args.dataset_size,
                  n_epochs=args.n_epochs,
                  start_epoch=args.start_epoch,
                  pretrained_weights=args.pretrained_weights,
                  trained_encoder_weights=args.trained_encoder_weights,
                  freeze_encoder=args.freeze_encoder,
                  reset_head=args.reset_head,
                  add_dense=args.add_dense,
                  dims=args.dims,
                  activations=args.activations,
                  n_contexts=args.n_contexts,
                  n_tokens=args.n_tokens,
                  vocab_size=args.vocab_size,
                  aggregate=args.aggregate,
                  n_layers=args.n_layers,
                  n_layers_context_encoder=args.n_layers_context_encoder, 
                  update_every=args.update_every,
                  test_only=args.test_only)