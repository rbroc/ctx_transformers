from reddit.utils import (load_tfrecord,
                          split_dataset,
                          triplet_transform, 
                          filter_triplet_by_n_anchors)
from reddit.models import (BatchTransformer, BatchTransformerFFN)
from reddit.losses import (TripletLossBase, TripletLossFFN)
from reddit.training import Trainer
from transformers import TFDistilBertModel
import glob
from pathlib import Path
import argparse
import tensorflow as tf
from official.nlp.optimization import create_optimizer

DATA_PATH = Path('..') /'reddit'/ 'data' / 'datasets'/ 'triplet'

# Initialize parser
parser = argparse.ArgumentParser()

# Training loop argument
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
parser.add_argument('--triplet-type', type=str, default='standard',
                    help='Should be standard or FFN')
parser.add_argument('--log-path', type=str, default=None,
                    help='Path for metrics and checkpoints within ../logs')
parser.add_argument('--per-replica-batch-size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--dataset-size', type=int, default=200000,
                    help='Number of examples in dataset (train + val)')
parser.add_argument('--n-epochs', type=int, default=3,
                    help='Number of epochs')        
parser.add_argument('--start-epoch', type=int, default=0,
                    help='Epoch to start from')
parser.add_argument('--update-every', type=int, default=16,
                    help='Update every n steps')
# Loss arguments
parser.add_argument('--loss-margin', type=float, default=1.0,
                    help='Margin for triplet loss')  
parser.add_argument('--pad-anchor', type=int, default=10,
                    help='Max number of anchor posts')
parser.add_argument('--n-anchor', type=int, default=None,
                    help='Number of anchor posts used in loss')
parser.add_argument('--n-pos', type=int, default=1,
                    help='Number of positive examples')
parser.add_argument('--n-neg', type=int, default=1,
                    help='Number of negative examples')
# Model arguments
parser.add_argument('--pretrained-weights', type=str, 
                    default='distilbert-base-uncased',
                    help='Pretrained huggingface model')
parser.add_argument('--compress-to', type=int, default=None,
                    help='Dimensionality of compression head')
parser.add_argument('--compress-mode', type=str, default=None,
                    help='Whether to compress with dense or vae')
parser.add_argument('--intermediate-size', type=int, default=None,
                    help='Dimensionality of intermediate layer in head')
parser.add_argument('--pooling', type=str, default='cls',
                    help='Whether to compress via pooling or other ways')
# Arguments for FFN triplet
parser.add_argument('--n-dense', type=int, default=None,
                    help='''Number of dense layers to add,
                            relevant for FFN''')
parser.add_argument('--dims', nargs='+', help='Number of nodes in layers', 
                    default=None)
parser.add_argument('--activations', nargs='+', help='Activations in layers', 
                    default=None)
# Define boolean args
parser.add_argument('--test-only', dest='test_only', action='store_true',
                    help='Whether to only run one test epoch')
parser.set_defaults(test_only=False)


def _run_training(log_path, 
                  dataset_name,
                  triplet_type,
                  per_replica_batch_size, 
                  dataset_size,
                  n_epochs,
                  start_epoch,
                  pad_anchor,
                  n_anchor,
                  n_pos,
                  n_neg,
                  loss_margin,
                  pretrained_weights,
                  compress_to,
                  compress_mode,
                  intermediate_size,
                  pooling,
                  n_dense,
                  dims,
                  activations,
                  update_every,
                  test_only):
    
    # Define anchor parameters
    if n_anchor is None:
        n_anchor = pad_anchor

    # Define type of training
    if triplet_type == 'standard':
        model_class = BatchTransformer
        loss = TripletLossBase(margin=loss_margin, 
                               n_pos=n_pos, n_neg=n_neg,
                               n_anc=n_anchor)
    elif triplet_type == 'ffn':
        model_class = BatchTransformerFFN
        loss = TripletLossFFN(margin=loss_margin)
   
    # Config
    METRICS_PATH = Path('..') / 'logs' / 'triplet' / log_path / triplet_type
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
    pattern = str(DATA_PATH / dataset_name / 'train'/ 'batch*')
    fs_train = glob.glob(pattern)
    ds = load_tfrecord(fs_train, deterministic=True, ds_type='triplet')
    # 
    ds_train, ds_val, _ = split_dataset(ds, 
                                        size=dataset_size, 
                                        perc_train=.8, 
                                        perc_val=.2, 
                                        perc_test=.0)
    
    # Compute number of batches
    global_batch_size = len(logical_gpus) * per_replica_batch_size
    n_train_examples = len([e for e in ds_train 
                            if filter_triplet_by_n_anchors(e, n_anchor)])
    n_test_examples = len([e for e in ds_val 
                           if filter_triplet_by_n_anchors(e, n_anchor)])
    print(f'{n_train_examples}, n test examples: {n_test_examples}')
    n_train_steps = int(n_train_examples / global_batch_size)
    n_test_steps = int(n_test_examples / global_batch_size)
    
    # initialize optimizer, model and loss object
    with strategy.scope():
        optimizer = create_optimizer(2e-5, # allow edit
                                     num_train_steps=n_train_steps * n_epochs, # allow edit
                                     num_warmup_steps=n_train_steps / 10) # allow edit
        
        if triplet_type == 'standard':
            model = model_class(transformer=TFDistilBertModel,
                                pretrained_weights=pretrained_weights,
                                trainable=True,
                                output_attentions=False,
                                compress_to=compress_to,
                                compress_mode=compress_mode,
                                intermediate_size=intermediate_size,
                                pooling=pooling)
        elif triplet_type == 'ffn':
            model = model_class(transformer=TFDistilBertModel,
                                pretrained_weights=pretrained_weights,
                                n_dense=n_dense,
                                dims=dims,
                                activations=activations,
                                trainable=False)

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
                      eval_before_training=True,
                      test_steps=n_test_steps,
                      update_every=update_every)

    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val,
                shuffle=True,
                transform=triplet_transform,
                transform_dynamic=False,
                transform_test=True,
                test_only=test_only,
                labels=False, 
                pad_to=[pad_anchor, 
                        n_pos, 
                        n_neg],
                batch_size=global_batch_size,
                min_anchor=n_anchor)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(args.log_path, 
                  args.dataset_name,
                  args.triplet_type,
                  args.per_replica_batch_size, 
                  args.dataset_size,
                  args.n_epochs,
                  args.start_epoch,
                  args.pad_anchor,
                  args.n_anchor,
                  args.n_pos,
                  args.n_neg,
                  args.loss_margin,
                  args.pretrained_weights,
                  args.compress_to,
                  args.compress_mode,
                  args.intermediate_size,
                  args.pooling,
                  args.n_dense,
                  args.dims,
                  args.activations,
                  args.update_every,
                  args.test_only)