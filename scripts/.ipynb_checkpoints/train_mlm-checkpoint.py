from reddit.utils import (load_tfrecord, 
                          split_dataset,
                          mlm_transform,
                          remove_short_targets)
from reddit.models import (BatchTransformerForMLM, 
                           BatchTransformerForContextMLM)
from reddit.losses import MLMLoss
from reddit.training import Trainer
from transformers import (TFDistilBertModel, 
                          TFDistilBertForMaskedLM)
import glob
from pathlib import Path
import tensorflow as tf
from official.nlp.optimization import create_optimizer
import argparse
import os


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
parser.add_argument('--context-type', type=str, default='single',
                   help='Type of grouping')
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
parser.add_argument('--load-encoder-weights', type=str, default=None,
                    help='Path to model weights to load (huggingface version)')
parser.add_argument('--freeze-encoder-layers', nargs='+', default=None,
                    help='Which layers to freeze in the encoder (0 to n-1, with 0 being first)')
parser.add_argument('--add-dense', type=int, default=None,
                    help='''Number of dense layers to add to MLM with context
                            after encoding and concatenating.''')

parser.add_argument('--dims', nargs='+', help='Number of nodes in layers', 
                    default=768)

parser.add_argument('--n-tokens', type=int, default=512,
                    help='Number of tokens in encoding')
parser.add_argument('--context-pooling', type=str, default='cls',
                    help='How to pool context')


# Define boolean args
parser.add_argument('--test-only', dest='test_only', action='store_true',
                    help='Whether to only run one test epoch')
parser.add_argument('--freeze-encoder-false', dest='freeze_encoder', action='store_false',
                    help='Whether to unfreeze the encoder')
parser.add_argument('--freeze-head', dest='freeze_head', action='store_true',
                    help='Whether to freeze classification head')
parser.add_argument('--reset-head', dest='reset_head', action='store_true',
                    help='Whether to reinitialize classification head')
parser.add_argument('--from-scratch', dest='from_scratch', action='store_true',
                    help='Whether to train from scratch')
parser.set_defaults(test_only=False, freeze_head=False, 
                    freeze_encoder=True, reset_head=False,
                    from_scratch=False)


def _run_training(log_path, 
                  dataset_name,
                  context_type,
                  per_replica_batch_size, 
                  dataset_size,
                  n_epochs,
                  start_epoch,
                  load_encoder_weights,
                  freeze_encoder,
                  freeze_encoder_layers,
                  freeze_head,
                  reset_head,
                  add_dense,
                  dims,
                  n_tokens,
                  test_only,
                  context_pooling, 
                  update_every,
                  from_scratch):
    
    # Define type of training
    if context_type == 'single':
        ds_type = 'mlm_simple'
        model_class = BatchTransformerForMLM
        is_context = False
    else:
        ds_type = 'mlm'
        model_class = BatchTransformerForContextMLM
        is_context = True
   
    # Config
    METRICS_PATH = Path('..') / 'logs' / ds_type.split('_')[0] / log_path / context_type 
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
    fs_train = glob.glob(f'../reddit/data/datasets/mlm/{dataset_name}/{context_type}/train/batch*')
    ds = load_tfrecord(fs_train, deterministic=True, ds_type=ds_type)
    ds_train, ds_val, _ = split_dataset(ds, 
                                        size=dataset_size, 
                                        perc_train=.8, 
                                        perc_val=.2, 
                                        perc_test=.0)
    
    
    
    # Compute number of batches
    global_batch_size = len(logical_gpus) * per_replica_batch_size
    n_train_examples = len([e for e in ds_train if remove_short_targets(e, .15)])
    n_test_examples = len([e for e in ds_val if remove_short_targets(e, .15)])
    print(f'{n_train_examples}, n test examples: {n_test_examples}')
    n_train_steps = int(n_train_examples / global_batch_size)
    n_test_steps = int(n_test_examples / global_batch_size)
    
    # initialize optimizer, model and loss object
    with strategy.scope():
        #optimizer = create_optimizer(2e-5,
        #                             num_train_steps=n_train_steps * n_epochs,
        #                             num_warmup_steps=n_train_steps / 10)
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        
        if context_type == 'single':
            model = model_class(transformer=TFDistilBertForMaskedLM,
                                init_weights='distilbert-base-uncased',
                                load_encoder_weights=load_encoder_weights,
                                load_encoder_model_class=TFDistilBertModel,
                                freeze_head=freeze_head,
                                freeze_encoder=freeze_encoder,
                                freeze_encoder_layers=freeze_encoder_layers,
                                reset_head=reset_head)
        else:
            model = model_class(transformer=TFDistilBertForMaskedLM,
                                init_weights='distilbert-base-uncased',
                                load_encoder_weights=load_encoder_weights,
                                load_encoder_model_class=TFDistilBertModel,
                                freeze_head=freeze_head,
                                reset_head=reset_head,
                                freeze_encoder=freeze_encoder,
                                freeze_encoder_layers=freeze_encoder_layers,
                                add_dense=add_dense,
                                dims=dims,
                                n_tokens=n_tokens,
                                context_pooling=context_pooling,
                                batch_size=per_replica_batch_size,
                                from_scratch=from_scratch)
        loss = MLMLoss()
        

    # Initialize trainer
    trainer = Trainer(model=model,
                      loss_object=loss,
                      optimizer=optimizer,
                      strategy=strategy, 
                      n_epochs=n_epochs, 
                      start_epoch=start_epoch,
                      steps_per_epoch=n_train_steps, 
                      log_every=1000,
                      ds_type=ds_type,
                      mlm_type=context_type,
                      log_path=str(METRICS_PATH),
                      checkpoint_device=None,
                      distributed=True,
                      eval_before_training=False, ### EDITED
                      test_steps=n_test_steps,
                      update_every=update_every)

    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val,
                shuffle=False, #### EDITED
                transform=mlm_transform,
                transform_dynamic=True,
                transform_test=True,
                test_only=test_only,
                labels=True, 
                is_context=is_context,
                mask_proportion=.15,
                batch_size=global_batch_size)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(log_path=args.log_path, 
                  dataset_name=args.dataset_name, 
                  context_type=args.context_type,
                  per_replica_batch_size=args.per_replica_batch_size,
                  dataset_size=args.dataset_size,
                  n_epochs=args.n_epochs,
                  start_epoch=args.start_epoch,
                  load_encoder_weights=args.load_encoder_weights,
                  freeze_head=args.freeze_head,
                  reset_head=args.reset_head,
                  freeze_encoder=args.freeze_encoder,
                  freeze_encoder_layers=args.freeze_encoder_layers,
                  add_dense=args.add_dense,
                  dims=args.dims,
                  n_tokens=args.n_tokens,
                  test_only=args.test_only,
                  context_pooling=args.context_pooling,
                  update_every=args.update_every,
                  from_scratch=args.from_scratch)