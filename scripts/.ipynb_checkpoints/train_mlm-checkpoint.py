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
from official.nlp.optimization import create_optimizer

DATA_PATH = Path('..') /'reddit'/ 'data' / 'datasets'/ 'mlm'


# Initialize parser
parser = argparse.ArgumentParser()
# Training loop argument
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
parser.add_argument('--context-type', type=str, default='single',
                   help='Type of grouping')
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

# Model arguments
parser.add_argument('--mlm-type', type=str,
                    help='Type of mlm model (standard, hier, biencoder)')
parser.add_argument('--pretrained-weights', type=str, 
                    default=None,
                    help='Pretrained huggingface model')
parser.add_argument('--trained-encoder-weights', type=str, default=None,
                    help='Path to trained encoder weights to load (hf format)')
parser.add_argument('--n-layers', type=int, default=None,
                    help='''Number of transformer layers for 
                            architecture (relevant if not passing 
                            pretrained weights)''')
parser.add_argument('--freeze-encoder', nargs='+', default=None,
                    help='''Which layers to freeze in the encoder 
                            (0 to n-1, with 0 being first)''')
parser.add_argument('--add-dense', type=int, default=None,
                    help='''Number of dense layers to add to MLM with context
                            after concatenation (see model specs)''')
parser.add_argument('--dims', nargs='+', help='Number of nodes in layers', 
                    default=None)
parser.add_argument('--activations', nargs='+', help='Number of activations', 
                    default=None)
parser.add_argument('--n-tokens', type=int, default=512,
                    help='Number of tokens in encoding')
parser.add_argument('--n-contexts', type=int, default=10,
                    help='''Number of contexts passed 
                            (relevant if context models)''')
parser.add_argument('--vocab-size', type=int, default=30522,
                    help='Vocabulary size for the model (if not pretrained)')
parser.add_argument('--aggregate', type=str, default='concat',
                    help='''For context MLM and biencoder model, 
                    how to aggregate target and 
                            context (add or concat for ContextMLM, 
                            + attention for biencoder)''')
parser.add_argument('--n-layers-context-encoder', type=int, default=None,
                    help='''Number of transformer layers for 
                            additional context encoder architecture 
                            in biencoder ''')

# Define boolean args
parser.add_argument('--reset-head', dest='reset_head', action='store_true',
                    help='Whether to reinitialize classification head')
parser.add_argument('--test-only', dest='test_only', action='store_true',
                    help='Whether to only run one test epoch')
parser.set_defaults(test_only=False, reset_head=False)


def _run_training(mlm_type,
                  log_path, 
                  dataset_name,
                  context_type,
                  per_replica_batch_size, 
                  dataset_size,
                  n_epochs,
                  start_epoch,
                  pretrained_weights,
                  trained_encoder_weights,
                  freeze_encoder,
                  reset_head,
                  add_dense,
                  dims,
                  activations,
                  n_contexts,
                  n_tokens,
                  vocab_size,
                  aggregate,
                  n_layers,
                  n_layers_context_encoder, 
                  update_every,
                  test_only):

    # Define type of training
    if context_type == 'single':
        ds_type = 'mlm_simple'
        model_class = BatchTransformerForMLM
        is_context = False
    else:
        ds_type = 'mlm'
        if mlm_type == 'standard':
            model_class = BatchTransformerForContextMLM
        elif mlm_type == 'hier':
            model_class = HierarchicalTransformerForContextMLM
        elif mlm_type == 'biencoder':
            model_class = BiencoderForContextMLM
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
    pattern = str(DATA_PATH/ dataset_name / context_type / 'train'/ 'batch*')
    fs_train = glob.glob(pattern)
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
        optimizer = create_optimizer(2e-5, # allow edit
                                     num_train_steps=n_train_steps * n_epochs, # allow edit
                                     num_warmup_steps=50000) # could change
        
        if context_type == 'single':
            model = model_class(transformer=TFDistilBertForMaskedLM,
                                pretrained_weights=pretrained_weights,
                                trained_encoder_weights=trained_encoder_weights,
                                trained_encoder_class=TFDistilBertModel,
                                n_layers=n_layers,
                                freeze_encoder=freeze_encoder,
                                reset_head=reset_head,
                                vocab_size=vocab_size)
        else:
            if mlm_type == 'standard':
                model = model_class(transformer=TFDistilBertForMaskedLM,
                                    pretrained_weights=pretrained_weights,
                                    trained_encoder_weights=trained_encoder_weights,
                                    trained_encoder_class=TFDistilBertModel,
                                    n_layers=n_layers,
                                    freeze_encoder=freeze_encoder,
                                    reset_head=reset_head,
                                    add_dense=add_dense,
                                    dims=dims,
                                    activations=activations,
                                    n_tokens=n_tokens,
                                    aggregate=aggregate,
                                    n_contexts=n_contexts,
                                    vocab_size=vocab_size)
            elif mlm_type == 'hier':
                model = model_class(transformer=TFDistilBertForMaskedLM,
                                    n_layers=n_layers,
                                    n_tokens=n_tokens,
                                    n_contexts=n_contexts,
                                    vocab_size=vocab_size)
            elif mlm_type == 'biencoder':
                model = model_class(transformer=TFDistilBertForMaskedLM,
                                    pretrained_token_encoder_weights=pretrained_weights,
                                    trained_token_encoder_weights=trained_encoder_weights,
                                    trained_token_encoder_class=TFDistilBertModel,
                                    n_layers_token_encoder=n_layers,
                                    n_layers_context_encoder=n_layers_context_encoder,
                                    freeze_token_encoder=freeze_encoder,
                                    add_dense=add_dense,
                                    dims=dims,
                                    activations=activations,
                                    n_tokens=n_tokens,
                                    aggregate=aggregate,
                                    n_contexts=n_contexts,
                                    vocab_size=vocab_size)
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
                      log_path=str(METRICS_PATH),
                      checkpoint_device=None,
                      distributed=True,
                      eval_before_training=False,
                      test_steps=n_test_steps,
                      update_every=update_every)

    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val,
                shuffle=False, # saving time
                transform=mlm_transform,
                transform_test=True,
                test_only=test_only,
                labels=True, 
                is_context=is_context,
                mask_proportion=.15,
                batch_size=global_batch_size)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(mlm_type=args.mlm_type,
                  log_path=args.log_path, 
                  dataset_name=args.dataset_name,
                  context_type=args.context_type,
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