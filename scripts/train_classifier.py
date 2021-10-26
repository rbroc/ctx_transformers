from reddit.utils import (load_tfrecord,
                          split_dataset,
                          classification_transform)
from reddit.models import BatchTransformerClassifier
from reddit.losses import ClassificationLoss
from reddit.training import Trainer
from transformers import TFDistilBertModel
import glob
from pathlib import Path
import argparse
import tensorflow as tf
from official.nlp.optimization import create_optimizer


DATA_PATH = Path('..') /'reddit'/ 'data' / 'datasets'/ 'classification'


# Initialize parser
parser = argparse.ArgumentParser()


# Training loop argument
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
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
parser.add_argument('--nposts', type=int, default=None,
                    help='Number of posts')
# Model arguments
parser.add_argument('--pretrained-weights', type=str, 
                    default='distilbert-base-uncased',
                    help='Pretrained huggingface model')
parser.add_argument('--trained-encoder-weights', type=str, default=None,
                    help='Path to trained encoder weights to load (hf format)')
parser.add_argument('--compress-to', type=int, default=None,
                    help='Dimensionality of compression head')
parser.add_argument('--compress-mode', type=str, default=None,
                    help='Whether to compress with dense or vae')
parser.add_argument('--intermediate-size', type=int, default=None,
                    help='Dimensionality of intermediate layer in head')
parser.add_argument('--pooling', type=str, default='cls',
                    help='Whether to compress via pooling or other ways')
parser.add_argument('--vocab-size', type=int, default=30522,
                    help='Vocab size (relevant if newn architecture')
parser.add_argument('--n-layers', type=int, default=None,
                    help='Nr layers if not pretrained')
# Define boolean args
parser.add_argument('--test-only', dest='test_only', action='store_true',
                    help='Whether to only run one test epoch')
parser.set_defaults(test_only=False)


def _run_training(log_path, 
                  dataset_name,
                  per_replica_batch_size, 
                  dataset_size,
                  n_epochs,
                  start_epoch,
                  nposts,
                  pretrained_weights,
                  trained_encoder_weights,
                  compress_to,
                  compress_mode,
                  intermediate_size,
                  pooling,
                  update_every,
                  test_only,
                  vocab_size, 
                  n_layers):
    
    
    # Define type of training
    model_class = BatchTransformerClassifier
    loss = ClassificationLoss()
   
    # Config
    METRICS_PATH = Path('..') / 'logs' / 'classification' / log_path
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
    ds = load_tfrecord(fs_train, 
                       deterministic=True, 
                       ds_type='classification')
    ds_train, ds_val, _ = split_dataset(ds, 
                                        size=dataset_size, 
                                        perc_train=.8, 
                                        perc_val=.2, 
                                        perc_test=.0)
    
    # Compute number of batches
    global_batch_size = len(logical_gpus) * per_replica_batch_size
    n_train_examples = len([e for e in ds_train])
    n_test_examples = len([e for e in ds_val])
    print(f'{n_train_examples}, n test examples: {n_test_examples}')
    n_train_steps = int(n_train_examples / global_batch_size)
    n_test_steps = int(n_test_examples / global_batch_size)
    with strategy.scope():
        optimizer = create_optimizer(2e-5, # allow params edit
                                     num_train_steps=n_train_steps * n_epochs,
                                     num_warmup_steps=10000)
        
        model = model_class(nposts=nposts,
                            transformer=TFDistilBertModel,
                            pretrained_weights=pretrained_weights,
                            trained_encoder_weights=trained_encoder_weights,
                            trained_encoder_class=TFDistilBertModel,
                            trainable=True,
                            compress_to=compress_to,
                            compress_mode=compress_mode,
                            intermediate_size=intermediate_size,
                            pooling=pooling,
                            vocab_size=vocab_size,
                            n_layers=n_layers)

    # Initialize trainer
    trainer = Trainer(model=model,
                      loss_object=loss,
                      optimizer=optimizer,
                      strategy=strategy, 
                      n_epochs=n_epochs, 
                      start_epoch=start_epoch,
                      steps_per_epoch=n_train_steps, 
                      log_every=1000,
                      ds_type='classification',
                      log_path=str(METRICS_PATH),
                      checkpoint_device=None,
                      distributed=True,
                      eval_before_training=False, # edited
                      test_steps=n_test_steps,
                      update_every=update_every)
    
    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val,
                shuffle=False, # edited
                transform=classification_transform,
                transform_test=True,
                test_only=test_only,
                labels=True,
                batch_size=global_batch_size,)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(args.log_path, 
                  args.dataset_name,
                  args.per_replica_batch_size, 
                  args.dataset_size,
                  args.n_epochs,
                  args.start_epoch,
                  args.nposts,
                  args.pretrained_weights,
                  args.trained_encoder_weights,
                  args.compress_to,
                  args.compress_mode,
                  args.intermediate_size,
                  args.pooling,
                  args.update_every,
                  args.test_only,
                  args.vocab_size,
                  args.n_layers)
