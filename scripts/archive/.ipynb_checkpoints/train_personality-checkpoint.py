from reddit.utils import (load_tfrecord,
                          split_dataset,
                          personality_transform)
from reddit.models import BatchTransformerForMetrics
from reddit.losses import MetricsLoss
from reddit.training import Trainer
from transformers import TFDistilBertModel
import glob
from pathlib import Path
import argparse
import tensorflow as tf
from official.nlp.optimization import create_optimizer

# Initialize parser
parser = argparse.ArgumentParser()

# Training loop argument
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
parser.add_argument('--log-path', type=str, default=None,
                    help='Path for metrics and checkpoints within ../logs')
parser.add_argument('--per-replica-batch-size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--n-epochs', type=int, default=3,
                    help='Number of epochs')
parser.add_argument('--start-epoch', type=int, default=0,
                    help='Epoch to start from')
parser.add_argument('--update-every', type=int, default=16,
                    help='Update every n steps')
# Model arguments
parser.add_argument('--loss-type', type=str, 
                    default='mse',
                    help='Loss type')
parser.add_argument('--weights', type=str, 
                    default='distilbert-base-uncased',
                    help='Pretrained huggingface model')
parser.add_argument('--metric-type', type=str, 
                    default='personality',
                    help='Should be aggregate or single')
parser.add_argument('--targets', nargs='+',  
                    default=['avg_posts'],
                    help='Which targets to predict')
parser.add_argument('--add-dense', type=int, 
                    default=0,
                    help='''Number of dense layers to add,
                            relevant for FFN''')
parser.add_argument('--dims', nargs='+', 
                    help='Number of nodes in layers', 
                    default=None)
parser.add_argument('--activations', nargs='+', 
                    help='Activations in layers', 
                    default=None)
# Define boolean args
parser.add_argument('--test-only', 
                    dest='test_only', 
                    action='store_true',
                    help='Whether to only run one test epoch')
parser.add_argument('--encoder-trainable', 
                    dest='encoder_trainable', 
                    action='store_true',
                    help='Whether to train the encoder')
parser.set_defaults(test_only=False, encoder_trainable=False)


def _run_training(log_path, 
                  dataset_name,
                  per_replica_batch_size, 
                  n_epochs,
                  start_epoch,
                  weights,
                  metric_type,
                  targets,
                  add_dense,
                  dims,
                  activations,
                  update_every,
                  test_only,
                  encoder_trainable,
                  loss_type):
    

    # Define type of training
    model_class = BatchTransformerForMetrics
    loss = MetricsLoss(loss_type=loss_type)
    mtype = 'personality'
    transform_fn = personality_transform
    DATA_PATH = Path('..') /'reddit'/ 'data' / 'datasets'/ mtype
   
    # Config
    METRICS_PATH = Path('..') / 'logs' / mtype / log_path
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
    pattern = str(DATA_PATH / 'personality' / 'train*')
    v_pattern = str(DATA_PATH / 'personality' / 'val*')
    fs_train = glob.glob(pattern)
    fs_val = glob.glob(v_pattern)
    ds_train = load_tfrecord(fs_train, 
                             deterministic=True, 
                             ds_type=mtype)
    ds_val = load_tfrecord(fs_val, 
                           deterministic=True, 
                           ds_type=mtype)
    
    # Compute number of batches
    global_batch_size = len(logical_gpus) * per_replica_batch_size
    n_train_examples = len([e for e in ds_train])
    n_test_examples = len([e for e in ds_val])
    print(f'{n_train_examples}, n test examples: {n_test_examples}')
    n_train_steps = int(n_train_examples / global_batch_size)
    n_test_steps = int(n_test_examples / global_batch_size)
    
    # initialize optimizer, model and loss object
    with strategy.scope():
        #optimizer = create_optimizer(2e-5, # allow edit
        #                             num_train_steps=n_train_steps * n_epochs, # allow edit
        #                             num_warmup_steps=n_train_steps / 10) # allow edit
        optimizer = tf.keras.optimizers.Adam(1e-2)
        model = model_class(transformer=TFDistilBertModel,
                            weights=weights,
                            metric_type=metric_type,
                            targets=targets,
                            add_dense=add_dense,
                            dims=dims,
                            activations=activations,
                            encoder_trainable=encoder_trainable)

    # Initialize trainer
    trainer = Trainer(model=model,
                      loss_object=loss,
                      optimizer=optimizer,
                      strategy=strategy, 
                      n_epochs=n_epochs, 
                      start_epoch=start_epoch,
                      steps_per_epoch=n_train_steps, 
                      log_every=1000,
                      ds_type=mtype,
                      log_path=str(METRICS_PATH),
                      checkpoint_device=None,
                      distributed=True,
                      eval_before_training=False,
                      test_steps=n_test_steps,
                      update_every=update_every,
                      metric_vars=['losses'] + targets)
    
    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val,
                shuffle=True,
                transform=transform_fn,
                transform_test=True,
                test_only=test_only,
                labels=True,
                batch_size=global_batch_size,
                targets=targets)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(args.log_path, 
                  args.dataset_name,
                  args.per_replica_batch_size,
                  args.n_epochs,
                  args.start_epoch,
                  args.weights,
                  args.metric_type,
                  args.targets,
                  args.add_dense,
                  args.dims,
                  args.activations,
                  args.update_every,
                  args.test_only,
                  args.encoder_trainable,
                  args.loss_type)
