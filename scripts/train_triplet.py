from reddit.utils import (load_tfrecord, 
                          pad_and_stack_triplet,
                          split_dataset)
from reddit.models import BatchTransformer
from reddit.losses import TripletLossBase
from reddit.training import Trainer
from transformers import TFDistilBertModel, DistilBertModel
import glob
from pathlib import Path
import tensorflow as tf
from official.nlp.optimization import create_optimizer
import argparse


# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
parser.add_argument('--log-path', type=str, default=None,
                    help='Path for metrics and checkpoints within ../logs')
parser.add_argument('--n-anchor', type=int, default=10,
                    help='Number of anchor posts')
parser.add_argument('--n-pos', type=int, default=1,
                    help='Number of positive examples')
parser.add_argument('--n-neg', type=int, default=1,
                    help='Number of negative examples')
parser.add_argument('--batch-size', type=int, default=3,
                    help='Batch size (coincides with number of GPUs)')
parser.add_argument('--dataset-size', type=int, default=1707606,
                    help='Number of examples in dataset')
parser.add_argument('--loss-margin', type=float, default=1.0,
                    help='Margin for triplet loss')
parser.add_argument('--n-epochs', type=int, default=3,
                    help='number of epochs')



def _run_training(log_path, dataset_name, n_anchor, n_pos, n_neg,
                  batch_size, dataset_size, loss_margin, 
                  n_epochs):
    
    # Config
    METRICS_PATH = Path('..') / 'logs' / log_path
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))

    try:
        tf.config.experimental.set_visible_devices(gpus[1:], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

    # Set up params
    strategy = tf.distribute.MirroredStrategy(devices=logical_gpus)
    ds_params = {'n_anchor': n_anchor,
                 'n_pos': n_pos,
                 'n_neg': n_neg,
                 'batch_size': batch_size}
    
    # Load and build dataset 
    fs = glob.glob(f'../reddit/data/datasets/triplet/{dataset_name}/batch*')
    ds = load_tfrecord(fs, deterministic=True, ds_type='triplet')
    n_batches = int(dataset_size / batch_size)
    ds = pad_and_stack_triplet(ds, pad_to=[ds_params['n_anchor'], 
                                           ds_params['n_pos'],
                                           ds_params['n_neg']]).batch(ds_params['batch_size'], 
                                                                      drop_remainder=True)
    ds_train, ds_val, _ = split_dataset(ds, 
                                        size=n_batches, 
                                        perc_train=.7, 
                                        perc_val=.1, 
                                        perc_test=.2)
    
    # Define training specs
    train_params = {'weights': 'distilbert-base-uncased',
                    'model': TFDistilBertModel,
                    'optimizer_learning_rate': 2e-5,
                    'optimizer_n_train_steps': int(n_batches * .7) * 3,
                    'optimizer_n_warmup_steps': int(n_batches * .7) / 10,
                    'loss_margin': loss_margin,
                    'n_epochs': n_epochs,
                    'steps_per_epoch': int(n_batches * .7),
                    'test_steps': int(n_batches * .1),
                    'train_vars': ['losses','metrics', 
                                   'dist_pos', 'dist_neg', 
                                   'dist_anchor'],
                    'test_vars': ['test_losses', 'test_metrics',
                                  'test_dist_pos', 'test_dist_neg',
                                  'test_dist_anchor'],
                    'log_every': 1000}

    
    # initialize optimizer, model and loss object
    with strategy.scope():
        #optimizer = create_optimizer(train_params['optimizer_learning_rate'],
        #                             num_train_steps=train_params['optimizer_n_train_steps'], 
        #                             num_warmup_steps=train_params['optimizer_n_warmup_steps'])
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
	model = BatchTransformer(train_params['model'], 
                                 train_params['weights'])
        loss = TripletLossBase(train_params['loss_margin'],
                               n_pos=ds_params['n_pos'],
                               n_neg=ds_params['n_neg'])

    # initialize trainer
    trainer = Trainer(model=model,
                      loss_object=loss,
                      optimizer=optimizer,
                      strategy=strategy, 
                      n_epochs=train_params['n_epochs'], 
                      steps_per_epoch=train_params['steps_per_epoch'], 
                      log_every=train_params['log_every'],
                      train_vars=train_params['train_vars'], 
                      test_vars=train_params['test_vars'], 
                      log_path=str(METRICS_PATH),
                      checkpoint_device=None,
                      distributed=True,
                      eval_before_training=True)

    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(args.log_path, 
                  args.dataset_name, 
                  args.n_anchor, args.n_pos, args.n_neg,
                  args.batch_size, 
                  args.dataset_size, 
                  args.loss_margin,
                  args.n_epochs)
