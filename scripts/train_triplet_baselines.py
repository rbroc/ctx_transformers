from reddit.utils import (load_tfrecord,
                          split_dataset,
                          triplet_baselines_transform)
from reddit.models import BatchTransformerForMetrics
from reddit.losses import SubredditClassificationLoss
from reddit.training import Trainer
import tensorflow as tf
from transformers import TFDistilBertModel
import glob
from pathlib import Path
import argparse
import tensorflow as tf
import numpy as np
from official.nlp.optimization import create_optimizer

# Initialize parser
parser = argparse.ArgumentParser()

# Training loop argument
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
parser.add_argument('--log-path', type=str, default=None,
                    help='Path for metrics and checkpoints within ../logs')
parser.add_argument('--n-epochs', type=int, default=3,
                    help='Number of epochs')
# Model arguments
parser.add_argument('--pad-to', type=int, default=3)
parser.add_argument('--nr', type=int, default=3)
parser.add_argument('--n-layers', type=int, default=1)


def _run_training(log_path, 
                  dataset_name,
                  n_epochs,
                  pad_to,
                  nr,
                  n_layers):
    
    mtype = 'triplet_baselines'
    DATA_PATH = Path('..') /'reddit'/ 'data' / 'datasets'/ mtype
    METRICS_PATH = Path('..') / 'logs' / mtype / dataset_name / log_path
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Set up dataset 
    fs_train, fs_val, fs_test = [glob.glob(str(DATA_PATH / dataset_name / split / 'batch*')) 
                                 for split in ['train', 'val', 'test']]
    ds_train = load_tfrecord(fs_train, 
                             deterministic=True, 
                             ds_type=mtype)
    ds_val = load_tfrecord(fs_val, 
                           deterministic=True, 
                           ds_type=mtype)
    ds_test = load_tfrecord(fs_test, 
                            deterministic=True, 
                            ds_type=mtype)
    input_shape = [i['iids'].shape[0] for i in ds_train.take(1)][0]
    ds_train = triplet_baselines_transform(ds_train, pad_to=pad_to, 
                                           nr=nr, batch_size=64, dedict=True)
    ds_val = triplet_baselines_transform(ds_val, pad_to=pad_to, 
                                         nr=nr, batch_size=64, dedict=True)
    ds_test = triplet_baselines_transform(ds_test, pad_to=pad_to, 
                                          nr=nr, batch_size=64, dedict=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5) # modulate learning rate
    model = tf.keras.Sequential()
    if n_layers == 1:
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dropout(.5))
    else:
        model.add(tf.keras.layers.Dense(256, activation="relu"))
        model.add(tf.keras.layers.Dropout(.5))
        model.add(tf.keras.layers.Dense(128, activation="relu"))
        model.add(tf.keras.layers.Dropout(.5))
    model.add(tf.keras.layers.Dense(1), activation="sigmoid")
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = tf.keras.metrics.BinaryAccuracy()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    model.fit(ds_train,
              epochs=n_epochs,
              callbacks=None,
              shuffle=True)
    train_outs = model.evaluate(ds_outs)
    val_outs = model.evaluate(ds_val)
    test_outs = model.evaluate(ds_test)
    train_dict = dict(zip([f'train_{m}' for m in model.metrics_names], train_outs))
    val_dict = dict(zip([f'val_{m}' for m in model.metrics_names], val_outs))
    test_dict = dict(zip([f'test{m}' for m in model.metrics_names], test_outs))
    train_dict.update(val_dict)
    train_dict.update(test_dict)
    with open(str(METRICS_PATH / 'metrics.json'), 'w') as json_file:
          json.dump(train_dict, json_file)
    
if __name__=='__main__':
    args = parser.parse_args()
    _run_training(args.log_path, 
                  args.dataset_name,
                  args.n_epochs,
                  args.pad_to,
                  args.nr,
                  args.n_layers)
