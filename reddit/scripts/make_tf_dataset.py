import pandas as pd
import numpy as np
from reddit.utils import (read_files,
                          save_tfrecord_triplet)
from pathlib import Path
import json
import tensorflow as tf
import argparse

ENCODED_DATA_PATH = Path('..') / 'data' / 'triplet'
DATASET_PATH = Path('..') / 'data' / 'datasets' / 'triplet'
DATASET_PATH.mkdir(exist_ok=True, parents=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n-shards', type=int, default=1000,
                    help='Number of files across which '
                    'TFRecord will be sharded.')

def _generate_example(df):
    for u in df.target_user.unique():
        udf = df[df['target_user'] == u]
        anchor = udf[udf['example_type']=='anchor']
        pos = udf[udf['example_type']=='positive']
        neg = udf[udf['example_type']=='negative']
        anchor_iids = anchor['input_ids'].values
        anchor_mask = anchor['attention_mask'].values
        pos_iids = pos['input_ids'].values
        pos_mask = pos['attention_mask'].values
        neg_iids = neg['input_ids'].values
        neg_mask = neg['attention_mask'].values
        # Stack
        iids = np.vstack([neg_iids, pos_iids, anchor_iids])
        masks = np.vstack([neg_mask, pos_mask, anchor_mask])
        # Return
        yield(iids, masks, np.array(u))

def make_tensorflow_dataset(n_shards=1000):
    df = read_files(ENCODED_DATA_PATH, 
                    converters=dict(zip(['input_ids', 'attention_mask'],
                                        [lambda x: json.loads(x)]*2)),
                    drop_duplicates=False)
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(df),
                                        output_types=tuple([tf.int32] * 3))
    save_tfrecord_triplet(ds, n_shards=n_shards,
                          path=str(DATASET_PATH),
                          compression='GZIP')

if __name__=="__main__":
    args = parser.parse_args()
    make_tensorflow_dataset(args.n_shards)