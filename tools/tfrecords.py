import pandas as pd 
import numpy as np
import tensorflow as tf
from pathlib import Path
import os
import pickle as pkl
from tensorflow.data import TFRecordDataset
from tensorflow.train import BytesList
from tensorflow.train import Example, Features, Feature
from tensorflow.io import FixedLenFeature


FEATURE_DESCRIPTION = {
    'input_ids': FixedLenFeature([], tf.string),
    'attention_mask': FixedLenFeature([], tf.string),
    'one_hot_subreddit': FixedLenFeature([], tf.string),
}

def _make_example_nn1(input_ids, attention_mask, one_hot_subreddit):
    ''' Example serializer'''
    input_ids_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(input_ids).numpy(),
        ])
    )
    attention_mask_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(attention_mask).numpy(),
        ])
    )
    one_hot_subreddit_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(one_hot_subreddit).numpy(),
        ])
    )
    features = Features(feature={
        'input_ids': input_ids_feature,
        'attention_mask': attention_mask_feature,
        'one_hot_subreddit': one_hot_subreddit_feature,
    }) 
    example = Example(features=features)
    return example.SerializeToString()


def _map_fn_nn1(x, y):
    ''' Maps read example function to whole dataset'''
    tf_string = tf.py_function(func=_make_example_nn1, 
                               inp=[x[0], x[1], y], 
                               Tout=tf.string)
    return tf_string


def _reduce_fn_nn1(key, dataset, compression, n_shards, path):
    ''' Shards dataset'''
    str2 = tf.strings.join([tf.strings.as_string(key), 
                            '-of-', str(n_shards-1), '.tfrecord'])
    filename = tf.strings.join([path, str2])    
    writer = tf.data.experimental.TFRecordWriter(filename, 
                                                 compression_type=compression)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)


def save_tfrecord_nn1(dataset, filename=None, 
                      shard=False, n_shards=20, 
                      path='datasets/',
                      compression=None):
    ''' Saves tfrecord using functions defined above'''
    dataset = dataset.map(_map_fn_nn1)
    if shard:
        dataset = dataset.enumerate()
        dataset = dataset.apply(tf.data.experimental.group_by_window(
        lambda i, _: i % n_shards, 
        lambda k, d: _reduce_fn_nn1(k, d, compression, n_shards, path), 
        tf.int64.max
        ))
        for s in dataset:
            print(f'Saving {s} ...')
    else:
        writer = tf.data.experimental.TFRecordWriter(path + filename)
        writer.write(dataset.map(lambda _, x: x))


def _parse_fn_nn1(example):
    ''' Parses each example in TFRecord '''
    example = tf.io.parse_single_example(example, FEATURE_DESCRIPTION)
    input_ids = tf.io.parse_tensor(example['input_ids'], tf.int32)
    attention_mask = tf.io.parse_tensor(example['attention_mask'], tf.int32)
    one_hot_subreddit = tf.io.parse_tensor(example['one_hot_subreddit'], tf.int32)
    return dict(zip(['input_ids', 'attention_mask'], [input_ids, attention_mask])), \
           dict(zip(['one_hot_subreddit'], [one_hot_subreddit]))


def load_tfrecord_nn1(filenames, **kwargs):
    ''' Loads TFRecord'''
    dataset = tf.data.TFRecordDataset(filenames, **kwargs)
    return dataset.map(_parse_fn_nn1)