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
from IPython.display import clear_output
import abc


FEATURE_DESCRIPTION_TRIPLET = {
    'input_ids': FixedLenFeature([], tf.string),
    'attention_mask': FixedLenFeature([], tf.string)
}

def _make_example_triplet_nn1(input_ids, attention_mask):
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
    features = Features(feature={
        'input_ids': input_ids_feature,
        'attention_mask': attention_mask_feature
    }) 
    example = Example(features=features)
    return example.SerializeToString()

    
def _map_fn_triplet_nn1(x, y):
    ''' Maps read example function to whole dataset'''
    tf_string = tf.py_function(func=_make_example_triplet_nn1, 
                               inp=[x[0], x[1]], 
                               Tout=tf.string)
    return tf_string


def save_tfrecord_triplet_nn1(dataset, filename=None, 
                      shard=False, n_shards=20, 
                      path='datasets/test/',
                      compression=None):
    ''' Saves tfrecord using functions defined above'''
    dataset = dataset.map(_map_fn_triplet_nn1)
    if shard:
        dataset = dataset.enumerate()
        dataset = dataset.apply(tf.data.experimental.group_by_window(
        lambda i, _: i % n_shards, 
        lambda k, d: _reduce_fn(k, d, compression, n_shards, path), 
        tf.int64.max
        ))
        for s in dataset:
            clear_output(wait=True)
            print(f'Saving {s} ...')
    else:
        writer = tf.data.experimental.TFRecordWriter(path + filename)
        writer.write(dataset.map(lambda _, x: x))

def _parse_fn_triplet_nn1(example):
    example = tf.io.parse_single_example(example, FEATURE_DESCRIPTION_TRIPLET)
    input_ids = tf.io.parse_tensor(example['input_ids'], tf.int32)
    attention_mask = tf.io.parse_tensor(example['attention_mask'], tf.int32)
    return dict(zip(['input_ids', 'attention_mask'], [input_ids, attention_mask]))


def load_tfrecord_triplet_nn1(filenames, **kwargs):
    dataset = tf.data.TFRecordDataset(filenames, **kwargs)
    return dataset.map(_parse_fn_triplet_nn1)



def _reduce_fn(key, dataset, compression, n_shards, path):
    ''' Shards dataset'''
    str2 = tf.strings.join([tf.strings.as_string(key), 
                            '-of-', str(n_shards-1), '.tfrecord'])
    filename = tf.strings.join([path, str2])    
    writer = tf.data.experimental.TFRecordWriter(filename, 
                                                 compression_type=compression)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)


######################## DRAFT STRUCTURE ########################
# Consider making tools/tfrecords subfolder with base classes and specific classes

class TFRecordMaker(metaclass=abc.ABCMeta):
    ''' Metaclass to save TFrecords '''

    def __init__(self, fspec, dataset):
        self.fspec = fspec
        self.dataset = dataset

    @abc.abstractmethod
    def make_example(self):
        pass

    @abc.abstractmethod
    def map_fn(self):
        pass


    def _reduce_fn(self, key,
                   compression, 
                   n_shards, path):
        fname = tf.strings.join([tf.strings.as_string(key), 
                                '-of-', 
                                str(n_shards-1), 
                                '.tfrecord'])
        fpath = tf.strings.join([path, fname])    
        writer = tf.data.experimental.TFRecordWriter(fpath, 
                                                     compression_type=compression)
        writer.write(self.dataset.map(lambda _, x: x))
        return tf.data.Dataset.from_tensors(fpath)
    
    @abc.abstractmethod
    def save_tfrecord(self):
        pass


class TFRecordLoader(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def parse_fn(self):
        pass

    @abc.abstractmethod
    def return_tfrecord(self):
        pass
