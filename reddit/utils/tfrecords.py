import tensorflow as tf
from tensorflow.data import TFRecordDataset
from tensorflow.train import Example, Features, Feature, BytesList
from tensorflow.io import FixedLenFeature
import os


# Define feature description
FEATURE_NAMES = ['iids', 'amask', 'pos_iids', 'pos_amask',
                 'neg_iids', 'neg_amask', 'author_id']
FEATURE_TYPES = [FixedLenFeature([], tf.string)]*len(FEATURE_NAMES)
FEATURE_DESCRIPTION = dict(zip(FEATURE_NAMES,
                               FEATURE_TYPES))


def _make_example(iids, amask, pos_iids, pos_amask,
                  neg_iids, neg_amask, author_id):
    ''' Serialize single example '''
    iids_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(iids).numpy(),
        ])
    )
    amask_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(amask).numpy(),
        ])
    )
    pos_iids_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(pos_iids).numpy(),
        ])
    )
    pos_amask_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(pos_amask).numpy(),
        ])
    )
    neg_iids_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(neg_iids).numpy(),
        ])
    )
    neg_amask_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(neg_amask).numpy(),
        ])
    )
    id_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(author_id).numpy(),
        ])
    )
    features_values = [iids_feature, amask_feature,
                       pos_iids_feature, pos_amask_feature,
                       neg_iids_feature, neg_amask_feature,
                       id_feature]
    features = Features(feature=dict(zip(FEATURE_NAMES,
                                         features_values)))
    example = Example(features=features)
    return example.SerializeToString()

    
def _make_examples(*x):
    ''' Maps make_example to whole dataset '''
    tf_string = tf.py_function(func=_make_example, 
                               inp=x, 
                               Tout=tf.string)
    return tf_string


def _shard_fn(k, ds, prefix, path, compression, n_shards):
    ''' Util function to shard dataset at save '''
    str2 = tf.strings.join([os.sep, 
                            prefix, 
                            '-',
                            tf.strings.as_string(k), 
                            '-of-', str(n_shards-1), '.tfrecord'])
    fname = tf.strings.join([str(path), str2])    
    writer = tf.data.experimental.TFRecordWriter(fname, 
                                                 compression_type=compression)
    writer.write(ds.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(fname)


def save_tfrecord(dataset, prefix, path,
                  n_shards=1,
                  compression='GZIP'):
    ''' Saves tfrecord in shards
        Args:
            dataset (TFDataset): dataset to be saved
            prefix (str): prefix for tfrecord file
            path (str or Path): output path for dataset
            n_shards (int): number of shards (defaults to 1)
            compression (str): type of compression to apply
    '''
    dataset = dataset.map(_make_examples).enumerate()
    dataset = dataset.apply(tf.data.experimental.group_by_window(
                            lambda i, _: i % n_shards, 
                            lambda k, d: _shard_fn(k, d, prefix,
                                                   path,
                                                   compression, 
                                                   n_shards), 
                            tf.int64.max )
                            )
    for s in dataset:
        print(f'Saving {s} from {prefix} ...')


def _parse_fn(example):
    ''' Parse examples at load '''
    example = tf.io.parse_single_example(example, 
                                         FEATURE_DESCRIPTION)
    inps = [tf.io.parse_tensor(example[f], tf.int32) 
            for f in FEATURE_NAMES]
    return dict(zip(FEATURE_NAMES, inps))


def load_tfrecord(fnames, 
                  num_parallel_calls=1,
                  deterministic=False,
                  cycle_length=16,
                  compression='GZIP'):
    ''' Loads dataset from tfrecord files
        Args:
            fnames (list): list of filenames for TFRecord
            num_parallel_calls (int): number of parallel reads
            deterministic (bool): does order matter (tradeoff with speed)
            cycle_length (int): number of input elements processed concurrently
            compression (str): type of compression of the target files
    '''
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    dataset = tf.data.Dataset.from_tensor_slices(fnames)
    dataset = dataset.with_options(opts)
    read_fn = lambda x: tf.data.TFRecordDataset(x, 
                                                compression_type=compression)
    dataset = dataset.interleave(read_fn, 
                                 cycle_length=cycle_length, 
                                 num_parallel_calls=num_parallel_calls)
    return dataset.map(_parse_fn, 
                       num_parallel_calls=num_parallel_calls)