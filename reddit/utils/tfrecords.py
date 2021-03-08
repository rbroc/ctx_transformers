import tensorflow as tf
from tensorflow.data import TFRecordDataset
from tensorflow.train import Example, Features, Feature, BytesList
from tensorflow.io import FixedLenFeature


FEATURE_DESCRIPTION_TRIPLET = {
    'input_ids': FixedLenFeature([], tf.string),
    'attention_mask': FixedLenFeature([], tf.string),
    'id': FixedLenFeature([], tf.string)
    }


def _make_example_triplet(input_ids, attention_mask, id):
    ''' Serialize single example '''
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
    id_feature = Feature(
        bytes_list=BytesList(value=[
            tf.io.serialize_tensor(id).numpy(),
        ])
    )
    features = Features(feature={
        'input_ids': input_ids_feature,
        'attention_mask': attention_mask_feature,
        'id': id_feature
    }) 
    example = Example(features=features)
    return example.SerializeToString()

    
def _map_fn_triplet(x, y):
    ''' Maps make_example to whole dataset'''
    tf_string = tf.py_function(func=_make_example_triplet, 
                               inp=[x[0], x[1], x[2]], 
                               Tout=tf.string)
    return tf_string


def _shard_fn(key, dataset, compression, n_shards, path):
    ''' Shard dataset at saving '''
    str2 = tf.strings.join([tf.strings.as_string(key), 
                            '-of-', str(n_shards-1), '.tfrecord'])
    filename = tf.strings.join([path, str2])    
    writer = tf.data.experimental.TFRecordWriter(filename, 
                                                 compression_type=compression)
    writer.write(dataset.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(filename)


def save_tfrecord_triplet(dataset, 
                          filename=None, 
                          n_shards=1, 
                          path='data',
                          compression='GZIP'):
    ''' Saves tfrecord in shards
        Args:
            dataset (TFDataset): dataset to be saved
            filename (str): prefix for filename of each dataset
            n_shards (int): number of shards (defaults to 1)
            compression (str): type of compression to apply
    '''
    dataset = dataset.map(_map_fn_triplet)
    dataset = dataset.enumerate()
    dataset = dataset.apply(tf.data.experimental.group_by_window(
                            lambda i, _: i % n_shards, 
                            lambda k, d: _shard_fn(k, d, compression, 
                                                   n_shards, path), 
                            tf.int64.max )
                            )
    for s in dataset:
        print(f'Saving {s} ...')


def _parse_fn_triplet(example):
    ''' Parse examples when loading '''
    example = tf.io.parse_single_example(example, FEATURE_DESCRIPTION_TRIPLET)
    iids = tf.io.parse_tensor(example['input_ids'], tf.int32)
    amasks = tf.io.parse_tensor(example['attention_mask'], tf.int32)
    ids = tf.io.parse_tensor(example['id'], tf.int32)
    return dict(zip(['input_ids', 'attention_mask'], [iids, amasks, ids]))


def load_tfrecord_triplet(filenames, 
                          num_parallel_calls=1,
                          deterministic=False,
                          cycle_length=16,
                          compression='GZIP'):
    ''' Loads tfrecord from files 
        Args:
            filenames (list): list of filenames for TFRecord
            num_parallel_calls (int): number of parallel reads
            deterministic (bool): does order matter (tradeoff with speed)
            cycle_length (int): number of input elements processed concurrently
            compression (str): type of compression of the target files
    '''
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(opts)
    read_fn = lambda x: tf.data.TFRecordDataset(x, 
                                                compression_type=compression)
    dataset = dataset.interleave(read_fn, 
                                 cycle_length=cycle_length, 
                                 num_parallel_calls=num_parallel_calls)
    return dataset.map(_parse_fn_triplet, 
                       num_parallel_calls=num_parallel_calls)


