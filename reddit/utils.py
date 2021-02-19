import tensorflow as tf

def split_dataset(dataset, size=None,
                  perc_train=.7, perc_val=.1, 
                  tuning=None):
    ''' Split dataset into training, validation and test set 
        Args:
            dataset (TFDataset): dataset to split
            size (int): number of examples. If not provided, 
                it is computed on the fly.
            perc_train (float): percentage of examples in training set
            perc_val (float): percentage of examples in training set
            tuning (optional): if provided, defines number of example for 
                additional tuning dataset.
    ''' 
    if size is None:
        size = 0
        for _ in dataset:
            size += 1
    size_train = int(size * perc_train)
    size_val = int(size * perc_val)
    d_train = dataset.take(size_train)
    d_test = dataset.skip(size_train + size_val)
    d_val = dataset.skip(size_train).take(size_val)
    if tuning is None:
        return d_train, d_val, d_test
    else:
        d_tuning = dataset.take(tuning)
        return d_tuning, d_train, d_val, d_test

def average_anchor(encodings, n_posts):
    out = tf.reduce_sum(encodings[:,2:,:], axis=1)
    out = tf.divide(out, n_posts-2)
    return out
