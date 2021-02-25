import tensorflow as tf

def split_dataset(dataset, size=None,
                  perc_train=.7, perc_val=.1, 
                  tuning=None):
    ''' Split dataset into training, validation and test set 
        Args:
            dataset (TFDataset): dataset to split (preprocessed and batched)
            size (int): number of examples. If not provided, 
                it is computed on the fly.
            perc_train (float): percentage of examples in training set
            perc_val (float): percentage of examples in training set
            tuning (optional): if provided, defines number of example for 
                additional tuning dataset.
        Returns:
            tuning, training, valdiation and test set (not distributed yet)
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
    out = tf.reduce_sum(encodings, axis=1, keepdims=1)
    n_posts = tf.expand_dims(n_posts-2, -1)
    n_posts = tf.expand_dims(n_posts, -1)
    out = tf.divide(out, n_posts)
    return out

def compute_mean_pairwise_distance(encodings):   
    ''' Computes mean distance between anchor embeddings '''     
    sqr_enc = tf.reduce_sum(encodings*encodings, axis=1)
    mask = tf.cast(tf.not_equal(sqr_enc, 0), tf.float32)
    sqr_enc = tf.reshape(sqr_enc, [-1,1])
    dists = sqr_enc - 2*tf.matmul(*[encodings]*2, transpose_b=True)
    dists = dists + tf.transpose(sqr_enc)
    dists = tf.transpose(dists * mask) * mask
    dists = tf.linalg.band_part(dists,-1,0)
    dists = dists - tf.linalg.band_part(dists,0,0)
    range_valid = tf.range(mask.shape[-1], dtype=tf.float32) * mask
    n_valid_dists = tf.reduce_sum(range_valid)
    mean_dist = tf.divide(tf.reduce_sum(dists), n_valid_dists)
    return mean_dist