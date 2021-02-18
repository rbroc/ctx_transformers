import tensorflow as tf
from tensorflow import keras


class TripletLoss:
    ''' Loss function for triplet loss network
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
        padded_size (int): number of posts after padding
    '''
    def __init__(self, margin, padded_size, name=None):
        self.name = name or f'triplet_loss_margin-{margin}_posts-{padded_size}'
        self.margin = margin
        self.padded_size = padded_size

    def _compute_anchor_mean_distance(self, a_encoding, n_posts):   
        ''' Computes mean distance between anchor embeddings '''     
        sqr_enc = tf.reduce_sum(a_encoding*a_encoding, axis=1)
        mask = tf.equal(sqr_enc, 0)
        mask = tf.abs(tf.cast(mask, tf.float32) - 1.0)
        sqr_enc = tf.reshape(sqr_enc, [-1,1])
        dists = sqr_enc - 2*tf.matmul(a_encoding, tf.transpose(a_encoding))
        dists = dists + tf.transpose(sqr_enc)
        dists = tf.transpose(dists * mask) * mask
        dists = tf.linalg.band_part(dists,-1,0)
        n_valid_dists = self._compute_n_distances(mask)
        mean_dist = tf.divide(tf.reduce_sum(dists), n_valid_dists)
        return mean_dist

    def _compute_n_distances(self, mask):
        ''' Compute number of valid pairwise encodings combinations '''
        range_dists = tf.range(1, self.padded_size-1)
        range_valid = tf.cast(range_dists, tf.float32) * mask
        n_valid = tf.reduce_sum(range_valid)
        return n_valid

    def call(self, encodings, n_posts):
        ''' Computes loss. Also returns distances between encodings '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2,:]
        dist_anch = tf.vectorized_map(self._compute_anchor_mean_distance, elems=a_enc)

        avg_a_enc = tf.reduce_sum(a_enc, axis=1) / (n_posts-2)
        dist_pos = tf.reduce_sum(tf.square(avg_a_enc - p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(avg_a_enc - n_enc), axis=1)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = tf.maximum(0.0, self.margin + (dist_pos-dist_neg))

        dist_pos = tf.reduce_mean(dist_pos, axis=0)
        dist_neg = tf.reduce_mean(dist_neg, axis=0)
        loss = tf.reduce_mean(loss, axis=0)
        metric = tf.reduce_mean(metric, axis=0)
        return loss, metric, dist_pos, dist_neg, dist_anch

