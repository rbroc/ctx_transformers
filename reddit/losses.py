import tensorflow as tf
from tensorflow import keras
from reddit.utils import average_anchor
from abc import ABC, abstractmethod


class TripletLoss(ABC):
    ''' Base class for triplet loss 
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
    '''
    def __init__(self, margin, name=None):
        self.name = name or f'triplet_loss_margin-{margin}'
        self.margin = margin
        super().__init__()
    
    @abstractmethod
    def __call__(self):
        ''' Computes loss '''
        pass


class TripletLossBase(TripletLoss):
    ''' Triplet loss for BatchTransformer with no head
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
    '''
    def __init__(self, margin, name=None):
        super().__init__(margin, name)

    def _compute_anchor_mean_distance(self, a_encoding):   
        ''' Computes mean distance between anchor embeddings '''     
        sqr_enc = tf.reduce_sum(a_encoding*a_encoding, axis=1)
        mask = tf.equal(sqr_enc, 0)
        mask = tf.abs(tf.cast(mask, tf.float32) - 1.0)
        sqr_enc = tf.reshape(sqr_enc, [-1,1])
        dists = sqr_enc - 2*tf.matmul(a_encoding, tf.transpose(a_encoding))
        dists = dists + tf.transpose(sqr_enc)
        dists = tf.transpose(dists * mask) * mask
        dists = tf.linalg.band_part(dists,-1,0)
        padded_size = a_encoding.shape[-2]
        n_valid_dists = self._compute_n_distances(mask, padded_size)
        mean_dist = tf.divide(tf.reduce_sum(dists), n_valid_dists)
        return mean_dist

    def _compute_n_distances(self, mask, padded_size):
        ''' Compute number of valid pairwise encodings combinations '''
        range_dists = tf.range(1, padded_size-1)
        range_valid = tf.cast(range_dists, tf.float32) * mask
        n_valid = tf.reduce_sum(range_valid)
        return n_valid

    def __call__(self, encodings, n_posts):
        ''' Computes loss. Also returns distances between encodings '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2:,:]
        dist_anch = tf.vectorized_map(self._compute_anchor_mean_distance, elems=a_enc) # how does this work?
        avg_a_enc = average_anchor(encodings, n_posts)
        dist_pos = tf.reduce_sum(tf.square(avg_a_enc - p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(avg_a_enc - n_enc), axis=1)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = tf.maximum(0.0, self.margin + (dist_pos-dist_neg))
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, metric, dist_pos, dist_neg, dist_anch]]
        return outs


class TripletLossFFN(TripletLoss):
    ''' Triplet loss for batch transformer with FFN head
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
    '''    
    def __init__(self, margin, name=None):
        super().__init__(margin, name)

    def __call__(self, encodings, posts):
        ''' Computes loss '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2,:]
        dist_pos = tf.reduce_sum(tf.square(a_enc - p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(a_enc - n_enc), axis=1)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = tf.maximum(0.0, self.margin + (dist_pos-dist_neg))
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, metric, dist_pos, dist_neg]]
        return outs
