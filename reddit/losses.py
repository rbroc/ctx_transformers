import tensorflow as tf
from tensorflow import keras
from reddit.utils import (average_anchor, 
                          compute_mean_pairwise_distance)
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

    def __call__(self, encodings, posts):
        ''' Computes loss. Also returns distances between encodings '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2:,:]
        dist_anch = tf.vectorized_map(compute_mean_pairwise_distance, elems=a_enc)
        avg_a_enc = tf.squeeze(average_anchor(a_enc, posts))
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
