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
    def __init__(self, margin, custom_loss_fn=None, name=None):
        self.name = name or f'triplet_loss_margin-{margin}'
        self.margin = margin
        self.custom_loss_fn = custom_loss_fn
        super().__init__()

    
    def _loss_function(self, dist_pos, dist_neg, dist_anch=None):
        ''' Defines how loss is computed from encoding distances '''
        if self.custom_loss_fn:
            return self.custom_loss_fn(dist_pos, dist_neg, dist_anch)
        else:
            return tf.maximum(0.0, self.margin + (dist_pos-dist_neg))
    
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
    def __init__(self, margin, custom_loss_fn=None, name=None):
        super().__init__(margin, custom_loss_fn, name)
    
    def __call__(self, encodings, n_posts):
        ''' Computes loss. Returns loss, metric and encodings distances 
        Args:
            encodings (tf.Tensor): posts encodings
            n_posts (tf.Tensor): number of user posts before padding
        '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2:,:]
        dist_anch = tf.vectorized_map(compute_mean_pairwise_distance, elems=a_enc)
        avg_a_enc = tf.squeeze(average_anchor(a_enc, n_posts))
        dist_pos = tf.reduce_sum(tf.square(avg_a_enc - p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(avg_a_enc - n_enc), axis=1)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = self._loss_function(dist_pos, dist_neg, dist_anch)
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, metric, dist_pos, dist_neg, dist_anch]]
        return outs


class TripletLossFFN(TripletLoss):
    ''' Triplet loss for batch transformer with FFN head
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
    '''    
    def __init__(self, margin, custom_loss_fn=None, name=None):
        super().__init__(margin, custom_loss_fn, name)

    def __call__(self, encodings, n_posts):
        ''' Computes loss. Returns loss, metric and encodings distances
        Args:
            encodings (tf.Tensor): posts encodings
            n_posts (tf.Tensor): number of user posts before padding
        '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2,:]
        dist_pos = tf.reduce_sum(tf.square(a_enc - p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(a_enc - n_enc), axis=1)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = self._loss_function(dist_pos, dist_neg)
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, metric, dist_pos, dist_neg]]
        return outs
