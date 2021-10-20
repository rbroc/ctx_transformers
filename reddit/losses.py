from re import L
import tensorflow as tf
from tensorflow import keras
from reddit.utils import (average_encodings, 
                          compute_mean_pairwise_distance)
from abc import ABC, abstractmethod


class TripletLoss(ABC):
    ''' Base class for triplet loss 
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
        custom_loss_fn (function): custom loss function
    '''
    def __init__(self, margin,
                 custom_loss_fn=None, name=None):
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
    def __init__(self, margin, n_neg=1, n_pos=1, n_anc=None, 
                 custom_loss_fn=None, name=None):
        super().__init__(margin, custom_loss_fn, name)
        self.n_neg = n_neg
        self.n_pos = n_pos
        self.n_anc = n_anc
    
    def __call__(self, encodings):
        ''' Computes loss. Returns loss, metric and encodings distances 
        Args:
            encodings (tf.Tensor): posts encodings
        '''       
        neg_idx = self.n_neg
        pos_idx = neg_idx + self.n_pos
        n_enc = encodings[:, :neg_idx, :]
        p_enc = encodings[:, neg_idx:pos_idx, :]
        if self.n_anc:
            a_enc = encodings[:, pos_idx:pos_idx+self.n_anc, :]
        else:
            a_enc = encodings[:, pos_idx:, :]
        dist_anch = tf.vectorized_map(compute_mean_pairwise_distance, elems=a_enc)
        avg_a_enc = tf.squeeze(average_encodings(a_enc), axis=1)
        avg_n_enc = tf.squeeze(average_encodings(n_enc), axis=1)
        avg_p_enc = tf.squeeze(average_encodings(p_enc), axis=1)
        dist_pos = tf.reduce_sum(tf.square(avg_a_enc - avg_p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(avg_a_enc - avg_n_enc), axis=1)
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

    def __call__(self, encodings):
        ''' Computes loss. Returns loss, metric and encodings distances
        Args:
            encodings (tf.Tensor): posts encodings
        '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2,:]
        dist_pos = tf.reduce_sum(tf.square(a_enc - p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(a_enc - n_enc), axis=1)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = self._loss_function(dist_pos, dist_neg)
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, metric, dist_pos, dist_neg]]
        return outs

    
class MLMLoss:
    ''' MLM loss 
    Args:
        from_logits (bool): True if passing logits.
        name (str): identifier.
    '''
    def __init__(self, from_logits=True, name=None):
        self.name = name or 'mlm'
        self.from_logits = from_logits
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=self.from_logits,
                                                                     reduction=tf.keras.losses.Reduction.NONE)
        super().__init__()
   
    
    def _mask_and_reduce(self, target, mask, nr_masked):
        masked = tf.multiply(target, mask)
        reduced = tf.divide(tf.reduce_sum(masked), nr_masked)
        return reduced
    

    def __call__(self, model_outs, labels):
        ''' Computes loss:
        Args:
            model_outs: logits from model (n_batch, n_tokens, vocab_size)
            labels: binary mask with zero if non-masked, true token if masked
        '''  
        # Number of masked items
        mask = tf.cast(tf.math.not_equal(labels, 0), tf.float32)
        nr_masked = tf.cast(tf.math.count_nonzero(labels), tf.float32)
        
        # Compute cross-entropy loss
        losses = self.loss_fn(labels, model_outs)
        entropy = tf.keras.losses.categorical_crossentropy(tf.nn.softmax(model_outs, axis=-1),
                                                           tf.nn.softmax(model_outs, axis=-1), 
                                                           from_logits=False)
        correct = tf.cast(tf.equal(tf.cast(tf.math.argmax(model_outs, 
                                                          axis=-1), tf.int32),
                                   labels),
                          tf.float32) 
        
        # Return outputs
        outs = [self._mask_and_reduce(o, mask, nr_masked) 
                for o in [losses, entropy, correct]]
        return outs
    

class AggregateLoss:
    ''' Loss function for aggregates prediction 
    Args:
        name (str): loss name
        loss_type (str): one of mae, mse, huber
        huber_delta (int): if loss is huber, pass delta here
    '''
    def __init__(self,
                 name=None,
                 loss_type='mse',
                 huber_delta=None):
        self.name = name or f'aggregate-{loss_type}'
        if loss_type == 'mse':
            self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_type == 'mae':
            self.loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        elif loss_type == 'huber':
            self.loss_fn = tf.keras.losses.Huber(delta=huber_delta, 
                                                 reduction=tf.keras.losses.Reduction.NONE)
        else:
            raise ValueError('loss_type must be one of \'mae\', \'mse\', \'huber\'')
        super.__init__()

        
    def __call__(self, model_outs, labels):
        ''' Computes loss:
        Args:
            model_outs: predictions on user-level aggregate variable
            labels: true score for each label
        '''
        losses = self.loss_fn(model_outs, labels)
        outs = [tf.reduce_mean(losses, axis=0)]
        return outs
    