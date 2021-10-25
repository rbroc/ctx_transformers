import tensorflow as tf
from tensorflow.keras import layers
from reddit.utils import sampling_vae
from tensorflow.keras.layers import (Dense,
                                     Concatenate,
                                     Add,
                                     MultiHeadAttention,
                                     LayerNormalization,
                                     Dropout,
                                     Lambda)
from transformers.modeling_tf_utils import get_initializer
from transformers.models.distilbert.modeling_tf_distilbert import TFMultiHeadSelfAttention


class SimpleCompressor(layers.Layer):
    ''' Compresses encodings through relu layer(s)'''
    def __init__(self, compress_to, intermediate_size=None):
        dropout = Dropout(.20)
        compress = Dense(units=compress_to, activation='relu')
        if intermediate_size:
            intermediate = Dense(units=intermediate_size, 
                                 activation='relu')
            layers = [dropout, intermediate, compress]
        else:
            layers = [dropout, compress]
        self.compressor = tf.keras.models.Sequential(layers)
        super().__init__()

    def call(self, encodings):
        out = self.compressor(encodings)
        return out


class VAECompressor(layers.Layer):
    ''' Compresses encodings through VAE '''
    def __init__(self, 
                 compress_to, 
                 intermediate_size=None,
                 encoder_dim=768,
                 batch_size=1):
        self.compress_to = compress_to
        self.batch_size = batch_size
        if intermediate_size:
            self.encoder_int = Dense(intermediate_size, activation='relu')
        else:
            self.encoder_int = None
        self.z_mean = Dense(compress_to, activation='relu')
        self.z_log_sigma = Dense(compress_to, activation='relu')
        self.z = Lambda(sampling_vae)
        if intermediate_size:
            self.decoder_int = Dense(intermediate_size, activation='relu')
        else:
            self.decoder_int = None
        self.outlayer = Dense(encoder_dim, activation='relu')
        super().__init__()
    
    def call(self, encodings):
        if self.encoder_int:
            x = self.encoder_int(encodings)
        else:
            x = encodings
        zm = self.z_mean(x)
        zls = self.z_log_sigma(x)
        x = self.z([zm, zls, self.compress_to, self.batch_size])
        if self.decoder_int:
            x = self.decoder_int(x)
        out = self.outlayer(x)
        return out


class MLMHead(layers.Layer):
    ''' Wraps mlm head for tidier model '''
    def __init__(self, mlm_model, reset=True):
        super(MLMHead, self).__init__()
        self.vocab_dense = mlm_model.layers[1]
        self.act = mlm_model.act
        self.vocab_layernorm = mlm_model.layers[2]
        self.vocab_projector = mlm_model.layers[-1]
        if reset is True:
            initializer = get_initializer()
            for layer in [self.vocab_dense, 
                          self.vocab_layernorm, 
                          self.vocab_projector]:
                layer.set_weights([initializer(w.shape) 
                                   for w in layer.weights])
    
    def call(self, target):
        x = self.vocab_dense(target)
        x = self.act(x)
        x = self.vocab_layernorm(x)
        x = self.vocab_projector(x)
        return x
    
    
class BatchTransformerContextAggregator(layers.Layer):
    ''' Aggregation layer for standard batch transformer encoder 
    Args:
        agg_fn (str): whether to aggregate by sum (add) or concatenation
            (concat)
        add_dense (int): how many dense layers to add after concatenation
        dims (list): how many dimensions for each added dense layer
        dense_act (str): activation function for added dense layers
        relu_dims (int): how many units for the relu layer (dimensionality 
            of the model)
        
    '''
    def __init__(self, agg_fn='add', 
                 add_dense=0, dims=None, 
                 dense_act='linear', 
                 relu_dims=768):
        super(BatchTransformerContextAggregator, self).__init__()
        assert agg_fn in ['add', 'concat']
        self.agg_layer = Add() if agg_fn == 'add' else Concatenate(axis=-1)
        if add_dense is None:
            add_dense = 0
        self.post_agg_dense = tf.keras.models.Sequential([Dense(units=dims[i], 
                                                                activation=dense_act)
                                                          for i in range(add_dense)] + 
                                                         [Dense(units=relu_dims, activation='relu')])
        self.post_agg_normalizer = LayerNormalization(epsilon=1e-12)
        
    def call(self, hidden_state, norm_ctx):
        aggd = self.agg_layer([hidden_state, norm_ctx])
        aggd = self.post_agg_dense(aggd)
        aggd = self.post_agg_normalizer(aggd + hidden_state)
        return aggd
    
    
class BiencoderSimpleAggregator(BatchTransformerContextAggregator):
    ''' Simple aggregator for Biencoder model '''
    def call(self, target, contexts):
        aggd = self.agg_layer([target, contexts])
        aggd_ffn = self.post_agg_dense(aggd)
        out = self.post_agg_normalizer(aggd_ffn + target)
        return out

            
class BiencoderAttentionAggregator(layers.Layer):
    ''' Attention aggregator for biencoder
    Args:
        num_heads (int): number of attention heads
        model_dim (int): model_dimensionality
        include_head (int): whether to include a layernorm + dense + layernorm
            head (not included in 10/19 training)
    '''
    def __init__(self, num_heads=6, model_dim=768, include_head=True):
        super(BiencoderAttentionAggregator, self).__init__()
        self.agg_layer = MultiHeadAttention(num_heads=num_heads, 
                                            key_dim=key_dim)
        self.include_head = include_head
        if self.include_head:
            self.att_norm = LayerNormalization(epsilon=1e-12)
            self.post_att_ffn = Dense(units=model_dim, activation='relu')
            self.pre_head_norm = LayerNormalization(epsilon=1e-12)
    
    def call(self, target, contexts):
        att_tgt = self.agg_layer(target, contexts)
        if self.include_head:
            att_tgt = self.att_norm(att_tgt + target)
            att_ffn = self.post_attention_ffn(att_tgt)
            out = self.pre_head_normalizer(att_ffn + att_tgt)
            return out
        else:
            return att_tgt
    
    
class HierarchicalAttentionAggregator(layers.Layer):
    ''' Hierarchical attention layer
    Args:
        n_contexts (int): number of contexts
        n_tokens (int): number of tokens
        relu_dims (int): how many units for the relu layer (dimensionality 
            of the model)
    '''
    def __init__(self, n_contexts, n_tokens, config, relu_dims=768):
        super(HierarchicalAttentionAggregator, self).__init__()
        
        self.ctx_transf = TFMultiHeadSelfAttention(config, name="attention")
        self.post_transf_dense = Dense(units=relu_dims, activation='relu')
        self.post_transf_normalizer = LayerNormalization(epsilon=1e-12)
        self.att_mask = tf.constant(1, shape=[1,n_contexts+1])
        self.padding_matrix = [[0,0], [0,n_tokens-1], [0,0]]
        
    def call(self, hidden_state):
        cls_tkn = hidden_state[:,0,:]
        cls_tkn = tf.expand_dims(cls_tkn, axis=0)
        cls_tkn = self.ctx_transf(cls_tkn, cls_tkn, cls_tkn,
                                  self.att_mask, None, False, True)[0][0,:,:]
        cls_tkn = tf.expand_dims(cls_tkn, axis=1)
        cls_tkn = tf.pad(cls_tkn, self.padding_matrix)
        merged = self.post_transf_dense(cls_tkn+hidden_state)
        hidden_state = self.post_transf_normalizer(merged+hidden_state)
        return hidden_state


class ContextPooler(layers.Layer):
    ''' Context pooler for simple batch transformer for context MLM '''
    def __init__(self):
        super(ContextPooler, self).__init__()
        self.normalizer = LayerNormalization(epsilon=1e-12)
    
    def call(self, hidden_state, n_contexts, n_tokens):
        ctx = tf.reduce_mean(hidden_state[:,1:,0,:],
                             axis=1, keepdims=True)
        ctx = self.normalizer(ctx)
        ctx = tf.expand_dims(ctx, axis=2)
        ctx = tf.repeat(ctx, n_contexts+1, axis=1)
        ctx = tf.repeat(ctx, n_tokens, axis=2)
        return ctx


class BiencoderContextPooler(ContextPooler):
    ''' Context pooler for biencoder for context MLM'''
    def __init__(self):
        super(BiencoderContextPooler, self).__init__()
        self.normalizer = LayerNormalization(epsilon=1e-12)
    
    def call(self, contexts, n_tokens): 
        # input bs x n_ctx x 768
        ctx = tf.reduce_mean(contexts, axis=1, keepdims=True) # bs x 1 x 768
        ctx = self.normalizer(ctx)
        ctx = tf.repeat(ctx, n_tokens, axis=1) # bs x 512 x 768
        return ctx
    