import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, 
                                     Concatenate, 
                                     Lambda)
from reddit.utils import average_anchor


class BatchTransformer(keras.Model):
    ''' Transformer model wrapping HuggingFace transformer to
        support 3D (batch, n_sentences, n_tokens) inputs.
        Args:
            transformer (model): model object from huggingface
                transformers (e.g. TFDistilBertMode)
            path_to_weights (str): path to pretrained weights
            name (str): model name. If not provided, uses path_to_weights
            trainable (bool): whether to freeze weights
    '''
    def __init__(self, transformer, path_to_weights,
                 name=None, trainable=True):
        if name is None:
            name = path_to_weights
        super(BatchTransformer, self).__init__(name=name)
        self.path_to_weights = path_to_weights
        self.encoder = transformer.from_pretrained(path_to_weights)
        self.trainable = trainable
        self.output_signature = tf.float32

    def _encode_batch(self, example):
        mask = tf.reduce_all(tf.equal(example, 0), axis=-1, keepdims=True)
        mask = tf.cast(mask, tf.float32)
        mask = tf.abs(tf.subtract(mask, 1.))
        encoding = self.encoder(example).last_hidden_state[:,0,:]
        masked_encoding = tf.multiply(encoding, mask)
        n_post = tf.reduce_sum(mask)
        return masked_encoding, n_post

    def call(self, input):
        encodings, n_posts = tf.vectorized_map(self._encode_batch, elems=input)
        return encodings, n_posts


class BatchTransformerFFN(BatchTransformer):
    ''' Batch transformer with added dense layers
    Args:
        transformer (model): model object from huggingface
            transformers (e.g. TFDistilBertMode) for batch
            transformer
        path_to_weights (str): path to pretrained weights
        n_dense (int): number of dense layers to add on top
            of batch transformer
        dims (int or list): number of nodes per layer
        activations (str, list or keras activation): type of 
            activation per layer
        trainable (bool): whether to freeze transformer weights
        name (str): model name. If not provided, concatenates
            path_to_weights, n_dense, dim, activation
    '''
    def __init__(self,
                 transformer, path_to_weights, 
                 n_dense=3,
                 dims=768,
                 activations='relu',
                 trainable=False,
                 name=None,
                 **kwargs):
        if name is None:
            name = f'''{path_to_weights}_layers-{n_dense}_'
                       dim-{dim}_{activation}'''
        super().__init__(transformer, path_to_weights, name, 
                         trainable)
        if isinstance(dim, int):
            dims = [dims] * n_dense
        if not isinstance(activations, list):
            activations = [activations] * n_dense
        self.dense_layers = keras.Sequential([Dense(dims[i], activations[i], **kwargs)
                                              for i in range(n_dense)])
        self.average_anchor_layer = Lambda(lambda x: average_anchor(*x))
        self.concat_layer = Concatenate(axis=1)

    def call(self, input):
        encodings, n_posts = super().call(input)
        avg_anchor = self.average_anchor_layer(encodings, n_posts)
        encodings = self.concat_layer([encodings[:,:2,:], avg_anchor])
        encodings = self.dense_layers(encodings)
        return encodings, n_posts