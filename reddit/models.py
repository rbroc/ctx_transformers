import tensorflow as tf
from tensorflow import keras


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
        self.model = transformer.from_pretrained(path_to_weights)
        self.trainable = trainable
        self.output_signature = tf.float32

    def _encode(self, example):
        mask = tf.reduce_all(tf.equal(example, 0), axis=-1, keepdims=True)
        mask = tf.cast(mask, tf.float32)
        mask = tf.abs(tf.subtract(mask, 1.))
        encoding = self.model(example).last_hidden_state[:,0,:]
        masked_encoding = tf.multiply(encoding, mask)
        n_post = tf.reduce_sum(mask)
        return masked_encoding, n_post

    def call(self, input):
        encodings, n_posts = tf.vectorized_map(self._encode, elems=input)
        return encodings, n_posts
