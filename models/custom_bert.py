from tensorflow.keras import layers
import tensorflow as tf

class CustomBertLayer(layers.Layer):
    ''' Custom BERT Layer supporting 4D input 
        Re-used across several models'''
    def __init__(self, model, name=None, trainable=True):
        super(CustomBertLayer, self).__init__(name=name)
        self.model = model
        self.trainable = trainable
        self.output_signature = tf.float32

    def call(self, input):
        enc = tf.map_fn(lambda x: self.model({'input_ids': x[0], 
                                              'attention_mask': x[1]})[0][:,0,:],
                        elems=tuple(input), 
                        dtype=self.output_signature)
        return enc