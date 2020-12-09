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

    def call(self, input, low_mem=True):
        if low_mem:
            enc = tf.vectorized_map(lambda x: self.model({'input_ids': tf.reshape(x[0],(1,-1)),
                                                          'attention_mask':  tf.reshape(x[1],(1,-1))})[0,0,:],
                            elems=input)
        else:
            enc = self.model(x)
        return enc