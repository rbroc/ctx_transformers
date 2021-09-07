import tensorflow as tf
from tensorflow import keras

class MLMContextMerger(keras.layers.Layer):
    ''' Layer that combines context prior with logits
        Args:
            units (int): number of units
    '''
    def __init__(self, units=1):
        super(MLMContextMerger, self).__init__()
        self.tkernel = self.add_weight(name='tkernel',
                                       shape=(units,), 
                                       initializer="random_normal",
                                       trainable=True)
        #self.ckernel = self.add_weight(name='ckernel',
        #                              shape=(units,), 
        #                              initializer="random_normal",
        #                              trainable=True)
        #self.matchprior = self.add_weight(name='mprior',
        #                                  shape=(units,), 
        #                                  initializer="random_normal",
        #                                  trainable=True)
        self.b = self.add_weight(name='bias',
                                 shape=(units,), 
                                 initializer="zeros", 
                                 trainable=True)
        
    def call(self, inputs):
        t = inputs[0] + inputs[1] # edited
        #c = self.ckernel * inputs[1] # edited
        #match = self.matchprior * logits * cprior # added
        return t + self.b #+c