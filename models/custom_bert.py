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

    @tf.function
    def _enc_loop(self, x):
        c = 0
        enc = tf.TensorArray(tf.float32, size=0, 
                            dynamic_size=True)
        inptup = tf.stack([x[0], x[1]], axis=1)
        for i in inptup:
            ienc = self.model({'input_ids':
                                tf.reshape(i[0], [1, -1]), 
                                'attention_mask': 
                                tf.reshape(i[1], [1, -1])})[0][:,0,:]
            enc = enc.write(c, ienc[0])
            c += 1
        return enc.stack()

    def call(self, input, low_mem=True):
        if low_mem:
            enc = tf.map_fn(self._enc_loop, elems=tuple(input), 
                            fn_output_signature=self.output_signature,
                            parallel_iterations=10)
        else:
            enc = tf.map_fn(lambda x: self.model({'input_ids': x[0], 
                                              'attention_mask': x[1]})[0][:,0,:],
                        elems=tuple(input), 
                        fn_output_signature=self.output_signature)
        return enc