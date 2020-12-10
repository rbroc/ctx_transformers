from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, Input
from transformers import TFDistilBertModel
from pathlib import Path

# Define path variable
FIG_PATH = Path('figures')
FIG_FILE = str(FIG_PATH / 'nn1.png')

# Define encoder model specs
encoder_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


class CustomBertLayer(layers.Layer):
    ''' Custom BERT Layer supporting 4D input '''
    def __init__(self, model=encoder_model, name=None, trainable=True):
        super(CustomBertLayer, self).__init__(name=name)
        self.model = model
        self.trainable = trainable

    def call(self, input, low_mem=True):
        enc = tf.map_fn(lambda x: self.model({'input_ids': x[0], 
                                            'attention_mask': x[1]})[0][:,0,:],
                        elems=tuple(input), 
                        fn_output_signature=tf.float32)
        return enc


class TransformerBlock(layers.Layer):
    ''' Transformer block with self-attention '''
    def __init__(self, embed_dim=768, ff_dim=32, name=None, rate=0.1):
        super(TransformerBlock, self).__init__(name=name)
        self.att = layers.Attention() # self-attention?
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att([inputs, inputs])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ClassificationBlock(layers.Layer):
    ''' Define classification block for 2d subreddits'''
    def __init__(self, relu_dim=32, n_classes=500, name=None):
        super(ClassificationBlock, self).__init__(name=name)
        self.ffn = keras.Sequential(
            [layers.Dense(relu_dim, activation="relu"), 
            layers.Dense(n_classes, activation="softmax")] )

    def call(self, inputs):
        return self.ffn(inputs)

