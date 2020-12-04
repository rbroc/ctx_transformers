from tensorflow import keras
from tensorflow.keras import layers, Input
from transformers import TFDistilBertModel
from custom_bert import CustomBertLayer
from pathlib import Path

# Define path variable
FIG_PATH = Path('figures')
FIG_FILE = str(FIG_PATH / 'nn1.png')

# Define encoder model specs
encoder_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


class TransformerBlock(layers.Layer):
    ''' Transformer block with self-attention '''
    def __init__(self, embed_dim, ff_dim, name=None, rate=0.1):
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
    def __init__(self, relu_dim, n_classes, name=None):
        super(ClassificationBlock, self).__init__(name=name)
        self.ffn = keras.Sequential(
            [layers.Dense(relu_dim, activation="relu"), 
            layers.Dense(n_classes, activation="softmax")] )

    def call(self, inputs):
        return self.ffn(inputs)

class Model(keras.Model):
    ''' Define full model '''
    def __init__(self, trainable=False, n_classes=500, 
                embed_dim=768, transformer_ff_dim=32, 
                pre_classifier_dim=32):
        super(Model1, self).__init__()
        self.encoder = CustomBertLayer(encoder_model, 
                                       name='encoder', 
                                       trainable=trainable)
        self.transformer = TransformerBlock(embed_dim=embed_dim, 
                                            ff_dim=transformer_ff_dim, 
                                            name='transformer')
        self.classifier = ClassificationBlock(relu_dim=pre_classifier_dim, 
                                              n_classes=n_classes, 
                                              name='one_hot_subreddit')
    
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.transformer(x)
        return self.classifier(x)
