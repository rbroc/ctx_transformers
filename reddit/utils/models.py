import tensorflow as tf
from transformers import TFDistilBertModel
from pathlib import Path


### COULD POSSIBLY REMOVE THIS
# Probably the most efficient way to do this would be to:
# Save the model in tensorflow
# Load 

def convert_weights_for_huggingface(ckpt_path,
                                    model=None,
                                    reddit_model_class=None,
                                    transformers_model_class=None, 
                                    transformer_weights=None,
                                    outpath=None):
    ''' Saves weights in format compatible with huggingface 
        transformers' from_pretrained method
    Args:
        ckpt_path: path to checkpoint
        model: initialized reddit model
        reddit_model_class: if model is not defined, pass Reddit model 
            class (e.g., BatchTransformer) here
        transformers_model_class: if model is not defined, pass huggingface's
            transformers class here (e.g., TFDistilBertModel)
        transformer_weights: if model is not defined,
            pass pretrained weights for transformers model here
    '''
    if outpath is None: 
        outpath = Path(ckpt_path) / '..' / '..'/ '..' / '..' / 'huggingface'
    outpath.mkdir(exist_ok=True, parents=True)
    if model is None:
        model = reddit_model_class(transformers_model_class,
                                   transformers_weights)
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    model.load_weights(ckpt)
    model.encoder.save_pretrained(outpath) # only saves encoder weights
    return 'Model saved!'


def load_weights_from_huggingface(model,
                                  transformers_model_class,
                                  weights_path,
                                  layer=0):
    ''' Load transformer weights from another huggingface model (for 
        entire model or one layer only). There may be more elegant 
        way to do this, but this creates a second model on the fly and
        gets the weights, then transfers them to the target model
    Args:
        transformers_model: transformer model (destination for weights import)
        transformers_model_class: transformer model to import weights from (source 
            for weights import)
        weights_path: path to read the weights from (or name of pretrained model)
        layer (optional): specifies which layer to import weights for, in case 
            only one layer's weights need to be loaded/updated
    '''
    if layer is not None:
        model.layers[layer]\
             .set_weights(transformers_model_class.from_pretrained(weights_path)\
                                                  .get_weights())
    else:
        model.set_weights(transformers_model_class.from_pretrained(weights_path)\
                                                  .get_weights())