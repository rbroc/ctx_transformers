import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense,
                                     Concatenate,
                                     Multiply, 
                                     MultiHeadAttention,
                                     Lambda, Dot, 
                                     LayerNormalization,
                                     Dropout)
from reddit.utils import (average_encodings, 
                          load_weights_from_huggingface)
from reddit import MLMContextMerger
from transformers.modeling_tf_utils import get_initializer
from transformers import DistilBertConfig
import itertools
from reddit.src.distilbert import CTXTransformerBlock                                               


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
                 name=None, trainable=True, 
                 output_attentions=False):
        if name is None:
            name = f'BatchTransformer-{path_to_weights}'
        super(BatchTransformer, self).__init__(name=name)
        self.path_to_weights = path_to_weights
        self.encoder = transformer.from_pretrained(path_to_weights, 
                                                   output_attentions=output_attentions)
        self.trainable = trainable
        self.output_signature = tf.float32
        self.output_attentions = output_attentions

        
    def _encode_batch(self, example):
        mask = tf.reduce_all(tf.equal(example['input_ids'], 0), 
                             axis=-1, keepdims=True)
        mask = tf.cast(mask, tf.float32)
        mask = tf.abs(tf.subtract(mask, 1.))
        output = self.encoder(
                              input_ids=example['input_ids'],
                              attention_mask=example['attention_mask']
                              )
        encoding = output.last_hidden_state[:,0,:]
        attentions = output.attentions if self.output_attentions else None
        masked_encoding = tf.multiply(encoding, mask)
        return masked_encoding, attentions

    
    def call(self, input):
        encodings, attentions = tf.vectorized_map(self._encode_batch, 
                                                  elems=input)
        if self.output_attentions:
            return encodings, attentions
        else:
            return encodings


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
        kwargs: kwargs for layers.Dense call
    '''
    def __init__(self,
                 transformer, path_to_weights, 
                 n_dense=3,
                 dims=768,
                 activations='relu',
                 trainable=False,
                 name=None, 
                 **kwargs):

        if isinstance(dims,list):
            if len(dims) != n_dense:
                raise ValueError('length of dims does '
                                 'match number of layers')
        elif isinstance(dims, int):
            dims = [dims] * n_dense
        self.dims = dims
        if isinstance(activations,list):
            if len(activations) != n_dense:
                raise ValueError('length of activations does '
                                 'match number of layers')
        if not isinstance(activations, list):
            activations = [activations] * n_dense
        self.activations = activations
        if name is None:
            name = f'''BatchTransformerFFN-{path_to_weights}_
                       layers-{n_dense}_
                       dim-{'_'.join([str(d) for d in dims])}_
                       {'_'.join(activations)}'''
        super().__init__(transformer, path_to_weights, name, 
                         trainable)
        self.dense_layers = keras.Sequential([Dense(dims[i], activations[i], **kwargs)
                                              for i in range(n_dense)])
        self.average_layer = Lambda(average_encodings)
        self.concat_layer = Concatenate(axis=1)
        
    def call(self, input):
        encodings, _ = super().call(input)
        avg_anchor = self.average_layer(encodings)
        avg_pos = self.average_layer(encodings)
        avg_neg = self.average_layer(encodings)
        encodings = self.concat_layer([avg_neg, avg_pos, avg_anchor])
        encodings = self.dense_layers(encodings)
        return encodings
    

class BatchTransformerForMLM(keras.Model):
    ''' Base model for masked language modeling (no context)
    Args:
        transformer: MLM model class from transformers library
        init_weights (str): path to model initialization weights
        name (str): identification string
        load_encoder_weights (str): path to encoder weights to load
        load_encoder_model_class (str): model class for encoder
        freeze_encoder (bool): whether to freeze the whole encoder
        freeze_encoder_layers (iterator): which layers of the encoder 
            to freeze. Relevant only if freeze_encoder is False.
            Indexing starts from zero.
        freeze_head (bool): whether to freeze the classification head
        reset_head (bool): whether to reinitialize the classification head
        '''
    def __init__(self, 
                 transformer,
                 init_weights,
                 name=None,
                 load_encoder_weights=None,
                 load_encoder_model_class=None,
                 freeze_encoder=True,
                 freeze_encoder_layers=None,
                 freeze_head=False,
                 reset_head=False,
                 from_scratch=False):
        
        # Name parameters
        if freeze_encoder:
            freeze_str = 'all'
        else:
            if freeze_encoder_layers:
                freeze_str = '_'.join(list(freeze_encoder_layers))
            else:
                freeze_str = 'none'
        if load_encoder_weights is None:
            load_str = 'pretrained'
        else:
            load_str = 'trained'
        if freeze_head:
            fhead_str = 'hfreeze'
        else:
            fhead_str = 'nohfreeze'
        if reset_head:
            reset_str = 'reset'
        else:
            reset_str = 'noreset'        
        if name is None:
            name = f'BatchTransformerForMLM-{freeze_str}-{load_str}-{reset_str}-{fhead_str}'
        super(BatchTransformerForMLM, self).__init__(name=name)
        
        # Initialize model
        if from_scrach:
            config = DistilBertConfig(vocab_size=30522, n_layers=3)
            mlm_model = transformer(config)
        else:
            mlm_model = transformer.from_pretrained(init_weights)
        self.encoder = mlm_model.layers[0]
        self.vocab_dense = mlm_model.layers[1] # this could be replaced
        self.act = mlm_model.act
        self.vocab_layernorm = mlm_model.layers[2]
        self.vocab_projector = mlm_model.layers[-1]
        self.vocab_size = mlm_model.vocab_size
        
        if load_encoder_weights and load_encoder_model_class:
            load_weights_from_huggingface(model=mlm_model, 
                                          transformers_model_class=load_encoder_model_class,
                                          weights_path=load_encoder_weights,
                                          layer=0)
        if freeze_encoder:
            self.encoder.trainable = False
        else:
            if freeze_encoder_layers:
                for fl in freeze_encoder_layers:
                    self.encoder._layers[1]._layers[0][int(fl)]._trainable = False
                self.encoder._layers[0]._trainable = False # freeze embedding layer
        if freeze_head:
            self.vocab_dense.trainable = False
            self.vocab_layernorm.trainable = False
            self.vocab_projector.trainable = False
        if reset_head:
            initializer = get_initializer()
            for layer in [self.vocab_dense, 
                          self.vocab_layernorm, 
                          self.vocab_projector]:
                layer.set_weights([initializer(w.shape) 
                                   for w in layer.weights])
        self.output_signature = tf.float32
        
    
    def _encode_batch(self, example):
        output = self.encoder(input_ids=example['input_ids'],
                              attention_mask=example['attention_mask']).last_hidden_state
        output = self.vocab_dense(output)
        output = self.act(output)
        output = self.vocab_layernorm(output)
        output = self.vocab_projector(output)
        return output

    def call(self, input):
        logits = tf.vectorized_map(self._encode_batch, elems=input)
        return logits
    

class BatchTransformerForContextMLM(keras.Model):
    ''' Base model for masked language modeling (no context)
    Args:
        transformer: MLM model class from transformers library
        init_weights (str): path to model initialization weights
        name (str): identification string
        load_weights (str): path to encoder weights to load
        load_encoder_model_class (str): model class for encoder
        freeze_encoder (bool): whether the whole encoder should be frozen
        freeze_encoder_layers (iterator): which layers of the encoder 
            to freeze (indexing starts from zero).
        freeze_head (bool): whether to freeze the classification head
        reset_head (bool): whether to re-initialize the classification head
        add_dense (int): number of additional dense layers to add 
            between the encoder and the MLM head, after concatenating
            aggregate context and target post.
        dims (int): dimensionality of dense layers
        n_tokens (int): number of tokens in sequence
        context_pooling (str): if 'cls', averages cls tokens to get context
            representations. If 'mean' or 'max' pools token-level 
            representations by averaging or taking the max.
        aggregate (str): if aggregate is 'dense', if no additional dense layers 
            are specified after concatenation it adds a converter layer.
            If 'multiply', multiplies each dimension of the context by each
                dimension of the token representation. 
                If 'attention', applies attention head to aggregated 
                context and token representation.
    '''
    def __init__(self, 
                 transformer, 
                 init_weights,
                 name=None, 
                 load_encoder_weights=None,
                 load_encoder_model_class=None,
                 freeze_head=False,
                 reset_head=False,
                 freeze_encoder=True,
                 freeze_encoder_layers=None,
                 add_dense=0,
                 dims=768,
                 n_tokens=512,
                 context_pooling='cls',
                 aggregate='concatenate',
                 batch_size=1,
                 from_scratch=False, 
                 hierarchical=False):
        
        # Name parameters
        if freeze_encoder:
            freeze_str = 'all'
        else:
            if freeze_encoder_layers:
                freeze_str = '_'.join(list(freeze_encoder_layers))
            else:
                freeze_str = 'none'
        if load_encoder_weights is None:
            load_str = 'pretrained'
        else:
            load_str = 'trained'
        if reset_head:
            reset_str = 'reset'
        else:
            reset_str = 'noreset'
        if freeze_head:
            fhead_str = 'hfreeze'
        else:
            fhead_str = 'nohfreeze'    
        # Handle dense layers creation
        if add_dense is None or add_dense == 0:
            add_dense = 0
            dims_str = 'none'
        else:
            if isinstance(dims, int):
                if dims == 768:
                    dims = [dims] * add_dense
                else:
                    raise ValueError('Dense layers must have 768 nodes')         
            elif isinstance(dims, list):
                dims = [int(d) for d in dims]
                if dims[-1] != 768:
                    raise ValueError('Dense layers must have 768 nodes')
                if len(dims) != add_dense:
                    raise ValueError('Length of dims must match add_dense')
            dims_str = '_'.join([str(d) for d in dims])
        
        if name is None:
            mtype = 'BatchTransformerForContextMLM'
            dense_args = f'{add_dense}-{dims_str}'
            ctx_args = f'{context_pooling}-{aggregate}'
            name = f'{mtype}-{freeze_str}-{load_str}-{dense_args}-{ctx_args}-{reset_str}-{fhead_str}'
        super(BatchTransformerForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        if context_pooling not in ['cls', 'mean', 'max']:
            raise ValueError('context_pooling must be cls, mean or max')
        self.context_pooling = context_pooling
        self.aggregate = aggregate
        
        # Create encoder
        config = DistilBertConfig(vocab_size=30522, n_layers=3)
        if from_scratch:
            mlm_model = transformer(config)
        else:
            mlm_model = transformer.from_pretrained(init_weights)
        self.encoder = mlm_model.layers[0]
        
        # Create aggregator
        if self.aggregate == 'concatenate': 
            self.agg_layer = Concatenate(axis=-1)
        elif self.aggregate == 'attention':
            self.agg_layer = MultiHeadAttention(num_heads=6, 
                                                key_dim=768)
            self.att_norm = LayerNormalization()
        
        # Create dense
        if add_dense is not None and add_dense > 0:
            self.dense_layers = [LayerNormalization(epsilon=1e-12),
                                 Dense(units=dims[0], activation='relu'),
                                 LayerNormalization(epsilon=1e-12)]
        else:
            self.dense_layers = None
        
        # Add head
        self.vocab_dense = mlm_model.layers[1]
        self.act = mlm_model.act
        self.vocab_layernorm = mlm_model.layers[2]
        self.vocab_projector = mlm_model.layers[-1]
        self.vocab_size = mlm_model.vocab_size
        self.hierarchical = hierarchical
        
        # Freeze
        if load_encoder_weights and load_encoder_model_class:
            load_weights_from_huggingface(model=mlm_model, 
                                          transformers_model_class=load_encoder_model_class,
                                          weights_path=load_encoder_weights,
                                          layer=0)
        if freeze_encoder:
            self.encoder.trainable = False
        else:
            if freeze_encoder_layers:
                for fl in freeze_encoder_layers:
                    self.encoder._layers[1]._layers[0][int(fl)]._trainable = False
                self.encoder._layers[0]._trainable = False
        if freeze_head:
            self.vocab_dense.trainable = False
            self.vocab_layernorm.trainable = False
            self.vocab_projector.trainable = False
        if reset_head:
            initializer = get_initializer()
            for layer in [self.vocab_dense, 
                          self.vocab_layernorm, 
                          self.vocab_projector]:
                layer.set_weights([initializer(w.shape) 
                                   for w in layer.weights])
        self.output_signature = tf.float32
        self.batch_size = batch_size
   
    def _encode_batch(self, example):
        out = self.encoder(input_ids=example['input_ids'], 
                            attention_mask=example['attention_mask'])
        return out.last_hidden_state
    
    def _encode_batch_hierarchical(self, example):
        
        
    
    def call(self, input):
        hidden_state = tf.vectorized_map(self._encode_batch, 
                                         elems=input)
        ctx = tf.reduce_mean(hidden_state[:,1:,0,:], axis=1, keepdims=True) # 1 x 1 x 768
        ctx = self.dense_layers[0](ctx)
        ctx = tf.expand_dims(ctx, axis=2) # 1 x 1 x 1 x 768 
        ctx = tf.repeat(ctx, 11, axis=1) # 1 x 11 x 1 x 768
        ctx = tf.repeat(ctx, 512, axis=2) # 1 x 11 x 512 x 768
        outs = hidden_state + ctx
        outs = self.dense_layers[1](outs)
        outs = self.dense_layers[2](outs + hidden_state)
        out = outs[:,0,:,:]
        out = self.vocab_dense(out)
        out = self.act(out)
        out = self.vocab_layernorm(out)
        logits = self.vocab_projector(out)
        return logits
    

class BatchTransformerForAggregates(keras.Model):
    ''' Encodes and averages posts, predict aggregate 
    Args:
        transformer (transformers.Model): huggingface transformer
        weights (str): path to pretrained or init weights
        name (str): name
        target (str): name of target metric within agg dataset
        add_dense (int): number of dense layers to add before classification
        dims (list): number of nodes per layer
        encoder_strainable (bool): if the encoder is trainable
    '''
    def __init__(self, 
                 transformer, 
                 weights=None,
                 name=None, 
                 target='avg_posts',
                 add_dense=0,
                 dims=[10],
                 encoder_trainable=False):
        dims_str = '_'.join(dims)
        if len(dims) != add_dense:
            raise ValueError('dims should have add_dense values')
        if name is None:
            name = f'BatchTransformerForAggregates-{add_dense}-{dims_str}'
        super(BatchTransformerForAggregates, self).__init__(name=name)
        self.trainable = True
        self.encoder = transformer.from_pretrained(weights)
        self.encoder.trainable = encoder_trainable
        self.target = target
        self.output_signature = tf.float32
        if add_dense > 0:
            self.dense_layers = keras.Sequential([Dense(dims[idx])
                                                  for idx in range(add_dense)])
        else:
            self.dense_layers = None
        self.head = Dense(1, activation='linear')

    def _encode_batch(self, example):
        
        output = self.encoder(input_ids=example['input_ids'],
                              attention_mask=example['attention_mask'])
        encoding = output.last_hidden_state[:,0,:]
        return encoding


    def call(self, input):
        if self.hierarchical is False:
            encodings = tf.vectorized_map(self._encode_batch, elems=input)
            out = tf.reduce_mean(encodings, axis=1)
            if self.dense_layers:
                out = self.dense_layers(out)
            out = self.head(out)
            
            
            
            return out
        else:
            

        