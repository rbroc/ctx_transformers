import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense,
                                     Concatenate,
                                     Add,
                                     MultiHeadAttention,
                                     LayerNormalization,
                                     Dropout)
from reddit.utils import (average_encodings, 
                          load_weights_from_huggingface)
from transformers.modeling_tf_utils import get_initializer
from transformers import DistilBertConfig
from transformers.models.distilbert.modeling_tf_distilbert import TFMultiHeadSelfAttention
import itertools


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
        pretrained_weights (str): path to pretrained model weights in 
            huggingface normal. Pass None to initialize from scratch
        trained_encoder_weights (str): path to encoder weights to load
        trained_encoder_class (str): model class for encoder
        name (str): identification string
        freeze_encoder (bool): which layers of the encoder 
            to freeze. Pass False or None for no freezing.
        reset_head (bool): whether to reinitialize the classification head
            after loading the weights.
        '''
    def __init__(self, 
                 transformer,
                 pretrained_weights,
                 trained_encoder_weights=None,
                 trained_encoder_class=None,
                 name=None,
                 n_layers=None,
                 freeze_encoder=False,
                 reset_head=False,
                 vocab_size=30522):
        
        # Set up id parameters        
        freeze_str = 'no' if not freeze_encoder else '_'.join(list(freeze_encoder))
        freeze_str = freeze_str + 'freeze' 
        reset_str = 'hreset' if reset_head else 'nhoreset'
        n_layers = n_layers or 6
        weight_str = pretrained_weights or trained_encoder_weights or 'scratch'
        weights_str = weight_str.replace('-', '_')
        
        if name is None:
            name = f'BatchTransformerForMLM-{freeze_str}-{reset_str}-'
                   f'{n_layers}layers-{weights_str}'
        super(BatchTransformerForMLM, self).__init__(name=name)
        
        # Initialize model
        if pretrained_weights is None:
            config = transformer.config_class(vocab_size=vocab_size, n_layers=n_layers)
            mlm_model = transformer(config)
        else:
            mlm_model = transformer.from_pretrained(pretrained_weights) # convert weights
        if trained_encoder_weights and trained_encoder_class:
            load_trained_encoder_weights(model=mlm_model, 
                                         transformers_model_class=trained_encoder_class,
                                         weights_path=trained_encoder_weights,
                                         layer=0)
        
        # Set up components
        self.encoder = mlm_model.layers[0]
        self.vocab_dense = mlm_model.layers[1]
        self.act = mlm_model.act
        self.vocab_layernorm = mlm_model.layers[2]
        self.vocab_projector = mlm_model.layers[-1]
        self.vocab_size = mlm_model.vocab_size
        
        # Freeze and reset stuff
        if not freeze_encoder:
            self.encoder.trainable = True
        else:
            for fl in freeze_encoder:
                self.encoder._layers[1]._layers[0][int(fl)]._trainable = False
            self.encoder._layers[0]._trainable = False # freeze embeddings
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
    ''' Model class for masked language modeling using context
        using standard transformers with aggregation
    Args:
        transformer: MLM model class from transformers library
        pretrained_weights (str): path to model initialization weights or 
            pretrained huggingface
        trained_encoder_weights (str): path to encoder weights to load
        trained_encoder_class (str): model class for encoder
        name (str): identification string
        n_layers (int): number of transformer layers for encoder (relevant 
            if not loading a pretrained configurations)
        freeze_encoder (list, False, or None): which layers of the encoder to 
            freeze
        reset_head (bool): whether to re-initialize the classification head
        add_dense (int): number of additional dense layers to add 
            between the encoder and the MLM head, after concatenating
            aggregate context and target post.
        dims (int): dimensionality of dense layers
        n_tokens (int): number of tokens in sequence
        aggregate (str): if aggregate is 'dense', if no additional dense layers 
            are specified after concatenation it adds a converter layer.
            If 'add', multiplies each dimension of the context by each
                dimension of the token representation. 
        n_contexts (int): number of contexts passed
    '''
    def __init__(self, 
                 transformer,
                 pretrained_weights,
                 trained_encoder_weights=None,
                 trained_encoder_class=None,
                 name=None,
                 n_layers=None,
                 freeze_encoder=False,
                 reset_head=False,
                 add_dense=0,
                 dims=768,
                 n_tokens=512,
                 aggregate='concatenate',
                 n_contexts=10,
                 vocab_size=30522):
        
        # Name parameters
        freeze_str = 'no' if not freeze_encoder else '_'.join(list(freeze_encoder))
        freeze_str = freeze_str + 'freeze' 
        reset_str = 'hreset' if reset_head else 'nhoreset'
        n_layers = n_layers or 6
        weight_str = pretrained_weights or trained_encoder_weights or 'scratch'
        weights_str = weight_str.replace('-', '_')
        
        # Check dense layers
        if add_dense == 0:
            dims_str = 'none'
        else:
            if isinstance(dims, int):
                dims = [dims] * add_dense       
            elif isinstance(dims, list):
                dims = [int(d) for d in dims]
                if len(dims) != add_dense:
                    raise ValueError('Length of dims must match add_dense')
            dims_str = '_'.join([str(d) for d in dims])
        
        if name is None:
            mtype = 'BatchTransformerForContextMLM'
            dense_args = f'dense{add_dense}-densedim{dims_str}'
            ctx_args = f'agg{aggregate}'
            name = f'{mtype}-{freeze_str}-{reset_str}-{n_layers}layers-{dense_args}-'
                   f'{weights_str}-{ctx_args}'
        super(BatchTransformerForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        self.aggregate = aggregate
        self.output_signature = tf.float32
        self.n_contexts = n_contexts
        
        # Initialize basic model components
        if pretrained_weights is None:
            config = transformer.config_class(vocab_size=vocab_size, 
                                              n_layers=n_layers)
            mlm_model = transformer(config)
        else:
            mlm_model = transformer.from_pretrained(pretrained_weights) # convert weights
        if trained_encoder_weights and trained_encoder_class:
            load_trained_encoder_weights(model=mlm_model, 
                                         transformers_model_class=trained_encoder_class,
                                         weights_path=trained_encoder_weights,
                                         layer=0)
            
            
        self.encoder = mlm_model.layers[0]
        self.vocab_dense = mlm_model.layers[1]
        self.act = mlm_model.act
        self.vocab_layernorm = mlm_model.layers[2]
        self.vocab_projector = mlm_model.layers[-1]
        self.vocab_size = mlm_model.vocab_size
        
        # Add intermediate layers
        self.context_normalizer = LayerNormalization(epsilon=1e-12)
        if self.aggregate == 'concatenate': 
            self.agg_layer = Concatenate(axis=-1)
        else:
            self.agg_layer = Add()
        self.aggregate_dense = keras.Sequential([Dense(units=dims[i], 
                                                           activation='linear')
                                                     for i in range(add_dense)] + 
                                                     [Dense(units=768, activation='relu')])
        self.aggregate_normalizer = LayerNormalization(epsilon=1e-12)
        
        # Freeze and reset stuff
        if not freeze_encoder:
            self.encoder.trainable = True
        else:
            for fl in freeze_encoder:
                self.encoder._layers[1]._layers[0][int(fl)]._trainable = False
            self.encoder._layers[0]._trainable = False # freeze embeddings
        if reset_head:
            initializer = get_initializer()
            for layer in [self.vocab_dense, 
                          self.vocab_layernorm, 
                          self.vocab_projector]:
                layer.set_weights([initializer(w.shape) 
                                   for w in layer.weights])
   

    def _encode_batch(self, example):
        out = self.encoder(input_ids=example['input_ids'], 
                            attention_mask=example['attention_mask'])
        return out.last_hidden_state

    
    def _pool_context(self, hidden_state):
        ctx = tf.reduce_mean(hidden_state[:,1:,0,:],
                             axis=1, keepdims=True)
        ctx = self.context_normalizer(ctx)
        ctx = tf.expand_dims(ctx, axis=2)
        ctx = tf.repeat(ctx, self.n_contexts+1, axis=1)
        ctx = tf.repeat(ctx, self.n_tokens, axis=2)
        return ctx

    
    def call(self, input):
        hidden_state = tf.vectorized_map(self._encode_batch, 
                                             elems=input)
        ctx = self._pool_context(self, hidden_state)
        aggregated = self.agg_layer([hidden_state, ctx])
        aggregated = self.aggregate_dense(aggregated)
        aggregated = self.aggregate_normalizer(aggregated + hidden_state)
        targets = aggregated[:,0,:,:]
        targets = self.vocab_dense(targets)
        targets = self.act(targets)
        targets = self.vocab_layernorm(targets)
        logits = self.vocab_projector(out)
        return logits


class HierarchicalTransformerForContextMLM(keras.Model):
    ''' Base model for masked language modeling with contexts
        using a hierarchical transformer
    Args:
        transformer: MLM model class from transformers library
        name (str): identification string
        n_layers (int): number of transformer layers for encoder (relevant 
            if not loading a pretrained configurations)
        add_dense (int): number of additional dense layers to add 
            between the encoder and the MLM head
        dims (int): dimensionality of dense layers
        n_tokens (int): number of tokens in sequence
        n_contexts (int): number of contexts passed
    '''
    def __init__(self, 
                 transformer,
                 name=None,
                 n_layers=3,
                 n_tokens=512,
                 n_contexts=10,
                 vocab_size=30522):
        
        
        if name is None:
            mtype = 'HierarchicalTransformerForContextMLM'
            name = f'{mtype}-{n_layers}'
        super(HierarchicalTransformerForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        self.output_signature = tf.float32
        
        # Create encoder
        config = transformer.config_class(vocab_size=vocab_size, n_layers=n_layers)
        mlm_model = transformer(config)
        self.encoder = mlm_model.layers[0]
        self.encoder.trainable = True
        self.ctx_transformer = [TFMultiHeadSelfAttention(config, name="attention") 
                                for _ in range(n_layers-1)]
        for ct in self.ctx_transformer:
            ct.trainable = True
        self.post_transformer_dense = [[Dense(units=dims[0], activation='relu'),
                                        LayerNormalization(epsilon=1e-12)]
                                       for _ in range(n_layers-1)]

        # Add head
        self.vocab_dense = mlm_model.layers[1]
        self.act = mlm_model.act
        self.vocab_layernorm = mlm_model.layers[2]
        self.vocab_projector = mlm_model.layers[-1]
        self.vocab_size = mlm_model.vocab_size
        
        # Reset head weight
        initializer = get_initializer()
        for layer in [self.vocab_dense, 
                      self.vocab_layernorm, 
                      self.vocab_projector]:
            layer.set_weights([initializer(w.shape) 
                               for w in layer.weights])
             
        
    def _encode_batch(self, example):
        hidden_state = self.encoder._layers[0](example['input_ids'])
        for i, layer_module in enumerate(self.encoder._layers[1].layer):
            layer_outputs = layer_module(hidden_state, 
                                         example['attention_mask'],
                                         None,
                                         False,
                                         training=True)
            hidden_state = layer_outputs[-1]
            if (i+1)!=len(self.encoder._layers[1].layer):
                cls_tokens = hidden_state[:,0,:] # get contexts
                cls_tokens = tf.expand_dims(cls_tokens, axis=0) # expand to fit expected dim
                cls_tokens = self.ctx_transformer[i](cls_tokens, cls_tokens, cls_tokens, # pass through attentions
                                                     tf.constant(1, shape=[1,self.n_contexts+1]), 
                                                     head_mask=None, 
                                                     output_attentions=False, 
                                                     training=True)[0][0,:,:] 
                cls_tokens = tf.expand_dims(cls_tokens, axis=1) # fit to shape
                cls_tokens = tf.pad(cls_tokens, [[0,0], 
                                                 [0,self.n_tokens-1], 
                                                 [0,0]])
                merged = self.post_transformer_dense[i][0](cls_tokens+hidden_state) # propagate by adding to CLS
                hidden_state = self.post_transformer_dense[i][1](merged+hidden_state) # layernorm
                return hidden_state
    
    
    def call(self, input):
        outs = tf.vectorized_map(self._encode_batch, 
                                 elems=input)
        out = outs[:,0,:,:]
        out = self.vocab_dense(out)
        out = self.act(out)
        out = self.vocab_layernorm(out)
        logits = self.vocab_projector(out)
        return logits
    

class BiencoderForContextMLM(keras.Model):
    ''' Model class for masked language modeling using context
        using standard transformers with aggregation
    Args:
        transformer: MLM model class from transformers library
        pretrained_weights (str): path to model initialization weights or 
            pretrained huggingface
        trained_encoder_weights (str): path to encoder weights to load
        trained_encoder_class (str): model class for encoder
        name (str): identification string
        n_layers (int): number of transformer layers for encoder (relevant 
            if not loading a pretrained configurations)
        freeze_encoder (list, False, or None): which layers of the encoder to 
            freeze
        reset_head (bool): whether to re-initialize the classification head
        add_dense (int): number of additional dense layers to add 
            between the encoder and the MLM head, after concatenating
            aggregate context and target post.
        dims (int): dimensionality of dense layers
        n_tokens (int): number of tokens in sequence
        aggregate (str): if aggregate is 'dense', if no additional dense layers 
            are specified after concatenation it adds a converter layer.
            If 'add', multiplies each dimension of the context by each
                dimension of the token representation. 
        n_contexts (int): number of contexts passed
    '''
    def __init__(self, 
                 transformer,
                 pretrained_token_encoder_weights,
                 trained_token_encoder_weights=None,
                 trained_token_encoder_class=None,
                 name=None,
                 n_layers_token_encoder=3,
                 n_layers_context_encoder=3,
                 freeze_token_encoder=False,
                 add_dense=0,
                 dims=768,
                 n_tokens=512,
                 aggregate='concatenate',
                 n_contexts=10,
                 vocab_size=30522):
        
        # Name parameters
        freeze_str = 'no' if not freeze_encoder else '_'.join(list(freeze_encoder))
        freeze_str = freeze_str + 'freeze' 
        n_layers_token_encoder = n_layers_token_encoder or 6
        n_layers_str = f'{n_layers_token_encoder}_{n_layers_context_encoder}'
        weight_str = pretrained_token_encoder_weights or \
                     trained_token_encoder_weights or \
                     'scratch'
        weights_str = weight_str.replace('-', '_')
        
        # Check dense layers
        if add_dense == 0:
            dims_str = 'none'
        else:
            if isinstance(dims, int):
                dims = [dims] * add_dense       
            elif isinstance(dims, list):
                dims = [int(d) for d in dims]
                if len(dims) != add_dense:
                    raise ValueError('Length of dims must match add_dense')
            dims_str = '_'.join([str(d) for d in dims])
        
        if name is None:
            mtype = 'BiencoderForContextMLM'
            dense_args = f'dense{add_dense}-densedim{dims_str}'
            ctx_args = f'agg{aggregate}'
            name = f'{mtype}-{freeze_str}-{n_layers_str}layers-{dense_args}-'
                   f'{weights_str}-{ctx_args}'
        super(BiencoderForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        self.aggregate = aggregate
        self.output_signature = tf.float32
        self.n_contexts = n_contexts
        
        # Initialize basic model components
        if pretrained_weights is None:
            config_token_encoder = transformer.config_class(vocab_size=vocab_size, 
                                                            n_layers=n_layers_token_encoder)
            mlm_model = transformer(config)
        else:
            mlm_model = transformer.from_pretrained(pretrained_weights) # convert weights
        if trained_encoder_weights and trained_encoder_class:
            load_trained_encoder_weights(model=mlm_model, 
                                         transformers_model_class=trained_encoder_class,
                                         weights_path=trained_encoder_weights,
                                         layer=0)
        config_ctx_encoder = transformer.config_class(vocab_size=vocab_size,
                                                      n_layers=n_layers_context_encoder)
        self.token_encoder = mlm_model.layers[0]
        self.vocab_dense = mlm_model.layers[1]
        self.act = mlm_model.act
        self.vocab_layernorm = mlm_model.layers[2]
        self.vocab_projector = mlm_model.layers[-1]
        self.vocab_size = mlm_model.vocab_size        
        self.context_encoder = transformer(config_ctx_encoder).layers[0]

        # Define aggregation module
        if self.aggregate != 'attention':
            # There could be dense here?
            # Could add some dropouts
            self.context_normalizer = LayerNormalization(epsilon=1e-12)
            if self.aggregate = 'concatenate':
                self.agg_layer = Concatenate(axis=-1)
            else:
                self.agg_layer = Add()
            self.aggregate_dense = keras.Sequential([Dense(units=dims[i], 
                                                           activation='linear')
                                                     for i in range(add_dense)] + 
                                                    [Dense(units=768, 
                                                           activation='relu')])
            self.aggregate_normalizer = LayerNormalization(epsilon=1e-12)
        else:
            self.aggregate_attention = MultiHeadAttention(num_heads=6,
                                                          key_dim=768)
            # Anything here?
            

        # Freeze and reset stuff
        if not freeze_token_encoder:
            self.encoder.trainable = True
        else:
            for fl in freeze_token_encoder:
                self.token_encoder._layers[1]._layers[0][int(fl)]._trainable = False
            self.token_encoder._layers[0]._trainable = False # freeze embeddings too
        if reset_head:
            initializer = get_initializer()
            for layer in [self.vocab_dense, 
                          self.vocab_layernorm, 
                          self.vocab_projector]:
                layer.set_weights([initializer(w.shape) 
                                   for w in layer.weights])


    def _encode_batch(self, example):
        ctx = self.context_encoder(input_ids=example['input_ids'][:,1:,:],
                                          attention_mask=example['attention_mask'][:,1:,:])
        return ctx.last_hidden_state

    def call(self, input):
        target = self.encoder(input_ids=input['input_ids'][:,0,:], 
                              attention_mask=input['attention_mask'][:,0,:]).last_hidden_state
        contexts = tf.vectorized_map(self._encode_batch, elems=input)
        contexts = hidden_states[:,:,0,:] # should this be averaged?
        out = self._aggregate(target, contexts)
        # out = self.added_dense_layer(out)
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
        encodings = tf.vectorized_map(self._encode_batch, elems=input)
        out = tf.reduce_mean(encodings, axis=1)
        if self.dense_layers:
            out = self.dense_layers(out)
        out = self.head(out)
        return out

        