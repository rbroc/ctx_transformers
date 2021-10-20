import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense,
                                     Concatenate, 
                                     Lambda)
from reddit.utils import (average_encodings,
                          dense_to_str, 
                          freeze_encoder_weights,
                          make_mlm_model_from_params)
from reddit.layers import (BatchTransformerContextAggregator,
                           BiencoderSimpleAggregator,
                           BiencoderAttentionAggregator,
                           HierarchicalContextAttention,
                           BatchTransformerContextPooler,
                           BiencoderContextPooler,
                           MLMHead)


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
            name = f'BatchTransformerForMLM-{freeze_str}-{reset_str}-{n_layers}layers-{weights_str}'
        super(BatchTransformerForMLM, self).__init__(name=name)
        
        # Initialize model
        mlm_model = make_mlm_model_from_params(transformer,
                                               pretrained_weights,
                                               vocab_size,
                                               n_layers,
                                               trained_encoder_weights,
                                               trained_encoder_class)
        self.encoder = mlm_model.layers[0]
        self.mlm_head = MLMHead(mlm_model, reset=reset_head)
        freeze_encoder_weights(self.encoder, freeze_encoder)
        self.output_signature = tf.float32
        
    
    def _encode_batch(self, example):
        output = self.encoder(input_ids=example['input_ids'],
                              attention_mask=example['attention_mask']).last_hidden_state
        output = self.mlm_head(output)
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
        vocab_size (int): vocabulary size for the model
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
        
        # Name
        freeze_str = 'no' if not freeze_encoder else '_'.join(list(freeze_encoder))
        freeze_str = freeze_str + 'freeze' 
        reset_str = 'hreset' if reset_head else 'nhoreset'
        n_layers = n_layers or 6
        weight_str = pretrained_weights or trained_encoder_weights or 'scratch'
        weights_str = weight_str.replace('-', '_')
        dims_str = dense_to_str(add_dense, dims)
        if name is None:
            mtype = 'BatchTransformerForContextMLM'
            dense_args = f'dense{add_dense}-densedim{dims_str}'
            ctx_args = f'agg{aggregate}'
            name = f'{mtype}-{freeze_str}-{reset_str}-{n_layers}layers' \
                   f'-{dense_args}-{weights_str}-{ctx_args}'
        super(BatchTransformerForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        self.aggregate = aggregate
        self.output_signature = tf.float32
        self.n_contexts = n_contexts
        
        # Define model components
        mlm_model = make_mlm_model_from_params(transformer,
                                               pretrained_weights,
                                               vocab_size,
                                               n_layers,
                                               trained_encoder_weights,
                                               trained_encoder_class)
        self.encoder = mlm_model.layers[0]
        freeze_encoder_weights(self.encoder, freeze_encoder)
        self.mlm_head = MLMHead(mlm_model, reset=reset_head)
        self.context_pooler = BatchTransformerContextPooler()
        self.aggregator = BatchTransformerContextAggregator(agg_fn=self.aggregate, 
                                                            add_dense=add_dense,
                                                            dims=dims)
   

    def _encode_batch(self, example):
        out = self.encoder(input_ids=example['input_ids'], 
                            attention_mask=example['attention_mask'])
        return out.last_hidden_state
    
    def call(self, input):
        hidden_state = tf.vectorized_map(self._encode_batch, 
                                         elems=input)
        ctx = self.context_pooler(hidden_state, 
                                  self.n_contexts, 
                                  self.n_tokens)
        aggd = self.aggregator(hidden_state, ctx)
        logits = self.mlm_head(aggd[:,0,:,:])
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
        vocab_size (int): vocabulary size for the model
    '''
    def __init__(self, 
                 transformer,
                 name=None,
                 n_layers=3,
                 n_tokens=512,
                 n_contexts=10,
                 vocab_size=30522):
        
        # Name
        if name is None:
            mtype = 'HierarchicalTransformerForContextMLM'
            name = f'{mtype}-{n_layers}'
        super(HierarchicalTransformerForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        self.n_contexts = n_contexts
        self.output_signature = tf.float32
        
        # Define components
        config = transformer.config_class(vocab_size=vocab_size, n_layers=n_layers)
        mlm_model = transformer(config)
        self.encoder = mlm_model.layers[0]
        self.hier_attentions = [HierarchicalContextAttention(self.n_contexts, 
                                                             self.n_tokens)
                                for _ in range(n_layers-1)]
        self.mlm_head = MLMHead(mlm_model, reset=True)
        
        
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
                hidden_state = self.hier_attentions[i](hidden_state)
            return hidden_state
    
    
    def call(self, input):
        outs = tf.vectorized_map(self._encode_batch, 
                                 elems=input)
        logits = self.mlm_model(outs[:,0,:,:])
        return logits
    

class BiencoderForContextMLM(keras.Model):
    ''' Model class for masked language modeling using 
        a biencoder architecture
    Args:
        transformer: MLM model class from transformers library
        pretrained_token_encoder_weights (str): path to model initialization 
            weights or pretrained huggingface for token encoder
        trained_token_encoder_weights (str): path to token encoder weights to load
        trained_token_encoder_class (str): model class for token encoder
        name (str): identification string
        n_layers_token_encoder (int): number of transformer layers for token encoder
        n_layers_context_encoder (int): number of transformer layers for context encoder
        freeze_token_encoder (list, False, or None): which layers of the token 
            encoder to freeze
        add_dense (int): number of additional dense layers to add 
            between the encoder and the MLM head, after aggregating context and target
        dims (int): dimensionality of dense layers
        n_tokens (int): number of tokens in sequence
        aggregate (str): if aggregate is 'dense', if no additional dense layers 
            are specified after concatenation it adds a converter layer.
            If 'add', multiplies each dimension of the context by each
                dimension of the token representation. If 'attention', applies
                attention between context and target
        n_contexts (int): number of contexts passed
        vocab_size (int): vocabulary size for the model
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
                 aggregate='concat',
                 n_contexts=10,
                 vocab_size=30522):
        
        # Name parameters
        freeze_str = 'no' if not freeze_token_encoder else '_'.join(list(freeze_token_encoder))
        freeze_str = freeze_str + 'freeze' 
        n_layers_token_encoder = n_layers_token_encoder or 6
        n_layers_str = f'{n_layers_token_encoder}_{n_layers_context_encoder}'
        weight_str = pretrained_token_encoder_weights or \
                     trained_token_encoder_weights or 'scratch'
        weights_str = weight_str.replace('-', '_')
        dims_str = dense_to_str(add_dense, dims)
        if name is None:
            mtype = 'BiencoderForContextMLM'
            dense_args = f'dense{add_dense}-densedim{dims_str}'
            ctx_args = f'agg{aggregate}'
            name = f'{mtype}-{freeze_str}-{n_layers_str}layers-{dense_args}' \
                   f'-{weights_str}-{ctx_args}'
        super(BiencoderForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        self.aggregate = aggregate
        self.output_signature = tf.float32
        self.n_contexts = n_contexts
        
        # Define components
        mlm_model = make_mlm_model_from_params(transformer,
                                               pretrained_token_encoder_weights,
                                               vocab_size,
                                               n_layers_token_encoder,
                                               trained_token_encoder_weights,
                                               trained_token_encoder_class)
        self.token_encoder = mlm_model.layers[0]
        freeze_encoder_weights(self.token_encoder, freeze_token_encoder)
        self.mlm_head = MLMHead(mlm_model, reset=True)
        config_ctx_encoder = transformer.config_class(vocab_size=vocab_size,
                                                      n_layers=n_layers_context_encoder)
        self.context_encoder = transformer(config_ctx_encoder).layers[0]
        
        # Define aggregation module
        if self.aggregate != 'attention':
            self.context_pooler = BiencoderContextPooler()
            self.aggregator = BiencoderSimpleAggregator(agg_fn=self.aggregate, 
                                                        add_dense=add_dense,
                                                        dims=dims)
        else:
            self.aggregator = BiencoderAttentionAggregator(include_head=True)

            
    def _encode_context(self, example):
        ctx = self.context_encoder(input_ids=example['input_ids'][:,1:,:],
                                   attention_mask=example['attention_mask'][:,1:,:])
        return ctx.last_hidden_state
    
            
    def call(self, input):
        target = self.encoder(input_ids=input['input_ids'][:,0,:], 
                              attention_mask=input['attention_mask'][:,0,:]).last_hidden_state
        contexts = tf.vectorized_map(self._encode_context, elems=input)[:,:,0,:]
        if self.aggregate != 'attention':
            contexts = self.context_pooler(contexts, self.n_tokens)
        target = self.aggregator(target, contexts)
        logits = self.mlm_model(target)
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

        