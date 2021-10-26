import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense,
                                     Concatenate, 
                                     Lambda)
from reddit.utils import (average_encodings,
                          dense_to_str, 
                          freeze_encoder_weights,
                          make_triplet_model_from_params,
                          make_mlm_model_from_params)
from reddit.layers import (BatchTransformerContextAggregator,
                           BiencoderSimpleAggregator,
                           BiencoderAttentionAggregator,
                           HierarchicalTransformerBlock,
                           ContextPooler,
                           BiencoderContextPooler,
                           MLMHead, 
                           SimpleCompressor, 
                           VAECompressor)


class BatchTransformer(keras.Model):
    ''' Transformer model wrapping HuggingFace transformer to
        support 3D (batch, n_sentences, n_tokens) inputs.
        Args:
            transformer (model): model object from huggingface
                transformers (e.g. TFDistilBertModel)
            pretrained_weights (str): path to pretrained weights
            
            name (str): model name.
            
            trainable (bool): whether to freeze weights
            output_attentions (bool): if attentions should be added
                to outputs (useful for diagnosing but not much more)
            compress_to (int): dimensionality for compression
            compress_mode (str): if compress_to is defined, can be
                'dense' for linear compression or 'vae' for auto-encoder
                compression
            intermediate_size (int): size of intermediate layers for 
                compression
            pooling (str): can be cls, mean, random. Random just pulls 
                random non-zero tokens.
    '''
    def __init__(self, transformer, 
                 pretrained_weights,
                 trained_encoder_weights=None,
                 trained_encoder_class=None,
                 name=None, trainable=True,
                 output_attentions=False, 
                 compress_to=None,
                 compress_mode=None,
                 intermediate_size=None,
                 pooling='cls',
                 vocab_size=30522,
                 n_layers=None, 
                 batch_size=1):
        if name is None:
            cto_str = str(compress_to) + '_' if compress_to else 'no'
            cmode_str = compress_mode or ''
            int_str = str(intermediate_size) + '_' if intermediate_size else 'no'
            weights_str = pretrained_weights or trained_encoder_weights or 'scratch'
            weights_str = weights_str.replace('-', '_')
            layers_str = n_layers or 6
            name = f'BatchTransformer-{layers_str}layers-{cto_str}{cmode_str}'
            name = name + f'-{int_str}int-{weights_str}'
        super(BatchTransformer, self).__init__(name=name)
        self.encoder = make_triplet_model_from_params(transformer,
                                                      pretrained_weights,  
                                                      vocab_size, 
                                                      n_layers,
                                                      trained_encoder_weights,
                                                      trained_encoder_class,
                                                      output_attentions)
        self.trainable = trainable
        self.output_signature = tf.float32
        self.output_attentions = output_attentions
        if compress_to:
            if compress_mode == 'dense':
                self.compressor = SimpleCompressor(compress_to, 
                                                   intermediate_size)
            elif compress_mode == 'vae':
                self.compressor = VAECompressor(compress_to, 
                                                intermediate_size, 
                                                batch_size=batch_size)
        else:
            self.compressor = None
        self.pooling = pooling

    def _encode_batch(self, example):
        mask = tf.reduce_all(tf.equal(example['input_ids'], 0), 
                             axis=-1, keepdims=True)
        mask = tf.cast(mask, tf.float32)
        mask = tf.abs(tf.subtract(mask, 1.))
        output = self.encoder(
                              input_ids=example['input_ids'],
                              attention_mask=example['attention_mask']
                              )
        if self.pooling == 'cls':
            encoding = output.last_hidden_state[:,0,:]
        elif self.pooling == 'mean':
            encoding = tf.reduce_sum(output.last_hidden_state[:,1:,:], axis=1)
            n_tokens = tf.reduce_sum(example['attention_mask'], axis=1, keepdims=1)
            encoding = encoding / tf.cast(n_tokens, tf.float32)
        elif self.pooling == 'random':
            n_nonzero = tf.reduce_sum(example['attention_mask'], axis=-1)
            idxs = tf.map_fn(lambda x: tf.random.uniform(shape=[], minval=1,
                             maxval=x, dtype=tf.int32), n_nonzero)
            encoding = tf.gather(output.last_hidden_state, idxs, 
                                  axis=1, batch_dims=1)
        attentions = output.attentions if self.output_attentions else None
        masked_encoding = tf.multiply(encoding, mask)
        return masked_encoding, attentions

    
    def call(self, input):
        encodings, attentions = tf.vectorized_map(self._encode_batch, 
                                                  elems=input)
        if self.compressor:
            encodings = self.compressor(encodings)
        if self.output_attentions:
            return encodings, attentions
        else:
            return encodings


class BatchTransformerFFN(BatchTransformer):
    ''' Batch transformer with added dense layers
    Args:
        transformer (model): model object from huggingface
            transformers (e.g. TFDistilBertModel) for batch
            transformer
        pretrained_weights (str): path to pretrained weights
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
                 transformer, 
                 pretrained_weights,
                 trained_encoder_weights=None,
                 trained_encoder_class=None,
                 n_dense=1,
                 dims=[768],
                 activations=['relu'],
                 trainable=False,
                 name=None,
                 n_layers=None,
                 vocab_size=30522):

        if len(dims) != n_dense:
            raise ValueError('length of dims does '
                                'match number of layers')
        if len(activations) != n_dense:
                raise ValueError('length of activations does '
                                 'match number of layers')           
        self.dims = dims
        self.activations = activations
        if name is None:
            weights_str = pretrained_weights or trained_encoder_weights or 'scratch'
            weights_str = weights_str.replace('-', '_')
            layers_str = n_layers or 6
            name = f'''BatchTransformerFFN-
                       {layers_str}layers-{n_dense}_
                       dim-{'_'.join([str(d) for d in dims])}_
                       {'_'.join(activations)}-{weights_str}'''
        super().__init__(transformer, pretrained_weights, 
                         trained_encoder_weights, trained_encoder_class,
                         name, vocab_size, n_layers, trainable)
        self.dense_layers = keras.Sequential([Dense(dims[i], activations[i])
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
                 add_dense=None,
                 dims=None,
                 activations=None,
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
            dense_args = f'{add_dense}dense-{dims_str}'
            ctx_args = f'{aggregate}'
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
        self.context_pooler = ContextPooler()
        self.aggregator = BatchTransformerContextAggregator(agg_fn=self.aggregate, 
                                                            add_dense=add_dense,
                                                            dims=dims,
                                                            activations=activations)
   

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
            name = f'{mtype}-{n_layers}layers'
        super(HierarchicalTransformerForContextMLM, self).__init__(name=name)
        
        # Some useful parameters
        self.n_tokens = n_tokens
        self.n_contexts = n_contexts
        self.output_signature = tf.float32
        
        # Define components
        config = transformer.config_class(vocab_size=vocab_size, n_layers=n_layers)
        mlm_model = transformer(config)
        self.embedder = mlm_model.layers[0]._layers[0]
        tlayers = mlm_model.layers[0]._layers[1].layer
        self.hier_layers = [HierarchicalTransformerBlock(tl,
                                                         self.n_contexts, 
                                                         self.n_tokens, 
                                                         config)
                            for tl in tlayers[:-1]]
        self.hier_last = HierarchicalTransformerBlock(tlayers[-1], 
                                                      self.n_contexts, 
                                                      self.n_tokens, 
                                                      config)
        self.mlm_head = MLMHead(mlm_model, reset=True)
        
        
    def _encode_batch(self, example):
        hidden_state = self.embedder(example['input_ids'])
        mask = example['attention_mask']
        for l in self.hier_layers:
            hidden_state = l(hidden_state, mask)
        hidden_state = self.hier_last(hidden_state, mask, True)
        return hidden_state
    
    
    def call(self, input):
        outs = tf.vectorized_map(self._encode_batch, 
                                 elems=input)
        logits = self.mlm_head(outs[:,0,:,:])
        return logits
    

class BiencoderForContextMLM(keras.Model):
    ''' Model class for masked language modeling using 
        a biencoder architecture
    Args:
        transformer: MLM model class from transformers library
        pretrained_token_encoder_weights (str): path to model initialization 
            weights or pretrained huggingface for token encoder
        trained_token_encoder_weights (str): path to token encoder weights
        trained_token_encoder_class (str): model class for token encoder
        name (str): identification string
        n_layers_token_encoder (int): number of transformer layers for 
            token encoder
        n_layers_context_encoder (int): number of transformer layers for 
            context encoder
        freeze_token_encoder (list, False, or None): which layers of the token 
            encoder to freeze
        add_dense (int): number of additional dense layers to add 
            between the encoder and the MLM head, after aggregating
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
                 pretrained_token_encoder_weights=None,
                 trained_token_encoder_weights=None,
                 trained_token_encoder_class=None,
                 name=None,
                 n_layers_token_encoder=3,
                 n_layers_context_encoder=3,
                 freeze_token_encoder=False,
                 add_dense=None,
                 dims=None,
                 activations=None,
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
            dense_args = f'{add_dense}dense-{dims_str}'
            ctx_args = f'{aggregate}'
            name = f'{mtype}-{freeze_str}-{n_layers_str}_layers-{dense_args}' \
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
                                                        dims=dims,
                                                        activations=activations)
        else:
            self.aggregator = BiencoderAttentionAggregator(include_head=True)

            
    def _encode_context(self, example):
        ctx = self.context_encoder(input_ids=example['input_ids'][1:,:], # bs x n_ctx x 512
                                   attention_mask=example['attention_mask'][1:,:]) # bs x n_ctx x 512
        return ctx.last_hidden_state
    
            
    def call(self, input):
        target = self.token_encoder(input_ids=input['input_ids'][:,0,:], # input: bs x 512
                                    attention_mask=input['attention_mask'][:,0,:]).last_hidden_state # input: bs x 512
        contexts = tf.vectorized_map(self._encode_context, elems=input)[:,:,0,:] # bs x n_ctx x 768
        if self.aggregate != 'attention':
            contexts = self.context_pooler(contexts, self.n_tokens)
        target = self.aggregator(target, contexts)
        logits = self.mlm_head(target)
        return logits

    
class BatchTransformerForMetrics(keras.Model):
    ''' Encodes and averages posts, predict aggregate 
    Args:
        transformer (transformers.Model): huggingface transformer
        weights (str): path to pretrained or init weights
        name (str): name
        metric_type (str): type of metric to predict. Should be 
            aggregate (if user-level metric) or single (if post-level
            metric)
        target (str): name of target metric within agg dataset
        add_dense (int): number of dense layers to add before classification
        dims (list): number of nodes per layer
        encoder_strainable (bool): if the encoder is trainable
    '''
    def __init__(self, 
                 transformer, 
                 weights=None,
                 name=None, 
                 metric_type='aggregate',
                 targets=['avg_posts'],
                 add_dense=0,
                 dims=None,
                 activations=None,
                 encoder_trainable=False):
        dims_str = '_'.join(dims) + '_dense' if dims else 'nodense'
        if dims:
            if len(dims) != add_dense:
                raise ValueError('dims should have add_dense values')
        if name is None:
            name = f'BatchTransformerForMetrics-{dims_str}'
        super(BatchTransformerForMetrics, self).__init__(name=name)
        self.encoder = transformer.from_pretrained(weights)
        self.encoder.trainable = encoder_trainable
        self.targets = targets
        self.output_signature = tf.float32
        if add_dense > 0:
            self.dense_layers = keras.Sequential([Dense(dims[i], 
                                                        activation=activations[i])
                                                  for i in range(add_dense)])
        else:
            self.dense_layers = None
        self.head = Dense(len(targets), activation='relu')
        self.metric_type = metric_type

    def _encode_batch(self, example):
        output = self.encoder(input_ids=example['input_ids'],
                              attention_mask=example['attention_mask'])
        encoding = output.last_hidden_state[:,0,:]
        return encoding

    def _aggregate(self, encodings):
        if self.metric_type == 'aggregate':
            return tf.reduce_mean(encodings, axis=1)
        elif self.metric_type == 'single':
            return encodings[:,0,:]

    def call(self, input):
        encodings = tf.vectorized_map(self._encode_batch, elems=input)
        out = self._aggregate(encodings)
        if self.dense_layers:
            out = self.dense_layers(out)
        out = self.head(out)
        return out

    