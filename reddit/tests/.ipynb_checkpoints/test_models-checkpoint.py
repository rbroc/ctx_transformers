import tensorflow as tf
from transformers import TFDistilBertModel
from reddit import BatchTransformer, BatchTransformerFFN
import pytest
from utils import build_distilbert_input


def test_batch_encoder():
    # Initialize model
    weights = 'distilbert-base-uncased'
    batch_model = BatchTransformer(TFDistilBertModel, weights)
    reference_model = TFDistilBertModel.from_pretrained(weights)
    # Create the inputs
    n_padded = 2
    n_batch = 2
    batch_iids, mask, n_posts_true = build_distilbert_input(weights, 
                                                            n_padded, 
                                                            n_batch)
    # Pass through batch network
    batch_encodings = batch_model(batch_iids)
    encodings = reference_model(input_ids=batch_iids['input_ids'][0,:,:],
                                attention_mask=batch_iids['attention_mask'][0,:,:]).last_hidden_state[:,0,:]
    encodings = tf.multiply(encodings, mask)
    assert batch_encodings.shape == (n_batch,n_posts_true+n_padded,768)
    assert tf.reduce_all(tf.equal(batch_encodings[0,-2,:], 0))
    assert tf.reduce_all(tf.equal(batch_encodings[0,-1,:], 0))
    assert tf.reduce_all(tf.equal(batch_encodings[0,3,:], 0)) == False
    # Check differences
    diffs = tf.experimental.numpy.isclose(batch_encodings[0,:,:], 
                                          encodings, 
                                          atol=5e-7)
    assert tf.reduce_all(diffs)


def test_batch_encoder_ffn():
    weights = 'distilbert-base-uncased'
    dims = [200, [200,300]]
    activations = ['relu', ['relu', 'sigmoid']]
    for i in range(2):
        model = BatchTransformerFFN(TFDistilBertModel, weights,
                                    n_dense=2, dims=dims[i], activations=activations[i],
                                    name=f'test_model_{i}')
        batch_iids, _ = build_distilbert_input(weights, batched_only=True)
        encodings = model(batch_iids)
        assert len(model.dense_layers.layers) == 2
        for idx, l in enumerate(model.dense_layers.layers):
            assert l.output_shape[1] == 3
            assert l.output_shape[-1] == model.dims[idx]
        assert encodings.shape == (2, 3, model.dims[-1])
    with pytest.raises(ValueError) as e:
        model = BatchTransformerFFN(TFDistilBertModel, weights,
                                    n_dense=2, dims=[100]*3, 
                                    activations=['relu']*3)
        assert 'length of dims does not match ' in str(e)