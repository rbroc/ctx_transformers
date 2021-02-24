import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from reddit import BatchTransformer, BatchTransformerFFN
import pytest

def _build_distilbert_input(weights, n_padded=2, n_batch=2, batched_only=False):
    # Read in files
    with open('data/sample_posts.txt', 'r') as fh:
        posts = fh.readlines()
    posts = [p.strip('\n') for p in posts]
    n_posts_true = len(posts)
    tknz = DistilBertTokenizer.from_pretrained(weights)
    # Tokenize
    iids = tknz.batch_encode_plus(posts, padding=True)['input_ids']
    n_tokens = len(iids[0])
    # Append zeros and batch input
    iids += [[0]*n_tokens] * n_padded
    batch_iids = {'input_ids': tf.constant([iids] * n_batch)}
    iids = tf.constant(iids)
    if batched_only:
        return batch_iids, n_posts_true
    else:
        mask = tf.constant([1.] * n_posts_true + [0.] * n_padded)
        mask = tf.expand_dims(mask, 1)
        return batch_iids, iids, mask, n_posts_true


def test_batch_encoder():
    # Initialize model
    weights = 'distilbert-base-uncased'
    batch_model = BatchTransformer(TFDistilBertModel, weights)
    reference_model = TFDistilBertModel.from_pretrained(weights)
    # Create the inputs
    n_padded = 2
    n_batch = 2
    batch_iids, iids, mask, n_posts_true = _build_distilbert_input(weights, 
                                                                   n_padded, 
                                                                   n_batch)
    # Pass through batch network
    batch_encodings, n_posts = batch_model(batch_iids) # remove n_posts
    encodings = reference_model(iids).last_hidden_state[:,0,:]
    encodings = tf.multiply(encodings, mask)
    assert tf.reduce_all(tf.equal(n_posts, n_posts_true))
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
        # Catch error
        batch_iids, n_posts_true = _build_distilbert_input(weights, batched_only=True)
        encodings, n_posts = model(batch_iids)
        assert len(model.dense_layers.layers) == 2
        for idx, l in enumerate(model.dense_layers.layers):
            assert l.output_shape[1] == 3
            assert l.output_shape[-1] == model.dims[idx]
        assert encodings.shape == (2, 3, model.dims[-1])
    with pytest.raises(ValueError) as e:
        model = BatchTransformerFFN(TFDistilBertModel, weights,
                                    n_dense=2, dims=[100]*3, activations=['relu']*3)
        assert 'length of dims does not match ' in str(e)