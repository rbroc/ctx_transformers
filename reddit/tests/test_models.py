import tensorflow as tf
from transformers import TFDistilBertModel, DistilBertTokenizer
from reddit import BatchTransformer
import pytest


def test_batch_encoder():
    # Read the examples
    with open('data/sample_posts.txt', 'r') as fh:
        posts = fh.readlines()
    posts = [p.strip('\n') for p in posts]
    n_posts_true = len(posts)

    # Initialize model
    weights = 'distilbert-base-uncased'
    batch_model = BatchTransformer(TFDistilBertModel, weights)
    reference_model = TFDistilBertModel.from_pretrained(weights)

    # Tokenize
    tknz = DistilBertTokenizer.from_pretrained(weights)
    iids = tknz.batch_encode_plus(posts, padding=True)['input_ids']
    n_tokens = len(iids[0])

    # Append zeros and batch input
    n_padded = 2
    n_batch = 2
    iids += [[0]*n_tokens] * n_padded
    batch_iids = tf.constant([iids] * n_batch)
    iids = tf.constant(iids)
    mask = tf.constant([1.] * n_posts_true + [0.] * n_padded)
    mask = tf.expand_dims(mask, 1)

    # Pass through batch network
    batch_encodings, n_posts = batch_model(batch_iids)
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