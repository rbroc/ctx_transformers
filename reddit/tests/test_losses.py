import tensorflow as tf
from reddit.losses import TripletLossBase, TripletLossFFN
from reddit.models import BatchTransformer, BatchTransformerFFN
from utils import build_distilbert_input
import pytest
from transformers import TFDistilBertModel

WEIGHTS = 'distilbert-base-uncased'


def test_triplet_loss_base():
    batch_iids, _ = build_distilbert_input(WEIGHTS, batched_only=True)
    model = BatchTransformer(TFDistilBertModel, WEIGHTS)
    encodings, n_posts = model(batch_iids)
    loss = TripletLossBase(1)
    loss_outs = loss(encodings, n_posts)
    n,p,a = encodings[:,0,:], encodings[:,1,:], encodings[:,2:,:]
    a_avg = tf.reduce_sum(a, axis=1) / 3
    dpos = tf.reduce_sum(tf.square(p-a_avg), axis=-1)
    dneg = tf.reduce_sum(tf.square(n-a_avg), axis=-1)
    dpos_avg = tf.reduce_mean(dpos)
    dneg_avg = tf.reduce_mean(dneg)
    danchs = tf.stack([tf.reduce_sum(tf.square(a[:,0,:]-a[:,1,:]), axis=-1),
                       tf.reduce_sum(tf.square(a[:,0,:]-a[:,2,:]), axis=-1),
                       tf.reduce_sum(tf.square(a[:,1,:]-a[:,2,:]), axis=-1)],
                     axis=-1)
    danch = tf.reduce_mean(tf.reduce_mean(danchs, axis=1), axis=0)
    assert tf.experimental.numpy.isclose(dpos_avg, loss_outs[2], atol=5e-07)
    assert tf.experimental.numpy.isclose(dneg_avg, loss_outs[3], atol=5e-07)
    assert tf.experimental.numpy.isclose(danch, loss_outs[4], atol=5e-07)
    assert tf.equal(loss_outs[1],
                    tf.reduce_mean(tf.cast(tf.greater(dneg, dpos), tf.float32)))
    assert tf.equal(loss_outs[0], 
                    tf.reduce_mean(tf.maximum(0.0, 1 + (dpos - dneg))))


def test_triplet_loss_ffn():
    model = BatchTransformerFFN(TFDistilBertModel, WEIGHTS,
                               n_dense=2, dims=256, activations='relu')
    batch_iids, _ = build_distilbert_input(WEIGHTS, batched_only=True)
    encodings, n_posts = model(batch_iids)
    loss = TripletLossFFN(1)
    loss_outs = loss(encodings, n_posts)
    n,p,a = encodings[:,0,:], encodings[:,1,:], encodings[:,2,:]
    dpos = tf.reduce_sum(tf.square(p-a), axis=-1)
    dneg = tf.reduce_sum(tf.square(n-a), axis=-1)
    dpos_avg = tf.reduce_mean(dpos)
    dneg_avg = tf.reduce_mean(dneg)
    # Check distances
    assert tf.experimental.numpy.isclose(dpos_avg, loss_outs[2], atol=5e-07)
    assert tf.experimental.numpy.isclose(dneg_avg, loss_outs[3], atol=5e-07)
    # Check metric and loss
    assert tf.equal(loss_outs[1],
                    tf.reduce_mean(tf.cast(tf.greater(dneg, dpos), tf.float32)))
    assert tf.equal(loss_outs[0], 
                    tf.reduce_mean(tf.maximum(0.0, 1 + (dpos - dneg))))