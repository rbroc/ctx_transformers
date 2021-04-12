import tensorflow as tf
from reddit.losses import TripletLossBase, TripletLossFFN
from reddit.models import BatchTransformer, BatchTransformerFFN
from reddit.utils import load_tfrecord, pad_and_stack
from utils import build_distilbert_input
import pytest
from transformers import TFDistilBertModel
from data.utils import duplicate_examples
import glob
from pathlib import Path


WEIGHTS = 'distilbert-base-uncased'
ds_files = glob.glob(str(Path('data')/'sample_dataset*'))

def test_triplet_loss_base():
    ''' Tests triplet loss function for Distilbert fine-tuning 
        with single positive and negative example
      '''
    batch_iids, _ = build_distilbert_input(WEIGHTS, batched_only=True)
    model = BatchTransformer(TFDistilBertModel, WEIGHTS)
    encodings = model(batch_iids)
    loss = TripletLossBase(1)
    loss_outs = loss(encodings)
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
    ''' Tests triplet loss function for FFN approach
        with single positive and negative example
    '''
    model = BatchTransformerFFN(TFDistilBertModel, WEIGHTS,
                                n_dense=2, dims=256, activations='relu')
    batch_iids, _ = build_distilbert_input(WEIGHTS, batched_only=True)
    encodings = model(batch_iids)
    loss = TripletLossFFN(1)
    loss_outs = loss(encodings)
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


def test_multiple_posts():
    ''' Test that loss works with multiple negative and positive examples '''
    loss = TripletLossBase(1, n_pos=2, n_neg=2)
    model = BatchTransformer(TFDistilBertModel, WEIGHTS, trainable=False)
    # Process dataset
    ds = duplicate_examples(load_tfrecord(ds_files))
    ds = pad_and_stack(ds, pad_to=[5,2,2]).take(2).batch(2)
    for example in ds:
      outs = model(example)
    loss_outs = loss(outs)
    # Compute manual
    negs, poss, anchs = outs[:,:2,:], outs[:,2:4,:], outs[:,4:,:]
    aneg = tf.reduce_mean(negs, axis=1)
    apos = tf.reduce_mean(poss, axis=1)
    aanch = tf.reduce_mean(anchs, axis=1)
    # Compute loss
    dpos = tf.reduce_sum(tf.square(apos-aanch), axis=-1)
    dneg = tf.reduce_sum(tf.square(aneg-aanch), axis=-1)
    dpos_avg = tf.reduce_mean(dpos)
    dneg_avg = tf.reduce_mean(dneg)
    ref_loss = tf.maximum(0.0, 1 + (dpos-dneg))
    ref_loss_avg = tf.reduce_mean(ref_loss)
    # Check all
    assert ref_loss_avg == loss_outs[0]
    assert dpos_avg == loss_outs[2]
    assert dneg_avg == loss_outs[3]