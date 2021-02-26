import tensorflow as tf
from transformers import DistilBertTokenizer
from pathlib import Path


def build_distilbert_input(weights, n_padded=2, n_batch=2, 
                           batched_only=False, type='train'):
    ''' Builds one training example for Distilbert models '''
    fname = Path('data') / f'{type}_posts.txt'
    with open(fname, 'r') as fh:
        posts = fh.readlines()
    posts = [p.strip('\n') for p in posts]
    n_posts_true = len(posts)
    tknz = DistilBertTokenizer.from_pretrained(weights)
    iids = tknz.batch_encode_plus(posts, padding=True)['input_ids']
    n_tokens = len(iids[0])
    iids += [[0]*n_tokens] * n_padded
    batch_iids = {'input_ids': tf.constant([iids] * n_batch),
                  'id': tf.cast(tf.random.poisson((n_batch,),lam=100), 
                                tf.int32)}
    iids = tf.constant(iids)
    if batched_only:
        return batch_iids, n_posts_true
    else:
        mask = tf.constant([1.] * n_posts_true + [0.] * n_padded)
        mask = tf.expand_dims(mask, 1)
        return batch_iids, iids, mask, n_posts_true


def build_distilbert_multiple_input(n_examples=3, **kwargs):
    ids, n = build_distilbert_input(**kwargs, 
                                    batched_only=True)
    ids = {'input_ids': tf.stack([ids['input_ids']] * n_examples),
           'id': tf.stack([ids['id']] * n_examples)}
    return ids, [n] * n_examples

    