import tensorflow as tf
from transformers import DistilBertTokenizer
from pathlib import Path


def build_distilbert_input(weights='distilbert-base-uncased', 
                           n_padded=2, n_batch=2, 
                           batched_only=False, kind='train'):
    ''' Builds one training example for Distilbert models 
    Args:
        weights (str): path to pretrained model weights
        n_padded (int): number of padding posts to add
        n_batch (int): batch dimensionality
        batched_only (bool): whether to return batched 
            input only (dict with input ids + attention mask  
            and number of posts before padding) or also mask 
            for non-padded/padded posts for the first item
            in the batch (used to check that dimensionality 
            and padding are handled correctly)
    '''
    fname = Path('data') / f'{kind}_posts.txt'
    with open(fname, 'r') as fh:
        posts = fh.readlines()
    posts = [p.strip('\n') for p in posts]
    n_posts_true = len(posts)
    tknz = DistilBertTokenizer.from_pretrained(weights)
    iids = tknz.batch_encode_plus(posts, padding=True)['input_ids']
    n_tokens = len(iids[0])
    iids += [[0]*n_tokens] * n_padded
    batch_iids = {'input_ids': tf.constant([iids]*n_batch),
                  'attention_mask': tf.cast(tf.constant([iids]*n_batch) != 0, 
                                            tf.int32),
                  'id': tf.cast(tf.random.poisson((n_batch,),lam=100), 
                                tf.int32)}
    iids = tf.constant(iids)
    if batched_only:
        return batch_iids, n_posts_true
    else:
        mask = tf.constant([1.] * n_posts_true + [0.] * n_padded)
        mask = tf.expand_dims(mask, 1)
        return batch_iids, mask, n_posts_true


def build_distilbert_multiple_input(n_examples=3, **kwargs):
    ''' Builds multiple training examples 
    Args:
        n_examples (int): number of training examples
        kwargs: keyword arguments for build_distilbert_input
        call
    '''
    ids, n = build_distilbert_input(**kwargs, 
                                    batched_only=True)
    ids = {'input_ids': tf.stack([ids['input_ids']] * n_examples),
           'attention_mask': tf.cast(tf.stack([ids['input_ids']] \
                                                * n_examples) != 0, 
                                     tf.int32),
           'id': tf.stack([ids['id']] * n_examples)}
    return ids, [n] * n_examples

    