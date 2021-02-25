
import tensorflow as tf
from transformers import DistilBertTokenizer


def build_distilbert_input(weights, n_padded=2, n_batch=2, batched_only=False):
    with open('data/sample_posts.txt', 'r') as fh:
        posts = fh.readlines()
    posts = [p.strip('\n') for p in posts]
    n_posts_true = len(posts)
    tknz = DistilBertTokenizer.from_pretrained(weights)
    iids = tknz.batch_encode_plus(posts, padding=True)['input_ids']
    n_tokens = len(iids[0])
    iids += [[0]*n_tokens] * n_padded
    batch_iids = {'input_ids': tf.constant([iids] * n_batch)}
    iids = tf.constant(iids)
    if batched_only:
        return batch_iids, n_posts_true
    else:
        mask = tf.constant([1.] * n_posts_true + [0.] * n_padded)
        mask = tf.expand_dims(mask, 1)
        return batch_iids, iids, mask, n_posts_true