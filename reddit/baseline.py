import numpy as np
import pandas as pd
from tools.datasets import load_tfrecord_triplet_nn1
import tensorflow as tf
import glob
from transformers import TFDistilBertModel

n_posts = 50
margins = [10 * 10**(x) for x in range(-3,3)]
fnames_triplet = glob.glob('datasets/example/triplet_nn1/*')
ds_triplet = load_tfrecord_triplet_nn1(filenames=fnames_triplet, compression_type='GZIP', 
                                       deterministic=True)
model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

def preprocess(x):
    return tf.reverse(x['input_ids'], [0])[:n_posts,:]

def compute_encodings(x):
    return tf.squeeze(model(x).last_hidden_state[:,0,:])

def euclidean_distance(x):
    n_enc = x[0]
    p_enc = x[1]
    a_enc = tf.reduce_mean(x[2:], axis=0)
    d_pos = tf.reduce_sum(tf.square(a_enc-p_enc))
    d_neg = tf.reduce_sum(tf.square(a_enc-n_enc))
    return (d_pos, d_neg)

def compute_loss(d_pos, d_neg, margin):
    return tf.maximum(0.0, margin + d_pos - d_neg)


def compute_baseline():
    pdist = []
    ndist = []
    losses = []
    e = 0
    for step in ds_triplet:
        e += 1
        #if e % 10 == 0:
        print(f'Processing example: {e}')
        r = euclidean_distance(compute_encodings(preprocess(step)))
        loss = [compute_loss(r[0], r[1], m) for m in margins]
        pdist.append(r[0].numpy())
        ndist.append(r[1].numpy())
        losses.append(tuple([l.numpy() for l in loss]))

    df = pd.DataFrame(zip(pdist, ndist, *zip(*losses)))
    df.columns = ['pdist', 'ndist'] + [f'loss_m_{str(m)}' for m in margins]
    df.to_csv('misc/triplet_pretrained_metrics_50posts.txt', 
              sep='\t', compression='gzip')


if __name__=="__main__":
    compute_baseline()