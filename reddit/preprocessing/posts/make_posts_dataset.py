import numpy as np
from pathlib import Path
import argparse
import glob
import gzip
import json
from reddit.utils import save_tfrecord
from transformers import AutoTokenizer
from multiprocessing import Pool
import tensorflow as tf
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser()
parser.add_argument('--n-cores', type=int, default=10,
                    help='Number of parallel calls to main loop')
parser.add_argument('--tokenizer-weights', type=str, 
                    default='distilbert-base-uncased',
                    help='Weights of pretrained tokenizer')
parser.add_argument('--n-shards', type=int, default=1, 
                    help='Number of shards for each batch of users')
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Dataset name for output path')


DATA_PATH = Path('..') / '..' / 'data'
POSTS_JSON_PATH = DATA_PATH / 'json' / 'posts'
POSTS_DS_PATH = DATA_PATH / 'datasets' / 'posts'


def _tknz(x, tokenizer):
    ''' Tokenize single user '''
    out = tokenizer.batch_encode_plus(x,
                                      truncation=True, 
                                      padding='max_length')
    return out


def _generate_example(d, tknzr):
    ''' Generator for TFDataset 
    Args:
        d (list): list of example dictionaries to generate 
                  examples from 
        tknzr (transformers.Tokenizer): Tokenizer
    '''
    for di in d:
        posts = _tknz([di['text']], tknzr)
        iids = posts['input_ids']
        amask = posts['attention_mask']
        score = di['score']
        comm = di['comments']
        ids = di['id']
        yield (iids, amask, 
               comm, score, 
               ids)


def make_dataset(f, outpath, tknzr, n_shards=1):
    fid = f.split('/')[-1].split('.')[0]
    d = json.load(gzip.open(f))
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(d, 
                                                                  tknzr),
                                        output_types=(tf.int32, tf.int32, 
                                                      tf.float32, tf.float32, tf.int32))
    save_tfrecord(ds, fid, outpath, n_shards=n_shards, ds_type='posts')
     

if __name__=='__main__':
    args = parser.parse_args()
    tknzr = AutoTokenizer.from_pretrained(args.tokenizer_weights)
    splits = ['train', 'test']
    for s in splits:
        JSON_PATH = POSTS_JSON_PATH / str(args.dataset_name) / s
        OUTPATH = POSTS_DS_PATH / str(args.dataset_name) / s
        OUTPATH.mkdir(exist_ok=True, parents=True)
        fs = glob.glob(str(JSON_PATH / '*'))
        pool = Pool(processes=args.n_cores)
        pool.starmap(make_dataset, zip(fs,
                                       [OUTPATH]*len(fs),
                                       [tknzr]*len(fs),
                                       [args.n_shards]*len(fs)))
        pool.close()