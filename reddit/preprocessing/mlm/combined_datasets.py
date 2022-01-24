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
MLM_JSON_PATH = DATA_PATH / 'json' / 'mlm'
MLM_DS_PATH = DATA_PATH / 'datasets' / 'mlm'


def _tknz(x, tokenizer):
    ''' Tokenize single user '''
    out = tokenizer.batch_encode_plus(x,
                                      truncation=True, 
                                      padding='max_length')
    return out


def _generate_example(d, tknzr, g_t):
    ''' Generator for TFDataset 
    Args:
        d (list): list of example dictionaries to generate 
                  examples from 
        tknzr (transformers.Tokenizer): Tokenizer
    '''
    if g_t == 'author':
        hm = tf.constant([1,1,1,0,0,0,1,1,1,1,1,1])
    elif g_t == 'subreddit':
        hm = tf.constant([0,0,0,1,1,1,1,1,1,1,1,1])
    else:
        hm = tf.constant([0,0,0,0,0,0,1,1,1,1,1,1]) # for 12 heads
    hm = tf.reshape(hm, [12,1,1])
    for di in d:
        trgt, ctxs = [_tknz(di[k], tknzr) for k in ['target', 'context']]
        tids, cids = [x['input_ids'] for x in [trgt, ctxs]]
        tmask, cmask = [x['attention_mask'] for x in [trgt, ctxs]]
        yield (tids, tmask, 
               cids, cmask,
               di['example_id'],
               hm, tknzr.mask_token_id)

def make_dataset(f, outpath, tknzr, g_t, add, n_shards=1):
    nvars = 7
    ds_type = 'mlm_combined'
    fid = f.split('/')[-1].split('.')[0]
    fid_1, fid_2 = fid.split('_')
    fid = fid_1 + '_' + str(int(fid_2)+add)
    d = json.load(gzip.open(f))
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(d, 
                                                                  tknzr,
                                                                  g_t),
                                        output_types=tuple([tf.int32]*nvars))
    save_tfrecord(ds, fid, outpath, n_shards=n_shards, ds_type=ds_type)
    
if __name__=='__main__':
    args = parser.parse_args()
    tknzr = AutoTokenizer.from_pretrained(args.tokenizer_weights)
    g_types = ['author', 'random', 'subreddit']
    splits = ['train', 'test']
    pool = Pool(processes=args.n_cores)
    for i, g_t in enumerate(g_types):
        for s in splits:
            if s == 'train':
                add = i*200
            else:
                add = i*10
            JSON_PATH = MLM_JSON_PATH / str(args.dataset_name) / g_t / s
            OUTPATH = MLM_DS_PATH / str(args.dataset_name) / 'combined_alt' / s
            OUTPATH.mkdir(exist_ok=True, parents=True)
            fs = glob.glob(str(JSON_PATH / '*'))
            pool.starmap(make_dataset, zip(fs,
                                           [OUTPATH]*len(fs),
                                           [tknzr]*len(fs),
                                           [g_t]*len(fs),
                                           [add]*len(fs),
                                           [args.n_shards]*len(fs)))
            #pool.close()