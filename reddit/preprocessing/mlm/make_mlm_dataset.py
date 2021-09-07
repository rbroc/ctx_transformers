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


def _generate_example(d, tknzr, is_context=True):
    ''' Generator for TFDataset 
    Args:
        d (list): list of example dictionaries to generate 
                  examples from 
        tknzr (transformers.Tokenizer): Tokenizer
        is_context (bool): whether any context is passed
            or it's a target-only dataset for simple mlm
    '''
    if is_context:
        for di in d:
            trgt, ctxs = [_tknz(di[k], tknzr) for k in ['target', 'context']]
            tids, cids = [x['input_ids'] for x in [trgt, ctxs]]
            tmask, cmask = [x['attention_mask'] for x in [trgt, ctxs]]
            yield (tids, tmask, 
                   cids, cmask,
                   di['example_id'],
                   tknzr.mask_token_id)
    else:
        for di in d:
            trgt = _tknz(di['target'], tknzr)
            tids = trgt['input_ids']
            tmask = trgt['attention_mask']
            yield (tids, tmask, di['example_id'],
                   tknzr.mask_token_id)


def make_dataset(f, outpath, tknzr, g_t, n_shards=1):
    if g_t != 'single':
        nvars = 6
        is_context = True
        ds_type = 'mlm'
    else:
        nvars = 4
        is_context = False
        ds_type = 'mlm_simple'
    fid = f.split('/')[-1].split('.')[0]
    d = json.load(gzip.open(f))
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(d, 
                                                                  tknzr, 
                                                                  is_context),
                                        output_types=tuple([tf.int32]*nvars))
    save_tfrecord(ds, fid, outpath, n_shards=n_shards, ds_type=ds_type)
    
if __name__=='__main__':
    args = parser.parse_args()
    tknzr = AutoTokenizer.from_pretrained(args.tokenizer_weights)
    g_types = ['author', 'random', 'subreddit', 'single']
    splits = ['train', 'test']
    for g_t in g_types:
        for s in splits:
            JSON_PATH = MLM_JSON_PATH / str(args.dataset_name) / g_t / s
            OUTPATH = MLM_DS_PATH / str(args.dataset_name) / g_t / s
            OUTPATH.mkdir(exist_ok=True, parents=True)
            fs = glob.glob(str(JSON_PATH / '*'))
            pool = Pool(processes=args.n_cores)
            pool.starmap(make_dataset, zip(fs,
                                           [OUTPATH]*len(fs),
                                           [tknzr]*len(fs),
                                           [g_t]*len(fs),
                                           [args.n_shards]*len(fs)))
            pool.close()