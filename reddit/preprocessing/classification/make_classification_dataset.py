from reddit.utils import save_tfrecord
from pathlib import Path
import json
import tensorflow as tf
import argparse
import glob
from multiprocessing import Pool
from transformers import AutoTokenizer
import gzip
import os
import random

TRIPLET_PATH = Path('..') / '..' / 'data' / 'json' / 'triplet'
DATASET_PATH = Path('..') / '..' / 'data' / 'datasets' / 'classification'

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
parser.add_argument('--n-posts', type=int, default=3,
                    help='Number of posts')
parser.add_argument('--out-dataset-name', type=str, default=None,
                    help='Output dataset name')


def _tknz(x, tokenizer):
    ''' Tokenize single user '''
    out = tokenizer.batch_encode_plus(x,
                                      truncation=True, 
                                      padding='max_length')
    return out


def _generate_example(d, tknzr, n_posts):
    ''' Generator for TFDataset '''
    for adict in d:
        label = tf.constant(random.sample([0,1], k=1))[0] # may have to not slice
        # get random item
        #n_anch = random.sample(list(range(len(adict['anchor']))), 1)[0]
        p_1 = _tknz(adict['anchor'][:n_posts], tknzr) 
        if label == 1:
            p_2 = _tknz(adict['positive'], tknzr)
        elif label == 0:
            p_2 = _tknz(adict['negative'], tknzr)
        ids_1, ids_2 = [x['input_ids'] for x in [p_1, p_2]]
        mask_1, mask_2 = [x['attention_mask'] for x in [p_1, p_2]]
        yield (ids_1, mask_1, 
               ids_2, mask_2,
               [label],
               adict['author_id'])


def make_dataset(f, outpath, tknzr, n_posts, n_shards=1):
    ''' Create dataset and save as tfrecord'''
    print(f'Processing {f}...')
    fid = f.split('/')[-1].split('.')[0]
    d = json.load(gzip.open(f))
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(d, 
                                                                  tknzr, 
                                                                  n_posts),
                                        output_types=tuple([tf.int32]*6))
    save_tfrecord(ds, fid, outpath, n_shards=n_shards, ds_type='classification')


if __name__=="__main__":
    args = parser.parse_args()
    for s in ['train', 'test']:
        OUTPATH = DATASET_PATH / str(args.out_dataset_name) / s
        OUTPATH.mkdir(exist_ok=True, parents=True)
        fs = glob.glob(str(TRIPLET_PATH / args.dataset_name / s / '*'))
        tknzr = AutoTokenizer.from_pretrained(args.tokenizer_weights)
        pool = Pool(processes=args.n_cores)
        pool.starmap(make_dataset, zip(fs,
                                       [OUTPATH]*len(fs),
                                       [tknzr]*len(fs),
                                       [args.n_posts]*len(fs),
                                       [args.n_shards]*len(fs)))
        pool.close()
