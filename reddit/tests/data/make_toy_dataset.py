from reddit.utils import save_tfrecord
from pathlib import Path
import json
import tensorflow as tf
import argparse
import glob
from transformers import AutoTokenizer
import gzip

TRIPLET_PATH = Path('..') / '..' / 'data' / 'triplet'
DATASET_PATH = Path('.')

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer-weights', type=str, 
                    default='distilbert-base-uncased',
                    help='Weights of pretrained tokenizer')
parser.add_argument('--n-examples', type=int, 
                    default=100,
                    help='Number of examples (max 10000)')

def _tknz(x, tokenizer):
    ''' Tokenize single user '''
    out = tokenizer.batch_encode_plus(x,
                                      truncation=True, 
                                      padding='max_length')
    return out


def _generate_example(d, tknzr, n_examples):
    ''' Generator for TFDataset '''
    for adict in d[:n_examples]:
        anchor, pos, neg = [_tknz(adict[e], tknzr) 
                            for e in ['anchor', 'positive', 'negative']]
        ids, pids, nids = [x['input_ids'] for x in [anchor, pos, neg]]
        mask, pmask, nmask = [x['attention_mask'] for x in [anchor, pos, neg]]
        yield (ids, mask, 
               pids, pmask, 
               nids, nmask, 
               adict['author_id'])


def make_sample_dataset(f, tknzr, n_examples):
    ''' Create dataset and save as tfrecord'''
    print(f'Processing {f}...')
    fid = 'sample_dataset'
    d = json.load(gzip.open(f))
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(d, 
                                                                  tknzr, 
                                                                  n_examples),
                                        output_types=tuple([tf.int32]*7))
    save_tfrecord(ds, fid, DATASET_PATH, n_shards=1)


if __name__=="__main__":
    args = parser.parse_args()
    tknzr = AutoTokenizer.from_pretrained(args.tokenizer_weights)
    n_examples = args.n_examples
    make_sample_dataset(glob.glob(str(TRIPLET_PATH/'*'))[0],
                        tknzr, n_examples)

