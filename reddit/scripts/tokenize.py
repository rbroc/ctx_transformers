import numpy as np
import pandas as pd
import timeit
from pathlib import Path
from transformers import AutoTokenizer
import argparse
import glob
import re

TRIPLET_PATH = Path('..') / 'data' / 'triplet'
triplet_files = glob.glob(str(TRIPLET_PATH / '*'))

parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer-weights', 
                    type=str, 
                    default='distilbert-base-uncased',
                    help='Weights of pretrained tokenizer')
parser.add_argument('--n-wds', type=int,
                    default=400, help='Number of words post '
                    'are truncated at before tokenization')
parser.add_argument('--batch-size', type=int,
                    default=100000, help='Number of '
                    'posts encoded at once')


def _tknz(x, tokenizer, n_wds):
    out = tokenizer.encode_plus(' '.join(x.split()[:n_wds]),
                                truncation=True, 
                                padding='max_length')
    return out


def tokenize(weights='distilbert-base-uncased', 
            n_wds=400, batch_size=100000):
    ''' Encodes all posts in the dataset using a given 
        huggingface tokenizer
    Args:
        weights ()
        tokenizer (Tokenizer): huggingface tokenizer object
        n_wds (int): number of words to truncate input at 
            (for efficiency,truncation would happen anyway
            at token level)
        batch_size (int): how many posts to encode at a 
            time
    '''
    tknzr = AutoTokenizer.from_pretrained(weights)
    for f in triplet_files:
        print(f'Reading {f}')
        df = pd.read_csv(f, sep='\t', compression='gzip')
        pid = list(np.arange(0, df.shape[0], batch_size)) 
        pid = pid.append(df.shape[0])
        start_t = timeit.default_timer() 
        current_t = start_t
        for i in range(len(pid) - 1):
            print(f'\t\tTimestamp previous step {current_t - start_t}')
            print(f'\tEncoding {pid[i]} to {pid[i+1]} of {df.shape[0]}')
            tknzd = df['selftext'][pid[i]:
                                   pid[i+1]].apply(lambda x: _tknz(x, 
                                                                   tknzr, 
                                                                   n_wds))
            tknzd = pd.DataFrame(tknzd)
            if i == 0:
                alltkn = tknzd
            else:
                alltkn = pd.concat([alltkn, 
                                    tknzd], 
                                    ignore_index=True)
            current_t = timeit.default_timer()
        for c in ['input_ids', 'attention_mask']:
            df[c] = alltkn['selftext'].apply(lambda x: x[c])
        outf = re.sub('\.txt', weights + '.txt', f)
        df.to_csv(outf, sep='\t', compression='gzip')
        

if __name__=="__main__":
    args = parser.parse_args()
    tokenize(args.tokenizer_weights, args.n_wds, args.batch_size)