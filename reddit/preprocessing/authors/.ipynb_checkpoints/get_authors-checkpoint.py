import sqlite3
import pandas as pd
from pathlib import Path
from datetime import date
import gzip
import json
import argparse
from reddit.utils import stringify
import numpy as np
from sklearn.preprocessing import quantile_transform, MinMaxScaler


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Number of examples per output file')
parser.add_argument('--perc-train', type=float, default=.9,
                    help='Percentage of authors in training set')
parser.add_argument('--perc-test', type=float, default=.1,
                    help='Percentage of authors in test set')


DATA_PATH = Path('..') / '..' / 'data'
DB_PATH = DATA_PATH / 'databases'
META_PATH = DATA_PATH / 'meta'
AGG_PATH = DATA_PATH / 'json' / 'authors'


conn = sqlite3.connect(str(DB_PATH / 'posts.db'))
cur = conn.cursor()


def _zip_dicts(ds):
    ''' Merges dictionaries with common keys'''
    for idx, k in enumerate(ds[0].keys()):
        
        ad = {'author': k,
              'author_id': idx}
        for d in ds:
            ad.update(d[k])
        yield ad
    
    
def _sample_authors(author_list):
    ''' Sample n posts from a certain author
    Args:
        author_list (list): list with author ids
        n (int): how many context posts to sample for each author
    '''
    ds = {}
    for idx, author in enumerate(author_list):
        if (idx + 1) % 10000 == 0:
            print(f'Sampling {idx + 1}')
        cur.execute(f''' SELECT subreddit, selftext 
                         FROM posts
                         WHERE author = '{author}'
                     ''')     
        res = cur.fetchall()
        subs = [r[0] for r in res]
        stext = [r[1] for r in res]
        ds[author] = {'author_id': idx,
                      'subreddits': subs,
                      'posts': stext}
    return ds


def _save_example_dicts(d, outpath, batch_size=10000):
    ''' Takes list of example dictionaries and stores in 
        json files with batch_size examples per file
    Args:
        d (lst): list of dictionary examples
        outpath (pathlib.Path): output path for json files
        size (int): number of examples in the dataset
        batch_size (int): number of examples per file
    '''
    idxs = list(np.arange(0, len(d), batch_size)) + [len(d)]
    for nidx in range(len(idxs[:-1])):
        ofile = outpath / f'batch_{nidx+1}.json.gz'
        with gzip.open(ofile, 'w') as fh:
            fh.write(json.dumps(d[idxs[nidx]:idxs[nidx+1]]).encode('utf-8'))


def make_aggregate_dataset(batch_size=10000, 
                           perc_train=.9, 
                           perc_test=.1):
    ''' Create datasets with n_posts posts per author
    Args:
        batch_size (int): how many posts per output file
        perc_train (float): percentage of authors in training set
        perc_train (float): percentage of authors in test set
    '''
    print('*** Splitting train/test authors ***')
    cur.execute('''SELECT DISTINCT(author) FROM posts
                   ORDER BY RANDOM()''')
    auths = [i[0] for i in cur.fetchall()]
    tr_auths = auths[:int(len(auths)*perc_train)]
    ts_auths = auths[int(len(auths)*perc_train):
                     int(len(auths)*perc_train)+int(len(auths)*perc_test)]
    
    print('*** Fetching reference posts per author ***')
    tr_posts = _sample_authors(tr_auths)
    ts_posts = _sample_authors(ts_auths)
    
    print('*** Saving json files ***')
    for split, gen in zip(['train', 'test'], 
                          [tr_posts, ts_posts]):
        lst = [e for e in gen]
        OUTPATH = AGG_PATH / 'all' / split
        OUTPATH.mkdir(exist_ok=True, parents=True)
        _save_example_dicts(lst, OUTPATH, batch_size)
    
    
if __name__=='__main__':
    args = parser.parse_args()
    make_aggregate_dataset(args.batch_size,
                           args.perc_train, 
                           args.perc_test)
    conn.close()